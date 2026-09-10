// Exact immutable FP8 row retrieval; bounded direct-mapped RAM cache.
// No CUDA calls in the callback: suitable for cudaLaunchHostFunc graph nodes.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Store {
  int fd;
  uint64_t rows, weight_offset, scale_offset, slots, row_lo, row_hi;
  uint8_t *cache;
  uint64_t *keys;
  uint8_t *resident = nullptr;
  size_t resident_size = 0;
  std::mutex locks[256];
  std::atomic<uint64_t> hits{0}, misses{0}, reads{0};
};
struct Work {
  Store *store;
  const int64_t *ids;
  uint8_t *weights, *scales;
  uint64_t count;
};

static void fail(const char *reason) {
  std::fprintf(stderr, "Engram retrieval failed: %s (errno=%d)\n", reason, errno);
  std::abort(); // Never allow a generation to continue with missing/stale rows.
}

static void read_bytes(Store *s, uint64_t offset, uint8_t *out, size_t length) {
  if (s->resident) {
    std::memcpy(out, s->resident + offset, length);
    return;
  }
  alignas(4096) uint8_t page[8192];
  const uint64_t base = offset & ~uint64_t(4095);
  const size_t delta = offset - base;
  const size_t requested = ((delta + length + 4095) / 4096) * 4096;
  ssize_t got;
  do { got = pread(s->fd, page, requested, base); } while (got < 0 && errno == EINTR);
  if (got < 0 || size_t(got) < delta + length) fail("short or failed direct read");
  std::memcpy(out, page + delta, length);
  s->reads.fetch_add(1, std::memory_order_relaxed);
}

extern "C" Store *row_store_open(const char *path, uint64_t rows,
                                 uint64_t woff, uint64_t soff, uint64_t budget) {
  auto *s = new Store;
  const char *mode = std::getenv("OFFLOAD_MODE");
  const bool ram = mode && std::strcmp(mode, "ram") == 0;
  s->fd = open(path, O_RDONLY | O_CLOEXEC | (ram ? 0 : O_DIRECT));
  if (s->fd < 0) { delete s; return nullptr; }
  struct stat statbuf;
  if (fstat(s->fd, &statbuf) || woff > uint64_t(statbuf.st_size) ||
      soff > uint64_t(statbuf.st_size) || rows > (uint64_t(statbuf.st_size)-woff)/256 ||
      rows > (uint64_t(statbuf.st_size)-soff)/8) fail("invalid table extent");
  if (ram) {
    s->resident_size = statbuf.st_size;
    s->resident = static_cast<uint8_t *>(mmap(nullptr,s->resident_size,
        PROT_READ,MAP_SHARED | MAP_POPULATE,s->fd,0));
    if (s->resident == MAP_FAILED) fail("resident mapping");
    if (mlock(s->resident,s->resident_size)) fail("RAM mode requires memlock capability and sufficient RAM");
    budget = 0;
  }
  s->rows = rows; s->weight_offset = woff; s->scale_offset = soff;
  s->row_lo = 0; s->row_hi = rows;
  s->slots = budget / (264 + sizeof(uint64_t));
  s->cache = nullptr; s->keys = nullptr;
  if (s->slots) {
    s->cache = static_cast<uint8_t *>(mmap(nullptr, s->slots * 264,
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    s->keys = static_cast<uint64_t *>(mmap(nullptr, s->slots * sizeof(uint64_t),
        PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (s->cache == MAP_FAILED || s->keys == MAP_FAILED) fail("cache allocation");
  }
  return s;
}

extern "C" void row_store_lookup(void *opaque) {
  auto *work = static_cast<Work *>(opaque);
  Store *s = work->store;
  for (uint64_t i = 0; i < work->count; ++i) {
    const int64_t id = work->ids[i];
    if (id < 0 || uint64_t(id) >= s->rows) fail("row ID out of bounds");
    if (uint64_t(id) < s->row_lo || uint64_t(id) >= s->row_hi) {
      std::memset(work->weights + i * 256, 0, 256);
      std::memset(work->scales + i * 8, 0, 8);
      continue;
    }
    uint8_t row[264];
    const uint64_t slot = s->slots ? uint64_t(id) % s->slots : 0;
    std::unique_lock<std::mutex> guard(s->locks[slot % 256]);
    if (s->slots && s->keys[slot] == uint64_t(id) + 1) {
      std::memcpy(row, s->cache + slot * 264, 264);
      ++s->hits;
    } else {
      read_bytes(s, s->weight_offset + uint64_t(id) * 256, row, 256);
      read_bytes(s, s->scale_offset + uint64_t(id) * 8, row + 256, 8);
      ++s->misses;
      if (s->slots) {
        std::memcpy(s->cache + slot * 264, row, 264);
        s->keys[slot] = uint64_t(id) + 1;
      }
    }
    guard.unlock();
    std::memcpy(work->weights + i * 256, row, 256);
    std::memcpy(work->scales + i * 8, row + 256, 8);
  }
}

extern "C" void row_store_stats(Store *s, uint64_t *out) {
  out[0] = s->hits.load(); out[1] = s->misses.load(); out[2] = s->reads.load();
  out[3] = s->slots * 272;
}
extern "C" void row_store_range(Store *s, uint64_t lo, uint64_t hi) {
  if (lo > hi || hi > s->rows) fail("invalid row ownership range");
  s->row_lo = lo; s->row_hi = hi;
}
extern "C" void row_store_close(Store *s) {
  if (s->resident) munmap(s->resident,s->resident_size);
  if (s->slots) {
    munmap(s->cache, s->slots * 264);
    munmap(s->keys, s->slots * sizeof(uint64_t));
  }
  close(s->fd); delete s;
}
