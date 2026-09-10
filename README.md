# DeepSeek V4.1 Flash on 4× RTX PRO 6000 Blackwell

Docker deployment for **four 96 GB RTX PRO 6000 Blackwell GPUs**, with native checkpoint precision, DSpark speculative decoding, native vision, and Engram offload to either NVMe or locked host RAM.

**Experimental, with real inference evidence.** The underlying NVMe serving configuration has completed a 400,000-token input and six concurrent 350,000-token inputs. The distributable Docker wrapper is being qualified separately; do not confuse these runtime results with full release acceptance. Full-model RAM-mode performance and quality remain unverified.

## What it does

`docker compose up --build` downloads the pinned checkpoint, verifies every file against Hugging Face's SHA256/Git-blob hashes, compiles the storage adapter, starts TP4/EP4 inference, and requires a fresh arithmetic completion before reporting ready. Download and kernel caches persist across restarts. It never deletes other models or stops other containers.

The backbone and compressed KV cache stay on the GPUs. Engram table lookups preserve the original FP8 bytes and scale bytes; hashing, gating, projections, expert weights and tensor-parallel reduction retain their model behavior. This is **not an EXL3 conversion**. The source mixes FP4 routed experts, FP8/BF16 components, and FP8 Engram tables.

## Requirements

- Linux x86-64, NVIDIA driver compatible with CUDA 13.0, Docker Engine with Compose and NVIDIA Container Toolkit.
- Exactly four visible 96 GB RTX PRO 6000 Blackwell GPUs, available for this model.
- Approximately 510 GB for the checkpoint, plus Docker layers, caches and download headroom. Use a local NVMe filesystem supporting direct I/O for NVMe mode.
- **NVMe mode:** tested on a 128 GB host, using a bounded 64 GiB RAM row-cache budget and exact NVMe misses.
- **RAM mode:** approximately 189 GiB of Engram shard storage plus at least 24 GiB of available headroom; a 256 GiB or larger host is the practical starting point. Startup checks actual available memory. All mapped table pages are locked with `mlock`; a failed lock aborts startup. Shared read-only mappings avoid four physical copies.

The container does not change GPU power limits, fans, clocks, or CPU policy. Measurements below used 275 W per GPU and PCIe connectivity, without NVLink.

## Start

```bash
git clone https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000.git
cd deepseek-v4.1-flash-4x-rtx-pro-6000
mkdir -p models state
docker compose up --build -d
docker compose logs -f
```

If you already downloaded this checkpoint, reuse its directory:

```bash
MODEL_DIRECTORY=/absolute/path/to/DeepSeek-V4.1-Flash docker compose up --build -d
```

The first launch includes a large download and full integrity check before model loading. Subsequent launches reuse verified, unchanged files. The default API binds to localhost. To bind a specific Tailscale address, set `BIND_ADDRESS` to that address.

The generated API key is stored in `state/api-key` with mode 0600. Alternatively, set `API_KEY` in a local `.env` file. Never commit that file.

```bash
API_KEY=$(cat state/api-key)
curl http://127.0.0.1:8010/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"What is 19 + 23?"}],"chat_template_kwargs":{"thinking":false}}'
```

## Offload modes

| Mode | Command | Storage behavior |
|---|---|---|
| NVMe + bounded RAM cache | `OFFLOAD_MODE=nvme docker compose up -d` | Native FP8 rows, direct NVMe reads on cache misses; 64 GiB cache budget by default |
| Full host RAM | `OFFLOAD_MODE=ram docker compose up -d` | Shared, prefaulted, locked mappings of both Engram shards; no duplicate row cache |

`DSV41_CACHE_GIB` changes the NVMe mode's aggregate row-cache budget. Reducing it saves host RAM but can increase I/O. A 128 GB host cannot hold both native Engram tables fully in RAM. Remote DDR4 and experimental NVFP4 Engram storage are research candidates, **not implemented by this release**.

## Context, speculation and modalities

- Default context: **409,600 tokens per request**, allowing a 400k input plus a response. `CONTEXT_LENGTH` accepts 400,000 through the model's published 1,048,576 limit; values above the tested default are not qualified.
- Up to eight running requests; 2,048-token prefill chunks and GPU memory fraction 0.85 reserve indexer workspace. The larger 4,096-chunk/0.88 configuration failed near 399k input, despite loading successfully.
- DSpark is enabled with block size 5. Decode/verification CUDA graphs are used; prefill CUDA graphs are disabled by this runtime.
- Native image input uses the OpenAI `image_url` content format. The checkpoint's full **1,024 image tokens per image** are retained. An actual 3,024×588 image consumed exactly 1,024 image tokens and was described correctly.
- Explicit V4.1 reasoning and tool parsers are enabled. A real tool-call/result round trip and schema-constrained JSON passed in the reference deployment.
- Native video/audio workflows are **not qualified**. Do not infer support from an OpenAI-compatible route or from successful image input.

## Measurements and remaining gates

The tested native NVMe configuration uses the exact pinned source and runtime below. These are exploratory single-wave measurements, not a comprehensive quality study.

| Test | Observed result |
|---|---|
| 400,000 actual input tokens | All three random retrieval codes found; first token at 60.59 s |
| Six concurrent 350k inputs | 2.1M input tokens; 75.64 s with all requests decoding simultaneously |
| Runtime occupancy during that test | 2,143,232 populated logical context tokens, six running requests, CUDA graphs active |
| Short code decode, concurrency 1 | 162.6 tok/s over emitted tokens 129–641 |
| Short code decode, concurrency 2 | 135.1–148.0 tok/s per request |
| Short code decode, concurrency 4 | 112.6–126.4 tok/s per request |
| Short code decode, concurrency 8 | 83.6–96.6 tok/s per request |

The concurrency capacity test deliberately ignored EOS and generated 8,192 tokens per request to maintain occupancy. It proves capacity, **not output quality**. Long-input tests used a synthetic repeated archive and unique request prefixes. KV capacity refers to model-native compressed attention state, not a conventional dense KV representation.

Still required: full Docker-wrapper end-to-end acceptance, full-table RAM-mode inference, repeated matched performance comparisons, representative quality/logit checks, remote RAM/NVFP4 comparisons, native or clearly labeled auxiliary video/audio workflows, and a one-hour soak. B12x passed standalone tests but is not used in this default serving path.

## Reproducibility and attribution

- Model: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), revision `fb2764a5cf321eaa5070ca8f9e892818f477c16d`.
- Runtime: [SGLang](https://github.com/sgl-project/sglang), image `lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5`.
- Two SM120 compatibility fixes: initialize the existing DeepGEMM planner for V4.1 ratio-1/2 indexers; split both sparse-prefill KV sources into supported 64-token pages with independent scratch buffers. The latter file retains SGLang's Apache-2.0 licensing.
- Related prior work: [B12x](https://github.com/local-inference-lab/b12x), [DeepSpark](https://github.com/brandonmmusic-max/deepspark), and [the earlier DS4 SM120 recipe](https://github.com/jacklarmer/deepseek-v4-flash-0731-sm120). Earlier DS4 settings informed investigation; they are not evidence of V4.1 acceptance.

Model weights are downloaded at runtime and are not included in the image. Their upstream license remains applicable. See `NOTICE` for code attribution.

## Tests and shutdown

```bash
docker compose run --rm --entrypoint python3 deepseek /opt/dsv41/tests/test_row_store.py
docker compose run --rm -e OFFLOAD_MODE=ram --entrypoint python3 deepseek /opt/dsv41/tests/test_row_store.py
docker compose exec deepseek python3 /opt/dsv41/boot.py smoke
docker compose exec deepseek python3 /opt/dsv41/benchmarks/sweep.py
docker compose exec deepseek python3 /opt/dsv41/benchmarks/matrix.py
docker compose down
```

Shutdown preserves the checkpoint, state directory and kernel cache. Restore your previous serving container using its own saved launch configuration.

The matrix writes an incremental `TABLE.md`, JSON summaries and raw token/timing records under `state/matrix-<timestamp>/`. It sweeps 512, 2k, 8k, 32k, 64k, 128k, 200k and 400k inputs at requested concurrency 1/2/4/8. Override `PREFILL_SIZES` or `CONCURRENCIES` for another grid. It reports effective input throughput including queuing, a common emitted-token decode window, and observed client overlap; a queued burst is not labeled simultaneous decoding. Large matrices can take tens of minutes.
