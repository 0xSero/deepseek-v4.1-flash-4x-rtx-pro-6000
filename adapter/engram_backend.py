"""Bounded exact file-backed replacement for EngramEmbedding's owned-row gather.

The original hash, gating, projections, TP all-reduce and model remain unchanged.
Native callback executes IO inside CUDA graphs without calling the CUDA API.
"""
import ctypes as C
import glob
import json
import logging
import os
from pathlib import Path
import struct

import torch

P, U = C.c_void_p, C.c_uint64
_lib = C.CDLL(str(Path(__file__).with_name('librow_store.so')))
_lib.row_store_open.argtypes = [C.c_char_p, U, U, U, U]
_lib.row_store_open.restype = P
_lib.row_store_range.argtypes = [P, U, U]

class Work(C.Structure):
    _fields_ = [('store', P), ('ids', P), ('weights', P), ('scales', P), ('count', U)]

_paths = glob.glob('/usr/local/cuda*/targets/x86_64-linux/lib/libcudart.so*')
_paths += glob.glob('/usr/local/lib/python*/dist-packages/nvidia/cuda_runtime/lib/libcudart.so*')
_paths += glob.glob('/usr/local/lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*')
_cuda = C.CDLL(_paths[0])
_cuda.cudaLaunchHostFunc.argtypes = [P, P, P]
_cuda.cudaLaunchHostFunc.restype = C.c_int

def install(module):
    cls = module.EngramEmbedding
    def init(self, num_embeddings, dim, layer_id):
        torch.nn.Module.__init__(self)
        assert dim == 256
        self.dim, self.tp_size = dim, module.get_parallel().tp_size
        rank = module.get_parallel().tp_rank
        self.row_start = num_embeddings * rank // self.tp_size
        end = num_embeddings * (rank + 1) // self.tp_size
        self.rows, self.host_table = end - self.row_start, None
        root = Path(os.environ['DSV41_SOURCE'])
        index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
        prefix = f'layers.{layer_id}.engram.embed.'
        filename = root/index[prefix+'weight']
        assert index[prefix+'weight'] == index[prefix+'scale']
        with filename.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(length))
        w, s = header[prefix+'weight'], header[prefix+'scale']
        assert w['shape'] == [num_embeddings, 256] and w['dtype'] == 'F8_E4M3'
        assert s['shape'] == [num_embeddings, 8] and s['dtype'] == 'F8_E8M0'
        budget = int(float(os.getenv('DSV41_CACHE_GIB', '64')) * 2**30) // (2*self.tp_size)
        self._store = _lib.row_store_open(str(filename).encode(), num_embeddings,
            8+length+w['data_offsets'][0], 8+length+s['data_offsets'][0], budget)
        if not self._store:
            raise RuntimeError(f'Could not open Engram backing shard: {filename}')
        _lib.row_store_range(self._store, self.row_start, end)
        self._staging = {}
        self._works = {}
        # The loader sees names but never allocates/copies the complete tables.
        self.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.scale = torch.nn.Parameter(torch.empty(0, dtype=torch.float8_e8m0fnu), requires_grad=False)
        def validate_weight(param, source):
            expected = w if param is self.weight else s
            if list(source.shape) != expected['shape']:
                raise ValueError('Engram checkpoint shape mismatch')
        self.weight.weight_loader = validate_weight
        self.scale.weight_loader = validate_weight
        logging.getLogger(__name__).info('Exact %s Engram layer=%s rank=%s rows=[%s,%s) cache_budget=%s',
                                        os.getenv('OFFLOAD_MODE', 'nvme'), layer_id, rank, self.row_start, end, budget)

    def owned(self, indices):
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather
        count = indices.numel()
        if not count:
            return self._empty(indices)
        capacity = 1 << (count - 1).bit_length()
        key = (indices.device.index, capacity)
        if key not in self._staging:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f'Engram staging {key} must be warmed before graph capture')
            ids = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
            w = torch.empty((capacity, 256), dtype=torch.uint8, pin_memory=True)
            s = torch.empty((capacity, 8), dtype=torch.uint8, pin_memory=True)
            dw, ds = w.to(indices.device), s.to(indices.device)
            sequential = torch.arange(capacity, dtype=torch.int64, device=indices.device)
            self._staging[key] = (ids, w, s, dw, ds, sequential)
        ids, w, s, dw, ds, sequential = self._staging[key]
        work_key = (indices.device.index, count)
        if work_key not in self._works:
            self._works[work_key] = Work(self._store, ids.data_ptr(), w.data_ptr(), s.data_ptr(), count)
        work = self._works[work_key]
        ids[:count].copy_(indices.reshape(-1), non_blocking=True)
        error = _cuda.cudaLaunchHostFunc(torch.cuda.current_stream().cuda_stream,
            C.cast(_lib.row_store_lookup, P), C.addressof(work))
        if error:
            raise RuntimeError(f'CUDA Engram host callback failed: {error}')
        dw[:count].copy_(w[:count], non_blocking=True)
        ds[:count].copy_(s[:count], non_blocking=True)
        out = self._empty(indices)
        engram_gather(dw.data_ptr(), ds.data_ptr(), sequential[:count], out.view(-1, 256), 256, 32)
        return out
    cls.__init__ = init
    cls._owned_rows = owned
