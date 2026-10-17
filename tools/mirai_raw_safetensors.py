"""Read bfloat16 tensors straight from a safetensors file as float32."""

from __future__ import annotations

import json
import os
import struct

import numpy as np


def bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2")
    return (u16.astype(np.uint32) << 16).view(np.float32)


class RawSafeTensors:
    """Minimal reader: safetensors header is 8-byte length + JSON + raw buffers."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")
        n = struct.unpack("<Q", self._fh.read(8))[0]
        self.header = json.loads(self._fh.read(n).decode("utf-8"))
        self.data_start = 8 + n
        self.size = os.path.getsize(path)

    def keys(self):
        return [k for k in self.header if k != "__metadata__"]

    def raw(self, name: str) -> bytes:
        info = self.header[name]
        start, end = info["data_offsets"]
        self._fh.seek(self.data_start + start)
        return self._fh.read(end - start)

    def tensor(self, name: str) -> np.ndarray:
        info = self.header[name]
        dtype, shape = info["dtype"], info["shape"]
        raw = self.raw(name)
        if dtype == "BF16":
            return bf16_to_f32(raw).reshape(shape)
        if dtype == "F32":
            return np.frombuffer(raw, dtype="<f4").reshape(shape)
        if dtype == "U8":
            return np.frombuffer(raw, dtype="u1").reshape(shape)
        if dtype == "I32":
            return np.frombuffer(raw, dtype="<i4").reshape(shape)
        raise ValueError("unsupported dtype " + dtype)

    def close(self):
        self._fh.close()


def find_weights(base: str) -> str:
    for entry in os.listdir(base):
        cand = os.path.join(base, entry, "model.safetensors")
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(base)


if __name__ == "__main__":
    path = find_weights(os.path.expanduser(
        "~/.cache/mirai/models/mirai/alibaba-qwen3.5-9b-mirai-mirai-m-4"))
    r = RawSafeTensors(path)
    prefix = "decoder.transformer.layers.0.mlp.up_projection.weights.quantized"
    w = r.tensor(prefix + ".weights")
    s = r.tensor(prefix + ".scales")
    z = r.tensor(prefix + ".zero_points")
    print("weights", w.shape, w.dtype)
    print("scales ", s.shape, s.dtype, "min", float(s.min()), "max", float(s.max()))
    print("zeros  ", z.shape, z.dtype, "min", float(z.min()), "max", float(z.max()))
    r.close()
