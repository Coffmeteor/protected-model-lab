from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MAGIC = b"SPCORE1\0"


@dataclass(frozen=True)
class CoreLayer:
    model_key: str
    w1: torch.Tensor
    w2: torch.Tensor
    layer_scale: float = 1.0
    adapter_type: str = "lokr"


@dataclass(frozen=True)
class ProtectedCore:
    architecture: str
    carrier_sha256: str
    split_method: str
    layers: tuple[CoreLayer, ...]


def _array_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().to(torch.float32).contiguous().numpy().tobytes(order="C")


def write_core(path: str | Path, core: ProtectedCore) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    payload = bytearray()
    entries = []
    for layer in core.layers:
        tensors = []
        for name, value in (("w1", layer.w1), ("w2", layer.w2)):
            raw = _array_bytes(value)
            tensors.append({"name": name, "shape": list(value.shape), "dtype": "F32", "offset": len(payload), "length": len(raw)})
            payload.extend(raw)
        entries.append({"model_key": layer.model_key, "layer_scale": layer.layer_scale, "adapter_type": layer.adapter_type, "tensors": tensors})
    header = {
        "format_version": 1,
        "architecture": core.architecture,
        "required_carrier_sha256": core.carrier_sha256,
        "split_method": core.split_method,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "layers": entries,
    }
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with temp.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, output)


def read_core(path: str | Path, carrier_sha256: str | None = None) -> ProtectedCore:
    with Path(path).open("rb") as handle:
        if handle.read(len(MAGIC)) != MAGIC:
            raise ValueError("not a protected-model-lab .spcore file")
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        payload = handle.read()
    if hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
        raise ValueError("private core payload checksum mismatch")
    required = header["required_carrier_sha256"]
    if carrier_sha256 is not None and carrier_sha256 != required:
        raise ValueError(f"carrier hash mismatch: core requires {required}, got {carrier_sha256}")
    layers = []
    for entry in header["layers"]:
        values = {}
        for tensor in entry["tensors"]:
            start = tensor["offset"]
            end = start + tensor["length"]
            array = np.frombuffer(payload[start:end], dtype=np.float32).copy().reshape(tensor["shape"])
            values[tensor["name"]] = torch.from_numpy(array)
        layers.append(CoreLayer(entry["model_key"], values["w1"], values["w2"], float(entry["layer_scale"]), entry.get("adapter_type", "lokr")))
    return ProtectedCore(header["architecture"], required, header["split_method"], tuple(layers))
