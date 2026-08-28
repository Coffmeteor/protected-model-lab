from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

import torch
from safetensors import safe_open


DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


@dataclass(frozen=True)
class TensorInfo:
    key: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


@dataclass(frozen=True)
class SafeTensorLayout:
    path: Path
    data_start: int
    tensors: tuple[TensorInfo, ...]
    metadata: dict[str, str] | None


def read_layout(path: str | Path) -> SafeTensorLayout:
    source = Path(path)
    with source.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"not a safetensors file: {source}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(handle.read(header_len).decode("utf-8").rstrip(" \t\r\n\0"))
    tensors = []
    for key, value in header.items():
        if key == "__metadata__":
            continue
        begin, end = value["data_offsets"]
        tensors.append(TensorInfo(key, value["dtype"], tuple(value["shape"]), int(begin), int(end)))
    return SafeTensorLayout(source, 8 + header_len, tuple(tensors), header.get("__metadata__"))


def _header_bytes(layout: SafeTensorLayout, metadata: dict[str, str] | None) -> bytes:
    offset = 0
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = metadata
    for info in layout.tensors:
        size = info.end - info.begin
        header[info.key] = {"dtype": info.dtype, "shape": list(info.shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return raw


def tensor_bytes(tensor: torch.Tensor, dtype_name: str) -> memoryview:
    tensor = tensor.detach().cpu().contiguous()
    if dtype_name == "BF16":
        return memoryview(tensor.to(torch.bfloat16).view(torch.uint16).numpy())
    numpy_value = tensor.numpy()
    return memoryview(numpy_value)


def copy_exact(source: BinaryIO, dest: BinaryIO, offset: int, size: int, chunk_size: int = 16 * 1024 * 1024) -> None:
    source.seek(offset)
    remaining = size
    while remaining:
        chunk = source.read(min(chunk_size, remaining))
        if not chunk:
            raise EOFError(f"source ended with {remaining} bytes left")
        dest.write(chunk)
        remaining -= len(chunk)


def write_streaming(
    source_path: str | Path,
    output_path: str | Path,
    transform: Callable[[str, torch.Tensor, str], torch.Tensor | None],
    metadata: dict[str, str] | None = None,
) -> None:
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")
    layout = read_layout(source_path)
    header = _header_bytes(layout, metadata if metadata is not None else layout.metadata)
    with source_path.open("rb") as source, temp.open("wb") as dest, safe_open(source_path, framework="pt", device="cpu") as tensors:
        dest.write(struct.pack("<Q", len(header)))
        dest.write(header)
        for info in layout.tensors:
            value = transform(info.key, tensors.get_tensor(info.key), info.dtype) if transform is not None else None
            if value is None:
                copy_exact(source, dest, layout.data_start + info.begin, info.end - info.begin)
            else:
                raw = tensor_bytes(value, info.dtype)
                if raw.nbytes != info.end - info.begin:
                    raise ValueError(f"transformed byte size mismatch for {info.key}: {raw.nbytes}")
                dest.write(raw)
        dest.flush()
        os.fsync(dest.fileno())
    with safe_open(temp, framework="pt", device="cpu") as check:
        if set(check.keys()) != {x.key for x in layout.tensors}:
            raise ValueError("output key verification failed")
        for info in layout.tensors:
            if tuple(check.get_slice(info.key).get_shape()) != info.shape:
                raise ValueError(f"output shape verification failed: {info.key}")
    os.replace(temp, output_path)
