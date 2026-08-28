from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class DirectLoKrLayer:
    adapter_prefix: str
    model_key: str
    w1_key: str
    w2_key: str
    alpha_key: str | None
    w1_shape: tuple[int, ...]
    w2_shape: tuple[int, ...]
    model_shape: tuple[int, ...]


def adapter_to_model_key(prefix: str) -> str:
    if not prefix.startswith("diffusion_model."):
        raise ValueError(f"unsupported Krea2 adapter prefix: {prefix}")
    return prefix.removeprefix("diffusion_model.") + ".weight"


def inspect_direct_lokr(adapter_path: str | Path, model_path: str | Path) -> tuple[list[DirectLoKrLayer], list[str]]:
    errors: list[str] = []
    layers: list[DirectLoKrLayer] = []
    with safe_open(adapter_path, framework="pt", device="cpu") as adapter, safe_open(model_path, framework="pt", device="cpu") as model:
        adapter_keys = set(adapter.keys())
        model_keys = set(model.keys())
        prefixes = sorted(k.removesuffix(".lokr_w1") for k in adapter_keys if k.endswith(".lokr_w1"))
        recognized: set[str] = set()
        for prefix in prefixes:
            w1_key = prefix + ".lokr_w1"
            w2_key = prefix + ".lokr_w2"
            alpha_key = prefix + ".alpha" if prefix + ".alpha" in adapter_keys else None
            recognized.add(w1_key)
            if alpha_key:
                recognized.add(alpha_key)
            if w2_key not in adapter_keys:
                errors.append(f"missing paired tensor: {w2_key}")
                continue
            recognized.add(w2_key)
            model_key = adapter_to_model_key(prefix)
            if model_key not in model_keys:
                errors.append(f"unmatched model tensor: {prefix} -> {model_key}")
                continue
            w1_shape = tuple(adapter.get_slice(w1_key).get_shape())
            w2_shape = tuple(adapter.get_slice(w2_key).get_shape())
            model_shape = tuple(model.get_slice(model_key).get_shape())
            if len(w1_shape) != 2 or len(w2_shape) != 2:
                errors.append(f"only direct 2D LoKr is supported: {prefix} {w1_shape} {w2_shape}")
                continue
            expected = (w1_shape[0] * w2_shape[0], w1_shape[1] * w2_shape[1])
            if expected != model_shape:
                errors.append(f"shape mismatch: {prefix} kron={expected} model={model_shape}")
                continue
            layers.append(DirectLoKrLayer(prefix, model_key, w1_key, w2_key, alpha_key, w1_shape, w2_shape, model_shape))
        for key in sorted(adapter_keys - recognized):
            errors.append(f"unsupported or unmatched adapter key: {key}")
    return layers, errors


def dense_delta(w1: torch.Tensor, w2: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    return torch.kron(w1.float(), w2.float()).mul_(strength)
