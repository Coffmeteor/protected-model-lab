from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from common.safetensors_stream import write_streaming
from lora_mapping.lokr import DirectLoKrLayer, dense_delta


def build_coefficient_carrier(
    base_path: str | Path,
    adapter_path: str | Path,
    output_path: str | Path,
    layers: list[DirectLoKrLayer],
    private_fraction: float,
    strength: float,
) -> None:
    by_model_key = {layer.model_key: layer for layer in layers}
    with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
        def transform(key: str, base: torch.Tensor, dtype_name: str) -> torch.Tensor | None:
            layer = by_model_key.get(key)
            if layer is None:
                return None
            w1 = adapter.get_tensor(layer.w1_key)
            w2 = adapter.get_tensor(layer.w2_key)
            update = dense_delta(w1, w2, strength * (1.0 - private_fraction))
            return base.float().add_(update).to(base.dtype)

        write_streaming(base_path, output_path, transform)
