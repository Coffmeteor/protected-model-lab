from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from common.safetensors_stream import write_streaming
from delta_split.kron_svd import selected_direction_factors
from lora_mapping.lokr import DirectLoKrLayer, dense_delta


def build_svd_carrier(
    base_path: str | Path,
    adapter_path: str | Path,
    output_path: str | Path,
    layers: list[DirectLoKrLayer],
    private_rank: int,
    strategy: str,
    strength: float,
    selected_blocks: set[int] | None = None,
) -> dict[str, tuple[str, torch.Tensor, torch.Tensor]]:
    if strategy not in {"top_singular_private", "tail_singular_private", "block_selected_private", "hybrid"}:
        raise ValueError(f"unsupported split strategy: {strategy}")
    by_model_key = {layer.model_key: layer for layer in layers}
    private: dict[str, tuple[str, torch.Tensor, torch.Tensor]] = {}
    selected_blocks = selected_blocks or set()
    with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
        def transform(key: str, base: torch.Tensor, dtype_name: str) -> torch.Tensor | None:
            layer = by_model_key.get(key)
            if layer is None:
                return None
            w1 = adapter.get_tensor(layer.w1_key)
            w2 = adapter.get_tensor(layer.w2_key)
            full = dense_delta(w1, w2, strength)
            parts = key.split(".")
            block = int(parts[1]) if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit() else None
            whole_private = strategy in {"block_selected_private", "hybrid"} and block in selected_blocks
            if whole_private:
                private[key] = ("lokr", w1.float(), w2.float() * strength)
                public = torch.zeros_like(full)
            elif strategy == "block_selected_private":
                public = full
            else:
                strongest = strategy != "tail_singular_private"
                up, down, _ = selected_direction_factors(w1, w2, private_rank, strongest=strongest)
                up = up * strength
                private[key] = ("lora", up, down)
                public = full - up @ down
            return base.float().add_(public).to(base.dtype)

        write_streaming(base_path, output_path, transform)
    return private
