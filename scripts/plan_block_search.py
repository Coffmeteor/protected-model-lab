#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.config import load_config
from lora_mapping.lokr import inspect_direct_lokr


def block_id(model_key: str) -> int | None:
    parts = model_key.split(".")
    return int(parts[1]) if len(parts) > 2 and parts[0] == "blocks" and parts[1].isdigit() else None


def contiguous_groups(blocks: list[int], group_count: int) -> list[list[int]]:
    return [
        blocks[(len(blocks) * index) // group_count : (len(blocks) * (index + 1)) // group_count]
        for index in range(group_count)
    ]


parser = argparse.ArgumentParser(description="Plan a small architecture-specific block search without building checkpoints.")
parser.add_argument("--config", default=str(ROOT / "config/experiment.yaml"))
args = parser.parse_args()
cfg = load_config(args.config)
layers, errors = inspect_direct_lokr(cfg.lora_files[0].path, cfg.base_model)
if errors:
    raise SystemExit("mapping errors prevent block planning; run inspect_assets.py first")
energy_by_block: dict[int, float] = defaultdict(float)
other_energy: dict[str, float] = defaultdict(float)
strength_squared = cfg.lora_files[0].strength_model ** 2
with safe_open(cfg.lora_files[0].path, framework="pt", device="cpu") as adapter:
    for layer in layers:
        w1 = adapter.get_tensor(layer.w1_key).float()
        w2 = adapter.get_tensor(layer.w2_key).float()
        energy = float(w1.square().sum() * w2.square().sum()) * strength_squared
        selected_block = block_id(layer.model_key)
        if selected_block is None:
            other_energy[layer.model_key.split(".", 1)[0]] += energy
        else:
            energy_by_block[selected_block] += energy
blocks = sorted(energy_by_block)
if not blocks:
    raise SystemExit("no numbered transformer blocks were detected")
total = sum(energy_by_block.values()) + sum(other_energy.values())
proposals = []
for count, label in ((2, "halves"), (4, "quarters")):
    for index, group in enumerate(contiguous_groups(blocks, count)):
        if not group:
            continue
        energy = sum(energy_by_block[item] for item in group)
        proposals.append({"name": f"{label}_{index + 1}", "selected_blocks": group, "energy_fraction": energy / total, "reason": "contiguous control candidate"})
half_count = max(1, len(blocks) // 2)
ranked = sorted(blocks, key=lambda item: energy_by_block[item], reverse=True)
selected = sorted(ranked[:half_count])
proposals.append({"name": "top_energy_half", "selected_blocks": selected, "energy_fraction": sum(energy_by_block[item] for item in selected) / total, "reason": "non-contiguous energy baseline; not a semantic optimum"})
report = {
    "architecture": cfg.architecture,
    "block_count": len(blocks),
    "block_ids": blocks,
    "total_delta_energy": total,
    "per_block": [{"block": block, "energy": energy_by_block[block], "energy_fraction": energy_by_block[block] / total} for block in blocks],
    "non_block_roots": {key: {"energy": value, "energy_fraction": value / total} for key, value in sorted(other_energy.items())},
    "proposals": proposals,
    "selection_rule": "Evaluate a small staged subset across multiple fixed prompts. Energy ranks candidates but does not identify where identity/style semantics live.",
    "warning": "No block range is universal. Do not carry a winning range to another architecture, checkpoint, adapter, or strength without re-running mapping and evaluation.",
}
output = cfg.project_root / "reports/block_search_plan.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(output)
