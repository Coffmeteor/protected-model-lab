#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.config import load_config
from common.hashing import sha256_file
from core_format.spcore import read_core
from lora_mapping.lokr import dense_delta, inspect_direct_lokr


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()))


parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(ROOT / "config/experiment.yaml"))
parser.add_argument("--private-fraction", type=float, default=0.35)
args = parser.parse_args()
cfg = load_config(args.config)
tag = f"p{round(args.private_fraction * 100):03d}"
carrier_path = cfg.output_root / "carriers" / f"carrier_coeff_{tag}.safetensors"
core_path = cfg.output_root / "cores" / f"core_coeff_{tag}.spcore"
carrier_hash = sha256_file(carrier_path)
core = read_core(core_path, carrier_hash)
core_by_key = {layer.model_key: layer for layer in core.layers}
layers, errors = inspect_direct_lokr(cfg.lora_files[0].path, cfg.base_model)
if errors:
    raise SystemExit("mapping errors prevent validation")

started = time.monotonic()
rows = []
max_static = 0.0
max_bf16 = 0.0
sum_teacher_restored_sq = 0.0
sum_carrier_teacher_sq = 0.0
sum_delta_sq = 0.0
with safe_open(cfg.base_model, framework="pt", device="cpu") as base, safe_open(carrier_path, framework="pt", device="cpu") as carrier, safe_open(cfg.lora_files[0].path, framework="pt", device="cpu") as adapter:
    for index, layer in enumerate(layers, 1):
        w1 = adapter.get_tensor(layer.w1_key)
        w2 = adapter.get_tensor(layer.w2_key)
        delta = dense_delta(w1, w2, cfg.lora_files[0].strength_model)
        public = delta * (1.0 - args.private_fraction)
        private = delta * args.private_fraction
        delta_norm = max(norm(delta), torch.finfo(torch.float32).eps)
        static_error = norm(delta - (public + private)) / delta_norm

        base_weight = base.get_tensor(layer.model_key)
        carrier_weight = carrier.get_tensor(layer.model_key)
        teacher = (base_weight.float() + delta).to(base_weight.dtype)
        restored = (carrier_weight.float() + private).to(base_weight.dtype)
        teacher_restored = norm(teacher.float() - restored.float())
        carrier_teacher = norm(carrier_weight.float() - teacher.float())
        bf16_relative = teacher_restored / delta_norm
        sum_teacher_restored_sq += teacher_restored ** 2
        sum_carrier_teacher_sq += carrier_teacher ** 2
        sum_delta_sq += delta_norm ** 2
        max_static = max(max_static, static_error)
        max_bf16 = max(max_bf16, bf16_relative)
        rows.append({
            "model_key": layer.model_key,
            "delta_frobenius_norm": delta_norm,
            "static_relative_error": static_error,
            "teacher_restored_bf16_frobenius": teacher_restored,
            "teacher_carrier_bf16_frobenius": carrier_teacher,
            "teacher_restored_relative_to_delta": bf16_relative,
        })
        del delta, public, private, base_weight, carrier_weight, teacher, restored
        if index % 32 == 0:
            print(f"validated {index}/{len(layers)}", flush=True)

aggregate_restored = math.sqrt(sum_teacher_restored_sq)
aggregate_carrier_gap = math.sqrt(sum_carrier_teacher_sq)
aggregate_delta = math.sqrt(sum_delta_sq)
recovery = 1.0 - aggregate_restored / max(aggregate_carrier_gap, torch.finfo(torch.float32).eps)
result = {
    "kind": "static_weight_validation",
    "private_fraction": args.private_fraction,
    "carrier_sha256": carrier_hash,
    "core_sha256": sha256_file(core_path),
    "layer_count": len(rows),
    "max_static_relative_error": max_static,
    "aggregate_teacher_restored_bf16_frobenius": aggregate_restored,
    "aggregate_teacher_carrier_bf16_frobenius": aggregate_carrier_gap,
    "aggregate_delta_frobenius": aggregate_delta,
    "aggregate_teacher_restored_relative_to_delta": aggregate_restored / aggregate_delta,
    "max_layer_teacher_restored_relative_to_delta": max_bf16,
    "weight_recovery_ratio": recovery,
    "elapsed_seconds": time.monotonic() - started,
    "layers": rows,
}
reports = cfg.project_root / "reports"
(reports / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
summary = f"""# Reconstruction error\n\nThis is static weight validation only; no image-quality claim is made.\n\n- Layers: {len(rows)}\n- Private fraction: {args.private_fraction}\n- Maximum float32 split reconstruction error: {max_static:.6e}\n- Aggregate teacher/restored BF16 error relative to original delta: {result['aggregate_teacher_restored_relative_to_delta']:.6e}\n- Maximum per-layer teacher/restored BF16 error relative to delta: {max_bf16:.6e}\n- Aggregate teacher/carrier gap: {aggregate_carrier_gap:.6e}\n- Weight recovery ratio: {recovery:.8f}\n- Runtime: {result['elapsed_seconds']:.2f} seconds\n\nThe BF16 result includes two-stage carrier baking and runtime-style private patch rounding.\n"""
(reports / "reconstruction_error.md").write_text(summary, encoding="utf-8")
(reports / "evaluation.md").write_text(summary, encoding="utf-8")
print(json.dumps({k: result[k] for k in result if k != "layers"}, indent=2))
