#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carrier_builder.svd import build_svd_carrier
from common.config import load_config
from common.hashing import sha256_file
from core_format.spcore import CoreLayer, ProtectedCore, write_core
from lora_mapping.lokr import inspect_direct_lokr

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(ROOT / "config/experiment.yaml"))
parser.add_argument("--strategy", choices=["top_singular_private", "tail_singular_private", "block_selected_private", "hybrid"], default="top_singular_private")
parser.add_argument("--private-rank", type=int, default=8)
parser.add_argument("--selected-blocks", default="", help="Required for block_selected_private/hybrid, for example 0-6 or 0,2,4")
args = parser.parse_args()
cfg = load_config(args.config)
layers, errors = inspect_direct_lokr(cfg.lora_files[0].path, cfg.base_model)
if errors:
    raise SystemExit("mapping errors prevent SVD carrier build")
if args.strategy in {"block_selected_private", "hybrid"} and not args.selected_blocks:
    raise SystemExit("--selected-blocks is required for block_selected_private and hybrid; run plan_block_search.py first")
blocks = set()
for part in args.selected_blocks.split(","):
    if "-" in part:
        first, last = map(int, part.split("-", 1))
        blocks.update(range(first, last + 1))
    elif part:
        blocks.add(int(part))
tag = f"{args.strategy}_r{args.private_rank}"
if args.strategy in {"block_selected_private", "hybrid"}:
    tag += "_b" + args.selected_blocks.replace(",", "-")
carrier = cfg.output_root / "carriers" / f"carrier_{tag}.safetensors"
private = build_svd_carrier(cfg.base_model, cfg.lora_files[0].path, carrier, layers, args.private_rank, args.strategy, cfg.lora_files[0].strength_model, blocks)
core_layers = tuple(CoreLayer(key, first, second, 1.0, kind) for key, (kind, first, second) in private.items())
core = ProtectedCore(cfg.architecture, sha256_file(carrier), tag, core_layers)
core_path = cfg.output_root / "cores" / f"core_{tag}.spcore"
write_core(core_path, core)
csv_path = cfg.project_root / "reports/split_candidates.csv"
write_header = not csv_path.exists()
with csv_path.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["strategy", "private_rank", "selected_blocks", "carrier", "carrier_sha256", "core", "core_sha256", "core_size_bytes"])
    if write_header:
        writer.writeheader()
    writer.writerow({"strategy": args.strategy, "private_rank": args.private_rank, "selected_blocks": args.selected_blocks, "carrier": str(carrier), "carrier_sha256": core.carrier_sha256, "core": str(core_path), "core_sha256": sha256_file(core_path), "core_size_bytes": core_path.stat().st_size})
print(carrier)
print(core_path)
