#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics import compare_images


def labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=/absolute/path/image.png")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("candidate label and path must be non-empty")
    return label, Path(raw_path).expanduser().resolve()


def recovery_spec(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected NAME:CARRIER_LABEL:RESTORED_LABEL")
    return parts[0], parts[1], parts[2]


parser = argparse.ArgumentParser(description="Compare already-generated local images without downloading metric models.")
parser.add_argument("--teacher", type=Path, required=True)
parser.add_argument("--candidate", type=labeled_path, action="append", required=True, help="LABEL=/absolute/path/image.png; repeat for each image")
parser.add_argument("--recovery", type=recovery_spec, action="append", default=[], help="NAME:CARRIER_LABEL:RESTORED_LABEL")
parser.add_argument("--output", type=Path, default=ROOT / "reports/candidate_metrics.json")
args = parser.parse_args()
teacher = args.teacher.expanduser().resolve()
if not teacher.is_file():
    raise SystemExit(f"teacher image not found: {teacher}")
result = {}
for label, path in args.candidate:
    if not path.is_file():
        raise SystemExit(f"candidate image not found: {path}")
    result[label] = compare_images(teacher, path)
for name, carrier_label, restored_label in args.recovery:
    if carrier_label not in result or restored_label not in result:
        raise SystemExit(f"unknown recovery labels: {carrier_label}, {restored_label}")
    carrier = result[carrier_label]["mse"]
    restored = result[restored_label]["mse"]
    result[name + "_recovery_ratio"] = 1.0 - restored / max(carrier, 1e-20)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(args.output)
