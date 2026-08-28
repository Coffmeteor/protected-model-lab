#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(ROOT / "config/experiment.yaml"))
args = parser.parse_args()
cfg = load_config(args.config)
source = ROOT / "custom_nodes/ComfyUI-ProtectedModelLab"
target = cfg.comfyui_root / "custom_nodes/ComfyUI-ProtectedModelLab"
if target.is_symlink():
    if target.resolve() != source.resolve():
        raise SystemExit(f"refusing to replace unrelated symlink: {target} -> {target.resolve()}")
    print(target)
elif target.exists():
    raise SystemExit(f"refusing to replace existing path: {target}")
else:
    os.symlink(source, target, target_is_directory=True)
    print(target)
