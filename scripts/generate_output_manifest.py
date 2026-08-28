#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.hashing import sha256_file

entries = []
for directory in (ROOT / "outputs/carriers", ROOT / "outputs/cores"):
    for path in sorted(directory.glob("*")):
        if path.is_file() and path.suffix in {".safetensors", ".spcore"}:
            entries.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
output = ROOT / "outputs/manifest.json"
output.write_text(json.dumps({"artifacts": entries}, indent=2), encoding="utf-8")
print(output)
