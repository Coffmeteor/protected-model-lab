#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "skills/protected-model-lab"

default_codex_root = Path(os.environ["CODEX_HOME"]).expanduser() if os.environ.get("CODEX_HOME") else Path.home() / ".codex"
parser = argparse.ArgumentParser(description="Install the bundled Codex skill without altering an existing skill.")
parser.add_argument("--skills-root", type=Path, default=default_codex_root / "skills")
args = parser.parse_args()
destination_root = args.skills_root.expanduser().resolve()
destination = destination_root / SOURCE.name
if destination.exists() or destination.is_symlink():
    raise SystemExit(f"refusing to replace existing skill: {destination}")
destination_root.mkdir(parents=True, exist_ok=True)
temporary = Path(tempfile.mkdtemp(prefix="protected-model-lab-skill-", dir=destination_root))
try:
    staged = temporary / SOURCE.name
    shutil.copytree(SOURCE, staged)
    os.replace(staged, destination)
finally:
    temporary.rmdir()
print(destination)
