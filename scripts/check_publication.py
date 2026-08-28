#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKABLE_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".jsonl", ".csv", ".gitignore"}
FORBIDDEN_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".spcore", ".png", ".jpg", ".jpeg", ".webp"}

errors: list[str] = []
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    relative = path.relative_to(ROOT)
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        result = subprocess.run(["git", "check-ignore", "-q", str(relative)], cwd=ROOT, check=False)
        if result.returncode != 0:
            errors.append(f"model/image artifact is not ignored: {relative}")
    if path.suffix.lower() in TRACKABLE_SUFFIXES and relative != Path("config/experiment.yaml") and not str(relative).startswith(("outputs/", "reports/")):
        text = path.read_text(encoding="utf-8", errors="replace")
        private_home_pattern = "/" + r"home/[^/]+/"
        if re.search(private_home_pattern, text):
            errors.append(f"absolute home path in publishable file: {relative}")

for private_path in (Path("config/experiment.yaml"), Path("outputs/carriers"), Path("outputs/cores"), Path("reports/asset_manifest.json")):
    result = subprocess.run(["git", "check-ignore", "-q", str(private_path)], cwd=ROOT, check=False)
    if result.returncode != 0:
        errors.append(f"private path is not ignored: {private_path}")

if errors:
    raise SystemExit("publication check failed:\n" + "\n".join(f"- {error}" for error in errors))
print("publication check passed")
