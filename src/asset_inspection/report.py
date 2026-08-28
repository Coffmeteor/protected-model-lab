from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import safetensors
import torch
from safetensors import safe_open

from common.config import ExperimentConfig
from common.hashing import sha256_file
from lora_mapping.lokr import inspect_direct_lokr


def tensor_manifest(path: Path) -> dict:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "metadata": handle.metadata(),
            "tensor_count": len(handle.keys()),
            "tensors": [{"key": key, "shape": handle.get_slice(key).get_shape(), "dtype": handle.get_slice(key).get_dtype()} for key in handle.keys()],
        }


def generate_reports(cfg: ExperimentConfig) -> tuple[dict, dict]:
    reports = cfg.project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    base = tensor_manifest(cfg.base_model)
    adapters = [tensor_manifest(x.path) | {"strength_model": x.strength_model, "strength_clip": x.strength_clip} for x in cfg.lora_files]
    manifest = {"architecture": cfg.architecture, "base_model": base, "adapters": adapters, "workflow": str(cfg.optional_workflow) if cfg.optional_workflow else None}
    layers, errors = inspect_direct_lokr(cfg.lora_files[0].path, cfg.base_model)
    mapping = {
        "adapter_format": "direct_lokr",
        "mapped_count": len(layers),
        "errors": errors,
        "reliable": not errors,
        "layers": [layer.__dict__ for layer in layers],
    }
    (reports / "asset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports / "lora_mapping_report.json").write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    git_commit = subprocess.run(["git", "-C", str(cfg.comfyui_root), "rev-parse", "HEAD"], text=True, capture_output=True, check=False).stdout.strip()
    env = f"""# Environment report\n\n- Platform: {platform.platform()}\n- Python: {sys.version.splitlines()[0]}\n- PyTorch: {torch.__version__}\n- CUDA runtime: {torch.version.cuda}\n- CUDA available: {torch.cuda.is_available()}\n- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'}\n- safetensors: {safetensors.__version__}\n- ComfyUI root: `{cfg.comfyui_root}`\n- ComfyUI git commit: `{git_commit}`\n- Detected architecture: `{cfg.architecture}`\n- Adapter format: direct LoKr (`delta = strength * kron(w1, w2)` in the installed ComfyUI implementation)\n- Mapping reliable: `{not errors}`\n- Mapped tensors: {len(layers)}\n- Unmatched or unsupported keys: {len(errors)}\n\nNo source asset was modified. Assets outside the filled configuration are outside this experiment.\n"""
    (reports / "environment_report.md").write_text(env, encoding="utf-8")
    return manifest, mapping
