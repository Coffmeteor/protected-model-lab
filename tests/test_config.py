from pathlib import Path

import pytest
import yaml

from common.config import load_config


def test_multiple_adapters_are_rejected_instead_of_silently_ignored(tmp_path: Path):
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    base = tmp_path / "base.safetensors"
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    for path in (base, first, second):
        path.write_bytes(b"placeholder")
    config = {
        "project_root": str(tmp_path / "project"),
        "comfyui_root": str(comfy),
        "base_model": str(base),
        "lora_files": [
            {"path": str(first), "strength_model": 1.0, "strength_clip": 0.0},
            {"path": str(second), "strength_model": 1.0, "strength_clip": 0.0},
        ],
        "output_root": str(tmp_path / "project/outputs"),
        "private_fractions": [0.35],
        "architecture": "mock",
    }
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one direct-factor LoKr"):
        load_config(config_path)
