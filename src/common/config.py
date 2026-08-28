from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AdapterConfig:
    path: Path
    strength_model: float
    strength_clip: float


@dataclass(frozen=True)
class ExperimentConfig:
    project_root: Path
    comfyui_root: Path
    base_model: Path
    lora_files: tuple[AdapterConfig, ...]
    optional_workflow: Path | None
    optional_calibration_prompts: Path | None
    output_root: Path
    private_fractions: tuple[float, ...]
    architecture: str


def _optional_path(value: Any) -> Path | None:
    return None if value in (None, "") else Path(str(value)).expanduser().resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    adapters = tuple(
        AdapterConfig(
            path=Path(item["path"]).expanduser().resolve(),
            strength_model=float(item.get("strength_model", 1.0)),
            strength_clip=float(item.get("strength_clip", 1.0)),
        )
        for item in raw.get("lora_files", [])
    )
    if not adapters:
        raise ValueError("lora_files must contain at least one adapter")
    if len(adapters) != 1:
        raise ValueError("this release supports exactly one direct-factor LoKr; refusing to ignore or mis-compose additional adapters")
    cfg = ExperimentConfig(
        project_root=Path(raw["project_root"]).expanduser().resolve(),
        comfyui_root=Path(raw["comfyui_root"]).expanduser().resolve(),
        base_model=Path(raw["base_model"]).expanduser().resolve(),
        lora_files=adapters,
        optional_workflow=_optional_path(raw.get("optional_workflow")),
        optional_calibration_prompts=_optional_path(raw.get("optional_calibration_prompts")),
        output_root=Path(raw["output_root"]).expanduser().resolve(),
        private_fractions=tuple(float(x) for x in raw.get("private_fractions", [0.35])),
        architecture=str(raw.get("architecture", "auto")),
    )
    for label, target in (("project_root", cfg.project_root), ("comfyui_root", cfg.comfyui_root)):
        if not target.is_absolute():
            raise ValueError(f"{label} must be absolute: {target}")
    for label, target in (("base_model", cfg.base_model), *[("lora", x.path) for x in cfg.lora_files]):
        if not target.is_file():
            raise FileNotFoundError(f"{label} does not exist: {target}")
    for p in cfg.private_fractions:
        if not 0.0 < p < 1.0:
            raise ValueError(f"private fraction must be between 0 and 1: {p}")
    return cfg
