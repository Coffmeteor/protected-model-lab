from __future__ import annotations

from pathlib import Path

import folder_paths


def resolved_model_path(folder: str, name: str, required_suffix: str) -> Path:
    path = Path(folder_paths.get_full_path_or_raise(folder, name)).resolve()
    if path.suffix.lower() != required_suffix:
        raise ValueError(f"expected {required_suffix} file, got: {path.name}")
    return path
