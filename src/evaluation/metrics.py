from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image


def compare_images(reference: str | Path, candidate: str | Path) -> dict[str, float]:
    first = np.asarray(Image.open(reference).convert("RGB"), dtype=np.float32) / 255.0
    second = np.asarray(Image.open(candidate).convert("RGB"), dtype=np.float32) / 255.0
    if first.shape != second.shape:
        raise ValueError(f"image shape mismatch: {first.shape} != {second.shape}")
    difference = first - second
    mse = float(np.mean(difference ** 2))
    result = {"mse": mse, "mae": float(np.mean(np.abs(difference))), "psnr": float(-10.0 * math.log10(max(mse, 1e-20)))}
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return result
    result["ssim"] = float(structural_similarity(first, second, channel_axis=2, data_range=1.0))
    return result
