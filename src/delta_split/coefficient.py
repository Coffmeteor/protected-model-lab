from __future__ import annotations

import torch


def coefficient_factors(w1: torch.Tensor, w2: torch.Tensor, private_fraction: float) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    if not 0.0 < private_fraction < 1.0:
        raise ValueError("private_fraction must be between 0 and 1")
    public = (w1, w2 * (1.0 - private_fraction))
    private = (w1, w2 * private_fraction)
    return public, private
