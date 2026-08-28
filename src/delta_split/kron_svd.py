from __future__ import annotations

import torch


def kron_singular_values(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Return singular values of kron(w1,w2) without forming the dense Kronecker matrix."""
    s1 = torch.linalg.svdvals(w1.float())
    s2 = torch.linalg.svdvals(w2.float())
    return torch.outer(s1, s2).flatten().sort(descending=True).values


def kron_svd_factors(w1: torch.Tensor, w2: torch.Tensor):
    """Small-factor SVD whose pairwise directions exactly describe kron(w1,w2)."""
    u1, s1, vh1 = torch.linalg.svd(w1.float(), full_matrices=False)
    u2, s2, vh2 = torch.linalg.svd(w2.float(), full_matrices=False)
    return u1, s1, vh1, u2, s2, vh2


def selected_direction_factors(w1: torch.Tensor, w2: torch.Tensor, rank: int, strongest: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Represent selected singular directions of kron(w1,w2) as standard LoRA up/down factors."""
    u1, s1, vh1, u2, s2, vh2 = kron_svd_factors(w1, w2)
    pairs = []
    for i in range(len(s1)):
        for j in range(len(s2)):
            pairs.append((float(s1[i] * s2[j]), i, j))
    pairs.sort(reverse=strongest, key=lambda item: item[0])
    chosen = pairs[: min(rank, len(pairs))]
    up_columns = []
    down_rows = []
    values = []
    for singular, i, j in chosen:
        root = singular ** 0.5
        up_columns.append(torch.kron(u1[:, i], u2[:, j]) * root)
        down_rows.append(torch.kron(vh1[i, :], vh2[j, :]) * root)
        values.append(singular)
    return torch.stack(up_columns, dim=1), torch.stack(down_rows, dim=0), torch.tensor(values)
