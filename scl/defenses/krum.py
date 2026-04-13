"""D2: Multi-Krum (Blanchard et al. 2017)."""
from __future__ import annotations

from typing import List, Optional

import torch


class Krum:
    """
    Multi-Krum: select the m gradient updates with lowest sum of
    distances to their n−f−2 nearest neighbours.
    """

    def __init__(self, f: Optional[int] = None, m: Optional[int] = None):
        """
        Args:
            f: Number of Byzantine clients (estimated).
            m: Number of 'good' gradients to select (defaults to n−f−2).
        """
        self.f = f
        self.m = m

    def aggregate(
        self,
        gradients: List[torch.Tensor],
        f: Optional[int] = None,
    ) -> torch.Tensor:
        n = len(gradients)
        f = f if f is not None else (self.f if self.f is not None else max(1, n // 5))
        m = self.m if self.m is not None else max(1, n - f - 2)
        m = min(m, n)

        stacked = torch.stack([g.flatten() for g in gradients])  # (n, d)

        # Pairwise squared distances
        dists = torch.cdist(stacked, stacked, p=2)  # (n, n)

        scores = []
        k = n - f - 2
        k = max(1, min(k, n - 1))
        for i in range(n):
            row = dists[i].clone()
            row[i] = float("inf")
            nearest = row.topk(k, largest=False).values
            scores.append(nearest.sum().item())

        # Select m gradients with smallest scores
        best_indices = sorted(range(n), key=lambda i: scores[i])[:m]
        selected = [gradients[i] for i in best_indices]
        return torch.stack(selected).mean(dim=0)
