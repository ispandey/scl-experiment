"""D2: Multi-Krum (Blanchard et al. 2017)."""
from __future__ import annotations

from typing import List, Optional

import torch


class Krum:
    """
    Multi-Krum: select the *m* gradient updates with the lowest sum of
    distances to their *n − f − 2* nearest neighbours.

    The distance matrix and all scoring operations are computed with
    PyTorch tensors, so the computation runs on whichever device the
    input gradients reside on (CPU or GPU).
    """

    def __init__(self, f: Optional[int] = None, m: Optional[int] = None):
        """
        Args:
            f: Estimated number of Byzantine clients.
               Defaults to ⌊n/5⌋ when not provided.
            m: Number of gradients to select and average (Multi-Krum).
               Defaults to *n − f − 2*, capped at *n*.
        """
        self.f = f
        self.m = m

    def aggregate(
        self,
        gradients: List[torch.Tensor],
        f: Optional[int] = None,
    ) -> torch.Tensor:
        n = len(gradients)
        num_byzantine = f if f is not None else (self.f if self.f is not None else max(1, n // 5))
        m = self.m if self.m is not None else max(1, n - num_byzantine - 2)
        m = min(m, n)

        # Stack into (n, d) on the same device as the inputs
        stacked = torch.stack([g.flatten() for g in gradients])  # (n, d)

        # ── Pairwise Euclidean distance matrix (GPU-accelerated) ──────────
        dists = torch.cdist(stacked, stacked, p=2)  # (n, n)

        # ── Vectorised score computation (replaces Python per-row loop) ───
        # Set diagonal to +inf so each gradient is not its own neighbour.
        dists_masked = dists.clone()
        dists_masked.fill_diagonal_(float("inf"))

        # Number of nearest neighbours to sum per gradient
        k = max(1, min(n - num_byzantine - 2, n - 1))

        # topk with largest=False gives the k smallest distances per row.
        # Shape: (n, k)
        nearest_k_vals = torch.topk(dists_masked, k, dim=1, largest=False).values
        scores = nearest_k_vals.sum(dim=1)  # (n,)

        # ── Select the m gradients with the smallest scores ───────────────
        # torch.topk keeps everything on-device; no Python sort needed.
        best_indices = torch.topk(scores, m, largest=False).indices  # (m,)

        selected = stacked[best_indices]  # (m, d)
        return selected.mean(dim=0)
