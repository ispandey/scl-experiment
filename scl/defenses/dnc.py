"""D5: DnC — Divide-and-Conquer / Spectral outlier removal (Shejwalkar & Houmansadr 2021)."""
from __future__ import annotations

from typing import List, Optional

import torch


class DnC:
    """
    Spectral-based Byzantine-robust aggregation.

    Projects gradients onto top-r PCA directions, removes outliers
    by z-score on projected coordinates, averages the rest.
    """

    def __init__(self, num_components: int = 2, z_thresh: float = 2.0):
        self.num_components = num_components
        self.z_thresh = z_thresh

    def aggregate(
        self, gradients: List[torch.Tensor], f: Optional[int] = None
    ) -> torch.Tensor:
        n = len(gradients)
        flat = torch.stack([g.flatten().float() for g in gradients])  # (n, d)

        # Centre
        mean = flat.mean(dim=0, keepdim=True)
        centred = flat - mean

        # SVD for PCA (use economy SVD for efficiency)
        r = min(self.num_components, n - 1, flat.shape[1])
        try:
            _, _, Vt = torch.linalg.svd(centred, full_matrices=False)
            top_V = Vt[:r]  # (r, d)
            projected = centred @ top_V.T  # (n, r)
        except Exception:
            return flat.mean(dim=0)

        # Remove outliers via z-score on each component
        keep = torch.ones(n, dtype=torch.bool)
        for comp in range(projected.shape[1]):
            col = projected[:, comp]
            mu = col.mean()
            std = col.std() + 1e-8
            z_scores = (col - mu).abs() / std
            keep &= z_scores < self.z_thresh

        if keep.sum() == 0:
            keep = torch.ones(n, dtype=torch.bool)

        selected = flat[keep]
        return selected.mean(dim=0)
