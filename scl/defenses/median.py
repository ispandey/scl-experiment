"""D1: Coordinate-wise Median."""
from __future__ import annotations

from typing import List

import torch


class CoordMedian:
    """Coordinate-wise median aggregation (Yin et al. 2018)."""

    def aggregate(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(gradients)  # (n, d)
        return stacked.median(dim=0).values
