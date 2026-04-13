"""D0: FedAvg — simple weighted average."""
from __future__ import annotations

from typing import List, Optional

import torch


class FedAvg:
    """Standard federated averaging."""

    def aggregate(
        self,
        gradients: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        if not gradients:
            raise ValueError("Empty gradient list")
        if weights is None:
            weights = [1.0 / len(gradients)] * len(gradients)
        total = sum(w for w in weights)
        agg = sum(w * g for w, g in zip(weights, gradients))
        return agg / total
