"""Ideal (no-channel) pass-through."""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from .base import BaseChannel


class NoiselessChannel(BaseChannel):
    """σ²_ε = 0 (ideal link). Returns z unchanged."""

    def forward(self, z: torch.Tensor, snr_db: float) -> Tuple[torch.Tensor, Dict]:
        return z, {"snr_effective_db": float("inf"), "sigma2_eps": 0.0}
