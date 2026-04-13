"""AWGN-Semantic channel (no fading)."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from .base import BaseChannel


class AWGNSemanticChannel(BaseChannel):
    """
    Deterministic SNR, no fading.
    z̃ = z + ε,   ε ~ N(0, σ²_ε · I),   σ²_ε = α/(1+SNR)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def forward(self, z: torch.Tensor, snr_db: float) -> Tuple[torch.Tensor, Dict]:
        snr_lin = 10 ** (snr_db / 10.0)
        sigma2_eps = self.alpha / (1.0 + snr_lin)
        noise = torch.randn_like(z) * math.sqrt(sigma2_eps)
        return z + noise, {"snr_effective_db": snr_db, "sigma2_eps": sigma2_eps}
