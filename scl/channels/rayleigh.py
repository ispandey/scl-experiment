"""Rayleigh-Semantic channel (primary channel model)."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from .base import BaseChannel


class RayleighSemanticChannel(BaseChannel):
    """
    Rayleigh fading + additive semantic noise in the latent space.

    h_k ~ CN(0,1) → |h_k|² ~ Exp(1)
    σ²_ε = α / (1 + SNR)
    z̃ = √|h|² · z + ε,   ε ~ N(0, σ²_ε · I)

    Per-client SNR heterogeneity (optional):
        SNR_k = SNR_base + N(0, σ²_snr_var)  [in linear domain after dB conversion]
    """

    def __init__(self, alpha: float = 1.0, snr_var_db: float = 3.0):
        self.alpha = alpha
        self.snr_var_db = snr_var_db  # std of per-client SNR offset (dB)

    def forward(
        self, z: torch.Tensor, snr_db: float, per_client: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        if per_client and self.snr_var_db > 0:
            offset = torch.randn(1).item() * self.snr_var_db
            snr_db = snr_db + offset

        snr_lin = 10 ** (snr_db / 10.0)
        sigma2_eps = self.alpha / (1.0 + snr_lin)

        # Rayleigh fading magnitude squared
        device = z.device
        h_mag2 = torch.distributions.Exponential(
            torch.ones(z.shape[0], device=device)
        ).sample()
        h_mag2 = h_mag2.view(-1, *([1] * (z.dim() - 1)))

        z_faded = torch.sqrt(h_mag2) * z
        noise = torch.randn_like(z) * math.sqrt(sigma2_eps)
        z_tilde = z_faded + noise

        return z_tilde, {
            "snr_effective_db": snr_db,
            "sigma2_eps": sigma2_eps,
            "h_mag2_mean": h_mag2.mean().item(),
        }
