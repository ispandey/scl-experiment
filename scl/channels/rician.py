"""Rician-Semantic channel."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from .base import BaseChannel


class RicianSemanticChannel(BaseChannel):
    """
    Rician fading + additive semantic noise.

    K-factor = ν²/(2σ²) = 1  → κ_Rice = K/(K+1) = 0.5
    σ²_ε = α / (1 + SNR_k · κ_Rice)

    h = ν·e^{jθ} + (σ/√2)(X+jY),  X,Y ~ N(0,1)
    We use |h|² directly.
    """

    def __init__(self, alpha: float = 1.0, nu: float = 1.0, sigma_rice: float = 1.0):
        self.alpha = alpha
        self.nu = nu
        self.sigma_rice = sigma_rice
        K = nu ** 2 / (2 * sigma_rice ** 2)
        self.kappa_rice = K / (K + 1)

    def forward(self, z: torch.Tensor, snr_db: float) -> Tuple[torch.Tensor, Dict]:
        snr_lin = 10 ** (snr_db / 10.0)
        sigma2_eps = self.alpha / (1.0 + snr_lin * self.kappa_rice)

        device = z.device
        bs = z.shape[0]
        # Real and imag parts of Rician fading
        real = self.nu + self.sigma_rice * torch.randn(bs, device=device)
        imag = self.sigma_rice * torch.randn(bs, device=device)
        h_mag2 = (real ** 2 + imag ** 2).view(-1, *([1] * (z.dim() - 1)))

        z_faded = torch.sqrt(h_mag2) * z
        noise = torch.randn_like(z) * math.sqrt(sigma2_eps)
        z_tilde = z_faded + noise

        return z_tilde, {
            "snr_effective_db": snr_db,
            "sigma2_eps": sigma2_eps,
            "kappa_rice": self.kappa_rice,
        }
