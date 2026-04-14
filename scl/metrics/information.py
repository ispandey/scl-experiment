"""Information-theoretic metric proxies."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def ib_IXZ(mu: torch.Tensor, logvar: torch.Tensor) -> float:
    """
    VAE-style KL divergence: I(X;Z) proxy.
    KL(q(z|x) || N(0,I)) averaged over the batch.
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.mean().item()


def ib_IZtildeY(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 10,
) -> float:
    """
    CE-based I(Z̃;Y) proxy.
    I(Z̃;Y) ≈ log|Y| - H(Y|Z̃)
            = log(num_classes) - CE(logits, y)
    """
    ce = F.cross_entropy(logits, y).item()
    return math.log(num_classes) - ce


def channel_semantic_loss(
    z_clean: torch.Tensor,
    z_tilde: torch.Tensor,
    server_model: torch.nn.Module,
    y: torch.Tensor,
    num_classes: int = 10,
) -> float:
    """
    Δ_ch = I(Z;Y) - I(Z̃;Y)  ≥ 0 by data-processing inequality.
    """
    with torch.no_grad():
        IZY = ib_IZtildeY(server_model(z_clean), y, num_classes)
        IZtildeY = ib_IZtildeY(server_model(z_tilde), y, num_classes)
    return max(0.0, IZY - IZtildeY)


def capacity_geometry_bound(
    Sigma_z: torch.Tensor,
    snr_db: float,
    alpha: float = 1.0,
) -> float:
    """
    Theorem 2 geometry bound (nats converted to bits):
    C_geom = ½ Σ_i log₂(1 + (1+SNR)/α · λ_i)
    """
    snr = 10 ** (snr_db / 10.0)
    scale = (1.0 + snr) / alpha
    eigvals = torch.linalg.eigvalsh(Sigma_z)
    eigvals = eigvals.clamp(min=0.0)
    return 0.5 * torch.log2(1.0 + scale * eigvals).sum().item()


def semantic_efficiency(
    IZtildeY: float,
    snr_db: float,
    bandwidth: float = 1.0,
) -> float:
    """
    η_s(SNR) = I(Z̃;Y) / C(SNR)   where C = B·log₂(1+SNR).

    IZtildeY is produced by ib_IZtildeY() in nats (natural-log units).
    Shannon capacity C is in bits (log₂ units).  Convert nats to bits
    by dividing by ln(2) before forming the ratio so both quantities
    share the same unit.
    """
    snr = 10 ** (snr_db / 10.0)
    C = bandwidth * math.log2(1.0 + snr + 1e-8)
    IZtildeY_bits = IZtildeY / math.log(2)  # nats → bits
    return IZtildeY_bits / (C + 1e-8)


def estimate_sigma_z(
    z_samples: torch.Tensor,
) -> torch.Tensor:
    """Empirical covariance of z samples (B, d) → (d, d)."""
    flat = z_samples.flatten(1).float()
    n = flat.shape[0]
    mean = flat.mean(dim=0, keepdim=True)
    centred = flat - mean
    return (centred.T @ centred) / (n - 1)
