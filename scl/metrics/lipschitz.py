"""Lipschitz constant estimation and Theorem 1 bound computation."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


def estimate_Ls(
    server_model: nn.Module,
    z_samples: torch.Tensor,
    n_pairs: int = 200,
) -> float:
    """
    Estimate L_s = max ‖f_s(z₁)−f_s(z₂)‖ / ‖z₁−z₂‖ over random pairs.
    """
    server_model.eval()
    n = len(z_samples)
    if n < 2:
        return 0.0
    ratios = []
    with torch.no_grad():
        for _ in range(n_pairs):
            i, j = np.random.choice(n, 2, replace=False)
            zi, zj = z_samples[i].unsqueeze(0), z_samples[j].unsqueeze(0)
            diff_out = (server_model(zi) - server_model(zj)).norm().item()
            diff_in = (zi - zj).norm().item()
            if diff_in > 1e-6:
                ratios.append(diff_out / diff_in)
    return float(max(ratios)) if ratios else 0.0


def estimate_Ll(
    server_model: nn.Module,
    criterion: nn.Module,
    z_samples: torch.Tensor,
    y_samples: torch.Tensor,
    n_pairs: int = 100,
) -> float:
    """Estimate L_ℓ = max |ℓ(z₁)−ℓ(z₂)| / ‖z₁−z₂‖."""
    server_model.eval()
    n = len(z_samples)
    if n < 2:
        return 0.0
    ratios = []
    with torch.no_grad():
        for _ in range(n_pairs):
            i, j = np.random.choice(n, 2, replace=False)
            zi, zj = z_samples[i].unsqueeze(0), z_samples[j].unsqueeze(0)
            yi, yj = y_samples[i:i+1], y_samples[j:j+1]
            li = criterion(server_model(zi), yi).item()
            lj = criterion(server_model(zj), yj).item()
            diff_in = (zi - zj).norm().item()
            if diff_in > 1e-6:
                ratios.append(abs(li - lj) / diff_in)
    return float(max(ratios)) if ratios else 0.0


def estimate_Lg(
    server_model: nn.Module,
    criterion: nn.Module,
    z_samples: torch.Tensor,
    y_samples: torch.Tensor,
    n_pairs: int = 50,
) -> float:
    """Estimate L_g = ‖∇θL(z₁)−∇θL(z₂)‖ / ‖z₁−z₂‖."""
    server_model.eval()
    n = len(z_samples)
    if n < 2:
        return 0.0
    ratios = []
    params = list(server_model.parameters())

    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        zi, zj = z_samples[i].unsqueeze(0), z_samples[j].unsqueeze(0)
        yi, yj = y_samples[i:i+1], y_samples[j:j+1]

        server_model.zero_grad()
        loss_i = criterion(server_model(zi), yi)
        loss_i.backward()
        grad_i = torch.cat([
            p.grad.flatten() for p in params if p.grad is not None
        ])

        server_model.zero_grad()
        loss_j = criterion(server_model(zj), yj)
        loss_j.backward()
        grad_j = torch.cat([
            p.grad.flatten() for p in params if p.grad is not None
        ])

        diff_grad = (grad_i - grad_j).norm().item()
        diff_in = (zi - zj).norm().item()
        if diff_in > 1e-6:
            ratios.append(diff_grad / diff_in)

        server_model.zero_grad()

    return float(max(ratios)) if ratios else 0.0


def power_iter_jacobian_norm(
    server_model: nn.Module,
    z: torch.Tensor,
    criterion,
    y: torch.Tensor,
    n_iter: int = 10,
) -> float:
    """
    Approximate ‖J_t‖₂ = max singular value of ∂(server_output)/∂z
    via power iteration.
    """
    server_model.eval()
    z = z.detach().clone().requires_grad_(True)
    d = z.numel()

    # Initialise random unit vector
    v = torch.randn(d, device=z.device)
    v = v / (v.norm() + 1e-8)

    for _ in range(n_iter):
        v = v.requires_grad_(False)
        z_in = z.clone().requires_grad_(True)
        out = server_model(z_in.unsqueeze(0) if z_in.dim() == 3 else z_in)
        loss = criterion(out, y.unsqueeze(0) if y.dim() == 0 else y)
        grad = torch.autograd.grad(loss, z_in, create_graph=False)[0]
        grad_flat = grad.flatten()

        # Jv
        Jv = grad_flat * v.dot(grad_flat)  # approximate J^T J v
        norm = Jv.norm()
        if norm < 1e-8:
            break
        v = Jv / norm

    sigma = norm.item() if isinstance(norm, torch.Tensor) else norm
    return sigma


def compute_theorem1_bound(
    Ls: float,
    Ll: float,
    d_z: int,
    alpha: float,
    snr_db: float,
) -> float:
    """
    Theorem 1 bound: B(SNR) = L_ℓ · L_s · √(d_z · α / (1+SNR)).
    """
    snr = 10 ** (snr_db / 10.0)
    return Ll * Ls * math.sqrt(d_z * alpha / (1.0 + snr))
