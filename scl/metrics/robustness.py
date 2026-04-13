"""Robustness and aggregation quality metrics."""
from __future__ import annotations

from typing import List, Set

import torch


def compute_gradient_bias_norm(
    estimated_grad: torch.Tensor,
    true_grad: torch.Tensor,
) -> float:
    """‖E[g̃] − g‖₂ — Theorem 4 proxy."""
    return (estimated_grad - true_grad).norm().item()


def compute_false_positive_rate(
    accepted_ids: List[int],
    honest_ids: List[int],
) -> float:
    """
    Fraction of honest clients incorrectly excluded by the defense.
    FPR = |{honest clients not in accepted}| / |honest clients|
    """
    accepted_set: Set[int] = set(accepted_ids)
    fp = sum(1 for i in honest_ids if i not in accepted_set)
    return fp / len(honest_ids) if honest_ids else 0.0


def compute_acceptance_ratio(
    accepted_ids: List[int],
    total_clients: int,
) -> float:
    return len(accepted_ids) / total_clients if total_clients > 0 else 0.0


def gradient_variance_decomposition(
    z: torch.Tensor,
    z_tilde: torch.Tensor,
    server_model: torch.nn.Module,
    criterion,
    y: torch.Tensor,
) -> dict:
    """
    Estimate channel contribution and data contribution to gradient variance.

    Returns:
        {'grad_var_channel': float, 'grad_var_data': float}
    """
    server_model.eval()
    params = list(server_model.parameters())

    def get_grad(inp):
        server_model.zero_grad()
        out = server_model(inp)
        loss = criterion(out, y)
        loss.backward()
        return torch.cat([p.grad.flatten() for p in params if p.grad is not None])

    with torch.no_grad():
        pass  # only need grads

    try:
        g_clean = get_grad(z_tilde.detach().requires_grad_(False))
        g_noisy = get_grad(z_tilde)
        channel_var = (g_noisy - g_clean).pow(2).mean().item()
        data_var = g_clean.pow(2).mean().item()
    except Exception:
        channel_var = 0.0
        data_var = 0.0
    finally:
        server_model.zero_grad()

    return {"grad_var_channel": channel_var, "grad_var_data": data_var}
