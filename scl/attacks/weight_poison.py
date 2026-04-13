"""A1: Weight Poisoning attack."""
from __future__ import annotations

import torch


class WeightPoisonAttack:
    """
    A1: Gradient poisoning — add a scaled random perturbation.

    g_adv = g + scale · û,   û = unit random vector
    """

    def __init__(self, scale: float = 5.0):
        self.scale = scale

    def attack(self, gradient: torch.Tensor) -> torch.Tensor:
        direction = torch.randn_like(gradient)
        direction = direction / (direction.norm() + 1e-8)
        return gradient + self.scale * direction
