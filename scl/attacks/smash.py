"""A3: SMASH attack — semantic-space perturbation of smash data."""
from __future__ import annotations

import torch


class SMASHAttack:
    """
    A3: Perturb the smash data z̃ in the latent space.

    z̃_adv = z̃ + ε · ‖z̃‖ · û,   û = unit random vector
    """

    def __init__(self, epsilon: float = 0.3):
        self.epsilon = epsilon

    def attack_smash(self, z_tilde: torch.Tensor) -> torch.Tensor:
        u = torch.randn_like(z_tilde)
        u = u / (u.norm() + 1e-8)
        return z_tilde + self.epsilon * z_tilde.norm() * u
