"""A4: IPM — Inner Product Manipulation attack (Fang et al. 2020)."""
from __future__ import annotations

from typing import List

import torch


class IPMAttack:
    """
    A4: Reverse the average honest gradient by factor γ.

    g_adv = -γ · ĝ_honest
    """

    def __init__(self, gamma: float = 10.0):
        self.gamma = gamma

    def attack(self, gradients_honest: List[torch.Tensor]) -> torch.Tensor:
        mean_honest = torch.stack(gradients_honest).mean(dim=0)
        return -self.gamma * mean_honest
