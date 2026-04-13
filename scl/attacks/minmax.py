"""A5: Min-Max attack — maximize loss while minimising detection."""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


class MinMaxAttack:
    """
    A5: Maximise task loss subject to gradient staying close to the honest mean.

    The malicious gradient is:
        g_adv = argmax { -CE(model(x), y) - λ·‖g_adv - g_global‖ }

    Implemented via one step of gradient ascent on the loss.
    """

    def __init__(self, lam: float = 1.0, lr: float = 0.01, steps: int = 5):
        self.lam = lam
        self.lr = lr
        self.steps = steps

    def attack(
        self,
        gradient: torch.Tensor,
        global_grad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Craft adversarial gradient.

        Args:
            gradient:    Honest gradient (used as starting point).
            global_grad: Global mean gradient for detection penalty.
        """
        g_adv = gradient.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([g_adv], lr=self.lr)

        for _ in range(self.steps):
            optimizer.zero_grad()
            # Maximise task loss (flip sign) while staying close to global mean
            adv_loss = -g_adv.norm()  # proxy for gradient magnitude flip
            if global_grad is not None:
                detection_penalty = self.lam * (g_adv - global_grad).norm()
                total = adv_loss + detection_penalty
            else:
                total = adv_loss
            total.backward()
            optimizer.step()

        return g_adv.detach()
