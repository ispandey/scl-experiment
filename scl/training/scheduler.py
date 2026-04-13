"""Learning rate scheduler: linear warmup + cosine annealing."""
from __future__ import annotations

import math


class WarmupCosineScheduler:
    """
    Linear warmup from start_lr to peak_lr over warmup_rounds,
    then cosine annealing from peak_lr to min_lr over remaining rounds.
    """

    def __init__(
        self,
        optimizer,
        warmup_rounds: int = 10,
        total_rounds: int = 100,
        start_lr: float = 0.01,
        peak_lr: float = 0.1,
        min_lr: float = 1e-4,
    ):
        self.optimizer = optimizer
        self.warmup_rounds = warmup_rounds
        self.total_rounds = total_rounds
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.min_lr = min_lr

    def step(self, round_t: int):
        """Update learning rate for round t (1-indexed)."""
        if round_t <= self.warmup_rounds:
            # Linear warmup
            lr = self.start_lr + (self.peak_lr - self.start_lr) * (
                round_t / self.warmup_rounds
            )
        else:
            # Cosine annealing
            T_max = self.total_rounds - self.warmup_rounds
            t = round_t - self.warmup_rounds
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                1 + math.cos(math.pi * t / T_max)
            )

        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        return lr
