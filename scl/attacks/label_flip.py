"""A2: Label Flipping attack."""
from __future__ import annotations

import torch


class LabelFlipAttack:
    """
    A2: y' = (y + 1) mod num_classes.
    Applied on-client before local training.
    """

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

    def attack_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return (labels + 1) % self.num_classes
