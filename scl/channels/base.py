"""Abstract base class for semantic channel models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch


class BaseChannel(ABC):
    """All channels apply noise/distortion in the latent (smash data) space."""

    @abstractmethod
    def forward(
        self, z: torch.Tensor, snr_db: float
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Apply channel to smash data z.

        Args:
            z:      Smash data tensor, shape (B, ...).
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            (z_tilde, info_dict) where info_dict contains channel diagnostics.
        """

    def __call__(self, z: torch.Tensor, snr_db: float) -> Tuple[torch.Tensor, Dict]:
        return self.forward(z, snr_db)
