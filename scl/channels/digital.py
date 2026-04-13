"""Digital BPSK channel (bit-level pipeline)."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from .base import BaseChannel


class DigitalBPSKChannel(BaseChannel):
    """
    Digital channel: flatten z → sign-quantize to ±1 → AWGN → hard decision → reshape.

    BER = Q(√(2·SNR)) per symbol (theoretical approximation computed for reporting).
    """

    def forward(self, z: torch.Tensor, snr_db: float) -> Tuple[torch.Tensor, Dict]:
        snr_lin = 10 ** (snr_db / 10.0)
        orig_shape = z.shape
        z_flat = z.detach().flatten(1)  # (B, D)

        # BPSK modulate: sign-based → ±1
        bits = torch.sign(z_flat)
        bits[bits == 0] = 1.0

        # AWGN noise with given SNR
        noise_std = 1.0 / math.sqrt(2.0 * snr_lin)
        received = bits + torch.randn_like(bits) * noise_std

        # Hard decision
        demod = torch.sign(received)
        demod[demod == 0] = 1.0

        # Re-attach original magnitudes with recovered signs to preserve scale
        z_rec = demod * z_flat.abs()
        z_tilde = z_rec.view(orig_shape)

        # Theoretical BER = Q(√(2·SNR))
        ber = _qfunc(math.sqrt(2.0 * snr_lin))

        return z_tilde.detach(), {
            "snr_effective_db": snr_db,
            "ber": ber,
        }


def _qfunc(x: float) -> float:
    """Q(x) = 0.5 · erfc(x / √2)"""
    import math
    return 0.5 * math.erfc(x / math.sqrt(2))
