"""Communication cost metrics."""
from __future__ import annotations

import torch


def compute_bytes_transmitted(
    z: torch.Tensor,
    compression_ratio: float = 1.0,
    bits_per_element: int = 32,
) -> int:
    """
    Estimate bytes transmitted for smash data z with optional top-k compression.

    Args:
        z:                 Smash data tensor.
        compression_ratio: Fraction of elements kept (1.0 = no compression).
        bits_per_element:  Bit-width per element (default float32 = 32 bits).
    """
    total_elements = z.numel()
    if compression_ratio >= 1.0:
        # Dense transmission
        return total_elements * bits_per_element // 8
    # Sparse transmission: values (float32) + indices (int32)
    kept = max(1, int(total_elements * compression_ratio))
    value_bits = kept * bits_per_element
    index_bits = kept * 32  # 32-bit indices
    return (value_bits + index_bits) // 8


def topk_compress(z: torch.Tensor, rho: float) -> torch.Tensor:
    """
    Top-k sparsification: keep top `rho` fraction of elements by magnitude,
    zero out the rest.  Gradient is passed through for backward compatibility.
    """
    if rho >= 1.0:
        return z
    flat = z.flatten()
    k = max(1, int(flat.numel() * rho))
    threshold = flat.abs().topk(k).values.min()
    mask = (z.abs() >= threshold).float()
    return z * mask
