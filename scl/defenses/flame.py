"""D3: FLAME (Nguyen et al. 2022) — cosine-similarity clustering + noise."""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn.functional as F


class FLAME:
    """
    FLAME: cluster client updates by cosine similarity,
    keep the majority cluster, clip norms, add Gaussian noise.
    """

    def __init__(self, noise_scale: float = 0.001, clip_norm: float = 1.0):
        self.noise_scale = noise_scale
        self.clip_norm = clip_norm

    def aggregate(self, gradients: List[torch.Tensor]) -> torch.Tensor:
        n = len(gradients)
        flat = torch.stack([g.flatten() for g in gradients])  # (n, d)

        # Cosine-distance-based 2-cluster grouping via single-linkage heuristic
        normed = F.normalize(flat, dim=1)
        cos_sim = normed @ normed.T  # (n, n)

        # Simple majority-cluster via spectral sign
        # Compute mean cosine similarity to group centre
        mean_vec = normed.mean(dim=0, keepdim=True)
        sims = (normed * mean_vec).sum(dim=1)  # (n,)
        median_sim = sims.median()
        accepted = (sims >= median_sim).nonzero(as_tuple=True)[0].tolist()
        if not accepted:
            accepted = list(range(n))

        selected = [gradients[i] for i in accepted]

        # Clip norms
        clipped = []
        for g in selected:
            norm = g.norm()
            if norm > self.clip_norm:
                g = g * self.clip_norm / (norm + 1e-8)
            clipped.append(g)

        agg = torch.stack(clipped).mean(dim=0)

        # Add Gaussian noise
        noise = torch.randn_like(agg) * self.noise_scale
        return agg + noise
