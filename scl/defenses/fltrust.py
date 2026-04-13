"""D4: FLTrust (Cao et al. 2022) — server-gradient-based trust weighting."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


class FLTrust:
    """
    FLTrust: weight each client gradient by ReLU(cosine_sim(g_k, g_server)).
    Requires a small clean root dataset on the server.
    """

    def aggregate(
        self,
        gradients: List[torch.Tensor],
        server_gradient: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if server_gradient is None:
            # Fallback: use mean of gradients as proxy server gradient
            server_gradient = torch.stack(gradients).mean(dim=0)

        sg_flat = server_gradient.flatten()
        weights = []
        for g in gradients:
            g_flat = g.flatten()
            sim = F.cosine_similarity(g_flat.unsqueeze(0), sg_flat.unsqueeze(0)).item()
            weights.append(max(0.0, sim))

        total = sum(weights)
        if total < 1e-8:
            return torch.stack(gradients).mean(dim=0)

        # Normalise gradient magnitudes to match server gradient, then weight-average
        sg_norm = sg_flat.norm().item()
        normalised = []
        for w, g in zip(weights, gradients):
            if w > 0:
                g_norm = g.norm().item()
                scale = sg_norm / (g_norm + 1e-8)
                normalised.append(w * g * scale)

        agg = sum(normalised) / total
        return agg
