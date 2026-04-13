"""D6: BCBSA — Byzantine-Channel-aware Blind Semantics-Aware Aggregation (proposed method)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class BCBSA:
    """
    BCBSA: trust-score-weighted aggregation using semantic channel signals.

    Trust score for client k:
        trust_k = ω₁·F_k + ω₂·(1 − D_k) + ω₃·cos(g_k, ĝ)

    where:
        F_k = semantic fidelity = 1 − ‖z̃_k − z_k‖² / (‖z_k‖² + ε)
        D_k = normalised distortion = ‖z̃_k − z_k‖² / (d_z + ε)
        ĝ   = running EMA of accepted mean gradient

    Temporal EMA:
        trust_k^t = (1−η)·trust_k^{t−1} + η·trust_k^t_raw

    Ablation variants are controlled via the omega parameters and `temporal` flag.
    """

    def __init__(
        self,
        omega1: float = 1.0,    # fidelity weight
        omega2: float = 1.0,    # distortion weight
        omega3: float = 0.1,    # gradient cosine weight
        tau: float = 0.3,       # acceptance threshold
        eta: float = 0.1,       # temporal EMA rate
        temporal: bool = True,
    ):
        self.omega1 = omega1
        self.omega2 = omega2
        self.omega3 = omega3
        self.tau = tau
        self.eta = eta
        self.temporal = temporal

        self._prev_trust: Dict[int, float] = {}
        self._running_mean_grad: Optional[torch.Tensor] = None

    def reset(self):
        self._prev_trust.clear()
        self._running_mean_grad = None

    def _compute_trust_raw(
        self,
        client_id: int,
        z_clean: torch.Tensor,
        z_tilde: torch.Tensor,
        gradient: torch.Tensor,
    ) -> float:
        # Fidelity
        num = (z_tilde - z_clean).pow(2).sum().item()
        den = z_clean.pow(2).sum().item() + 1e-8
        fidelity = 1.0 - num / den

        # Normalised distortion (0-1 range via sigmoid-like clip)
        dz = z_clean.numel()
        dist = (z_tilde - z_clean).pow(2).mean().item()
        dist_norm = min(1.0, dist)   # clip to [0,1]

        # Gradient cosine similarity
        cos_sim = 0.0
        if self._running_mean_grad is not None:
            g_flat = gradient.flatten()
            rg_flat = self._running_mean_grad.flatten()
            cos_sim = F.cosine_similarity(
                g_flat.unsqueeze(0), rg_flat.unsqueeze(0)
            ).item()

        trust_raw = (
            self.omega1 * max(0.0, fidelity)
            + self.omega2 * (1.0 - dist_norm)
            + self.omega3 * max(0.0, cos_sim)
        )
        return trust_raw

    def aggregate(
        self,
        gradients: List[torch.Tensor],
        client_ids: List[int],
        z_cleans: List[torch.Tensor],
        z_tildes: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, float, Dict[int, float]]:
        """
        Returns:
            (aggregated_gradient, acceptance_ratio, trust_scores_dict)
        """
        trusts: Dict[int, float] = {}
        for k, cid in enumerate(client_ids):
            trust_raw = self._compute_trust_raw(
                cid, z_cleans[k], z_tildes[k], gradients[k]
            )
            if self.temporal and cid in self._prev_trust:
                trust = (1.0 - self.eta) * self._prev_trust[cid] + self.eta * trust_raw
            else:
                trust = trust_raw
            self._prev_trust[cid] = trust
            trusts[cid] = trust

        # Build accepted set
        accepted_k = [
            k for k, cid in enumerate(client_ids) if trusts[cid] > self.tau
        ]
        if not accepted_k:
            accepted_k = list(range(len(gradients)))  # fallback

        accepted_grads = [gradients[k] for k in accepted_k]
        w = torch.tensor(
            [trusts[client_ids[k]] for k in accepted_k], dtype=torch.float32
        )
        w = F.softmax(w, dim=0)

        agg = sum(wi * g for wi, g in zip(w.tolist(), accepted_grads))

        # Update running mean gradient
        self._running_mean_grad = agg.detach().clone()

        acceptance_ratio = len(accepted_k) / len(gradients)
        return agg, acceptance_ratio, trusts


# ---------------------------------------------------------------------------
# Pre-configured ablation variants
# ---------------------------------------------------------------------------

def bcbsa_full() -> BCBSA:
    return BCBSA(omega1=1.0, omega2=1.0, omega3=0.1, temporal=True)


def bcbsa_nofid() -> BCBSA:
    """No fidelity term (ω₁=0)."""
    return BCBSA(omega1=0.0, omega2=1.0, omega3=0.1, temporal=True)


def bcbsa_nodist() -> BCBSA:
    """No distortion term (ω₂=0)."""
    return BCBSA(omega1=1.0, omega2=0.0, omega3=0.1, temporal=True)


def bcbsa_nosem() -> BCBSA:
    """No semantic signals (ω₁=ω₂=0). Gradient cosine only."""
    return BCBSA(omega1=0.0, omega2=0.0, omega3=0.1, temporal=True)


def bcbsa_notemp() -> BCBSA:
    """No temporal regularisation."""
    return BCBSA(omega1=1.0, omega2=1.0, omega3=0.1, temporal=False)
