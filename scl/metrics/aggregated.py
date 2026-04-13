"""Aggregated metrics data structures."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import pandas as pd


@dataclass
class RoundMetrics:
    """All per-round metrics as defined in Part V of the design document."""
    round_t: int = 0

    # ── Learning ──────────────────────────────────────────────────────────
    test_accuracy: float = 0.0
    train_loss: float = 0.0
    excess_loss: float = 0.0

    # ── Channel ───────────────────────────────────────────────────────────
    semantic_fidelity: float = 0.0
    distortion_Dk: float = 0.0
    snr_effective_db: float = 0.0
    ber: float = 0.0

    # ── Information-theoretic ─────────────────────────────────────────────
    IXZ_proxy: float = 0.0
    IZtildeY_proxy: float = 0.0
    delta_ch: float = 0.0
    semantic_efficiency: float = 0.0

    # ── Aggregation / robustness ──────────────────────────────────────────
    acceptance_ratio: float = 1.0
    false_positive_rate: float = 0.0
    gradient_bias_norm: float = 0.0
    grad_var_channel: float = 0.0
    grad_var_data: float = 0.0

    # ── Communication ─────────────────────────────────────────────────────
    bytes_transmitted: int = 0
    compression_ratio: float = 1.0
    wall_clock_sec: float = 0.0

    # ── Lipschitz (EG-1 only) ─────────────────────────────────────────────
    Ls_estimate: float = 0.0
    Lg_estimate: float = 0.0
    bound_tightness: float = 0.0


class MetricsTracker:
    """Accumulates per-round metrics and provides export utilities."""

    def __init__(self):
        self.history: List[RoundMetrics] = []

    def record(self, m: RoundMetrics):
        self.history.append(m)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([asdict(m) for m in self.history])

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(m) for m in self.history], f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MetricsTracker":
        tracker = cls()
        with open(path) as f:
            records = json.load(f)
        for r in records:
            tracker.record(RoundMetrics(**r))
        return tracker

    def __len__(self):
        return len(self.history)

    def final_accuracy(self) -> float:
        if not self.history:
            return 0.0
        return self.history[-1].test_accuracy

    def auc_accuracy(self) -> float:
        """Area under accuracy-vs-round curve (simple trapezoidal)."""
        accs = [m.test_accuracy for m in self.history]
        if not accs:
            return 0.0
        return sum(accs) / len(accs)
