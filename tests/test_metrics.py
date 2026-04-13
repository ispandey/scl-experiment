"""Tests for metrics utilities."""
import math
import torch
import pytest
import numpy as np

from scl.metrics.information import (
    ib_IXZ, ib_IZtildeY, channel_semantic_loss,
    capacity_geometry_bound, semantic_efficiency, estimate_sigma_z,
)
from scl.metrics.lipschitz import compute_theorem1_bound
from scl.metrics.robustness import compute_false_positive_rate, compute_gradient_bias_norm
from scl.metrics.communication import compute_bytes_transmitted, topk_compress
from scl.metrics.aggregated import RoundMetrics, MetricsTracker
from scl.analysis.stats import paired_ttest, wilcoxon_test, bonferroni_correct, power_analysis_paired_t


# ── Information-theoretic ────────────────────────────────────────────────────

def test_ib_IXZ():
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    kl = ib_IXZ(mu, logvar)
    assert kl == pytest.approx(0.0, abs=1e-6)


def test_ib_IZtildeY():
    # Perfect prediction → I should approach log(num_classes)
    logits = torch.eye(5) * 100  # near-perfect
    y = torch.arange(5)
    val = ib_IZtildeY(logits, y, num_classes=5)
    assert val == pytest.approx(math.log(5), rel=0.05)


def test_semantic_efficiency():
    eta = semantic_efficiency(IZtildeY=1.0, snr_db=20.0)
    assert eta > 0.0


def test_capacity_geometry_bound():
    sigma = torch.eye(4) * 2.0
    bound = capacity_geometry_bound(sigma, snr_db=15.0)
    assert bound > 0.0


def test_estimate_sigma_z():
    z = torch.randn(16, 8)
    S = estimate_sigma_z(z)
    assert S.shape == (8, 8)


# ── Lipschitz ────────────────────────────────────────────────────────────────

def test_theorem1_bound():
    b = compute_theorem1_bound(Ls=1.0, Ll=1.0, d_z=100, alpha=1.0, snr_db=10.0)
    assert b > 0.0
    # Higher SNR → smaller bound
    b2 = compute_theorem1_bound(Ls=1.0, Ll=1.0, d_z=100, alpha=1.0, snr_db=30.0)
    assert b2 < b


# ── Robustness ───────────────────────────────────────────────────────────────

def test_false_positive_rate():
    accepted = [0, 1, 2, 4]
    honest = [0, 1, 2, 3, 4]
    fpr = compute_false_positive_rate(accepted, honest)
    assert fpr == pytest.approx(0.2)


def test_gradient_bias_norm():
    g1 = torch.ones(10)
    g2 = torch.zeros(10)
    bn = compute_gradient_bias_norm(g1, g2)
    assert bn == pytest.approx(math.sqrt(10), rel=1e-4)


# ── Communication ────────────────────────────────────────────────────────────

def test_bytes_transmitted():
    z = torch.randn(4, 128, 16, 16)
    b = compute_bytes_transmitted(z, compression_ratio=1.0)
    assert b == z.numel() * 4  # float32 = 4 bytes

    # At 25% compression, sparse cost = 25% × (32+32) bits = 25% × 8 bytes/elem
    # Dense cost = 100% × 4 bytes/elem → sparse is smaller when rho < 0.5
    b_compressed = compute_bytes_transmitted(z, compression_ratio=0.25)
    assert b_compressed < b


def test_topk_compress():
    z = torch.randn(100)
    z_c = topk_compress(z, rho=0.5)
    nnz = (z_c != 0).sum().item()
    assert nnz == 50


# ── Aggregated metrics ───────────────────────────────────────────────────────

def test_metrics_tracker():
    tracker = MetricsTracker()
    for i in range(5):
        tracker.record(RoundMetrics(round_t=i, test_accuracy=0.1 * i))
    assert len(tracker) == 5
    assert tracker.final_accuracy() == pytest.approx(0.4)
    df = tracker.to_dataframe()
    assert len(df) == 5
    assert "test_accuracy" in df.columns


def test_metrics_tracker_save_load(tmp_path):
    tracker = MetricsTracker()
    tracker.record(RoundMetrics(round_t=1, test_accuracy=0.5))
    path = str(tmp_path / "metrics.json")
    tracker.save(path)
    loaded = MetricsTracker.load(path)
    assert loaded.final_accuracy() == pytest.approx(0.5)


# ── Statistical tests ─────────────────────────────────────────────────────────

def test_paired_ttest_significant():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.05, 10).tolist()
    b = rng.normal(0.7, 0.05, 10).tolist()
    res = paired_ttest(a, b, alternative="greater")
    assert res["p_value"] < 0.05
    assert res["cohens_d"] > 0


def test_wilcoxon():
    a = [0.8, 0.82, 0.79, 0.81, 0.83]
    b = [0.70, 0.71, 0.69, 0.72, 0.70]
    res = wilcoxon_test(a, b)
    assert res["p_value"] < 0.05


def test_bonferroni():
    pvals = [0.01, 0.02, 0.03]
    corrected = bonferroni_correct(pvals, m=3)
    assert corrected == pytest.approx([0.03, 0.06, 0.09])


def test_power_analysis():
    power = power_analysis_paired_t(cohens_d=1.0, n=10)
    assert power > 0.5
