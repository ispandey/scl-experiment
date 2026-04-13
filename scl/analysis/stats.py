"""Statistical testing utilities."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


def paired_ttest(
    a: List[float],
    b: List[float],
    alternative: str = "greater",
) -> dict:
    """
    One-sided paired t-test.

    Returns dict with keys: t_stat, p_value, cohens_d, ci_95, n.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    d = a - b
    n = len(d)
    if n < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0, "ci_95": (0.0, 0.0), "n": n}

    t_stat, p_two = stats.ttest_rel(a, b)

    if alternative == "greater":
        p_one = p_two / 2.0 if t_stat > 0 else 1.0 - p_two / 2.0
    elif alternative == "less":
        p_one = p_two / 2.0 if t_stat < 0 else 1.0 - p_two / 2.0
    else:
        p_one = p_two

    cohens_d = d.mean() / (d.std(ddof=1) + 1e-12)
    se = d.std(ddof=1) / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = d.mean() - t_crit * se
    ci_high = d.mean() + t_crit * se

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_one),
        "cohens_d": float(cohens_d),
        "ci_95": (float(ci_low), float(ci_high)),
        "n": n,
    }


def wilcoxon_test(
    a: List[float],
    b: List[float],
    alternative: str = "greater",
) -> dict:
    """Paired Wilcoxon signed-rank test."""
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    d = a - b
    if len(d) < 1 or np.all(d == 0):
        return {"stat": 0.0, "p_value": 1.0}
    try:
        stat, p = stats.wilcoxon(d, alternative=alternative, zero_method="wilcox")
    except Exception:
        stat, p = 0.0, 1.0
    return {"stat": float(stat), "p_value": float(p)}


def bonferroni_correct(p_values: List[float], m: Optional[int] = None) -> List[float]:
    """Bonferroni correction: adjusted p = min(p·m, 1)."""
    if m is None:
        m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def effect_size_cohens_d(a: List[float], b: List[float]) -> float:
    d = np.array(a) - np.array(b)
    return float(d.mean() / (d.std(ddof=1) + 1e-12))


def power_analysis_paired_t(
    cohens_d: float, n: int, alpha: float = 0.05
) -> float:
    """Estimated power for one-sided paired t-test."""
    if n < 2:
        return 0.0
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha, df=df)
    ncp = abs(cohens_d) * math.sqrt(n)
    power = 1.0 - stats.t.cdf(t_crit, df=df, loc=ncp)
    return float(power)


def summarise_group(
    accs: List[float],
) -> dict:
    a = np.array(accs, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "n": len(a),
    }
