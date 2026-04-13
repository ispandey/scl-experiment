"""Table generation for all 10 JSAC tables."""
from __future__ import annotations

import os

import pandas as pd
import numpy as np


def _save_table(df: pd.DataFrame, path: str, caption: str = ""):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    df.to_csv(path.replace(".tex", ".csv"), index=True)
    try:
        with open(path, "w") as f:
            f.write(f"% {caption}\n")
            f.write(df.to_latex(float_format="%.4f"))
    except Exception:
        pass


def make_table_T2(df: pd.DataFrame, output_dir: str):
    """T2: Lipschitz estimates and bound tightness."""
    if "snr_db" not in df.columns:
        return
    cols = {c: "mean" for c in ["Ls", "Ll", "Lg", "bound_tightness"] if c in df.columns}
    if not cols:
        return
    tbl = df.groupby("snr_db").agg(cols).round(4)
    _save_table(tbl, os.path.join(output_dir, "T2_lipschitz.tex"), "Lipschitz Estimates")


def make_table_T3(df: pd.DataFrame, output_dir: str):
    """T3: Semantic capacity measurement."""
    if "snr_db" not in df.columns:
        return
    cols = {c: "mean" for c in ["IZY", "IZtildeY", "delta_ch", "eta_s"] if c in df.columns}
    if not cols:
        return
    tbl = df.groupby(["snr_db", "method"] if "method" in df.columns else "snr_db").agg(cols).round(4)
    _save_table(tbl, os.path.join(output_dir, "T3_semantic_capacity.tex"), "Semantic Capacity")


def make_table_T4(df: pd.DataFrame, output_dir: str, snr_db: float = 15.0):
    """T4: Full robustness matrix."""
    if "attack" not in df.columns or "defense" not in df.columns:
        return
    sub = df[df["snr_db"] == snr_db] if "snr_db" in df.columns else df
    if sub.empty:
        sub = df
    mean_t = sub.groupby(["attack", "defense"])["test_accuracy"].mean().unstack()
    std_t = sub.groupby(["attack", "defense"])["test_accuracy"].std().unstack()
    # Combine as "mean±std"
    combined = mean_t.round(3).astype(str) + " ± " + std_t.round(3).fillna(0).astype(str)
    _save_table(combined, os.path.join(output_dir, "T4_robustness_matrix.tex"), "Robustness Matrix")


def make_table_T5(df: pd.DataFrame, output_dir: str):
    """T5: Multi-dataset generalization."""
    if "dataset" not in df.columns:
        return
    tbl = df.groupby(["dataset", "defense"])["final_accuracy"].mean().unstack().round(4)
    _save_table(tbl, os.path.join(output_dir, "T5_multidataset.tex"), "Multi-Dataset Generalization")


def make_table_T6(df: pd.DataFrame, output_dir: str):
    """T6: Non-IID robustness."""
    if "partition" not in df.columns:
        return
    cols = {c: "mean" for c in ["final_accuracy", "false_positive_rate"] if c in df.columns}
    tbl = df.groupby(["partition", "defense"]).agg(cols).round(4)
    _save_table(tbl, os.path.join(output_dir, "T6_noniid.tex"), "Non-IID Robustness")


def make_table_T7(df: pd.DataFrame, output_dir: str):
    """T7: BCBSA component ablation."""
    if "variant" not in df.columns:
        return
    tbl = df.groupby(["variant", "attack"])["final_accuracy"].mean().unstack().round(4)
    _save_table(tbl, os.path.join(output_dir, "T7_bcbsa_ablation.tex"), "BCBSA Ablation")


def make_table_T8(df: pd.DataFrame, output_dir: str):
    """T8: Coefficient sensitivity heatmap (β × λ)."""
    if "beta" not in df.columns or "lambda" not in df.columns:
        return
    tbl = df.groupby(["beta", "lambda"])["final_accuracy"].mean().unstack().round(4)
    _save_table(tbl, os.path.join(output_dir, "T8_ib_sensitivity.tex"), "IB Coefficient Sensitivity")


def make_table_T9(df: pd.DataFrame, output_dir: str):
    """T9: Communication-accuracy tradeoff."""
    if "rho" not in df.columns:
        return
    cols = {c: "mean" for c in ["final_accuracy", "bytes_tx", "semantic_fidelity"] if c in df.columns}
    tbl = df.groupby("rho").agg(cols).round(4)
    _save_table(tbl, os.path.join(output_dir, "T9_compression.tex"), "Compression Tradeoff")


def make_table_T10(results_dict: dict, output_dir: str):
    """T10: Hypothesis test summary."""
    rows = []
    for test_name, res in results_dict.items():
        rows.append({
            "Test": test_name,
            "p-value": res.get("p_value", "–"),
            "Cohen's d": res.get("cohens_d", "–"),
            "CI_low": res.get("ci_95", ("–", "–"))[0],
            "CI_high": res.get("ci_95", ("–", "–"))[1],
            "n": res.get("n", "–"),
        })
    if not rows:
        return
    tbl = pd.DataFrame(rows).set_index("Test")
    _save_table(tbl, os.path.join(output_dir, "T10_hypothesis_tests.tex"), "Hypothesis Tests")
