"""Figure generation for all 13 JSAC figures."""
from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_FIG_DPI = 150


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ── F2: Excess loss vs SNR (EG-1) ────────────────────────────────────────────

def plot_eg1_excess_loss(df: pd.DataFrame, output_dir: str):
    """F2: Excess loss Δℓ vs SNR (dB), measured vs theoretical bound."""
    fig, ax = plt.subplots()
    if "snr_db" in df.columns and "excess_loss" in df.columns:
        grp = df.groupby("snr_db")["excess_loss"].agg(["mean", "std"]).reset_index()
        ax.errorbar(grp["snr_db"], grp["mean"], yerr=grp["std"],
                    label="Measured Δℓ", marker="o")
        if "bound" in df.columns:
            grp_b = df.groupby("snr_db")["bound"].mean().reset_index()
            ax.plot(grp_b["snr_db"], grp_b["bound"],
                    linestyle="--", label="Theorem 1 Bound")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Excess Loss")
    ax.set_title("F2: Theorem 1 Calibration — Excess Loss vs SNR")
    ax.legend()
    ax.set_yscale("log")
    _save(fig, os.path.join(output_dir, "F2_excess_loss_vs_snr.pdf"))


# ── F3: Gradient noise decomposition (EG-1) ───────────────────────────────────

def plot_eg1_grad_noise(df: pd.DataFrame, output_dir: str):
    """F3: Gradient variance decomposition (channel vs data) over SNR."""
    fig, ax = plt.subplots()
    if "snr_db" in df.columns:
        for col, label in [("grad_var_channel", "Channel"), ("grad_var_data", "Data")]:
            if col in df.columns:
                grp = df.groupby("snr_db")[col].mean()
                ax.plot(grp.index, grp.values, marker="o", label=label)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Gradient Variance")
    ax.set_title("F3: Gradient Noise Decomposition")
    ax.legend()
    _save(fig, os.path.join(output_dir, "F3_grad_noise.pdf"))


# ── F4: I(Z;Y) and I(Z̃;Y) vs SNR (EG-2) ─────────────────────────────────────

def plot_eg2_mutual_info(df: pd.DataFrame, output_dir: str):
    """F4: Mutual information proxies vs SNR."""
    fig, ax = plt.subplots()
    if "snr_db" in df.columns and "method" in df.columns:
        for method, grp in df.groupby("method"):
            by_snr = grp.groupby("snr_db")
            if "IZY" in df.columns:
                ax.plot(
                    by_snr["IZY"].mean().index,
                    by_snr["IZY"].mean().values,
                    linestyle="--", label=f"{method} I(Z;Y)"
                )
            if "IZtildeY" in df.columns:
                ax.plot(
                    by_snr["IZtildeY"].mean().index,
                    by_snr["IZtildeY"].mean().values,
                    linestyle="-", label=f"{method} I(Z̃;Y)"
                )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MI Proxy (nats)")
    ax.set_title("F4: Semantic Capacity — I(Z;Y) and I(Z̃;Y) vs SNR")
    ax.legend(fontsize=7)
    _save(fig, os.path.join(output_dir, "F4_mutual_info.pdf"))


# ── F5: η_s vs SNR (EG-2) ────────────────────────────────────────────────────

def plot_eg2_semantic_efficiency(df: pd.DataFrame, output_dir: str):
    """F5: Semantic efficiency η_s vs SNR."""
    fig, ax = plt.subplots()
    if "snr_db" in df.columns and "method" in df.columns and "eta_s" in df.columns:
        for method, grp in df.groupby("method"):
            by_snr = grp.groupby("snr_db")["eta_s"].mean()
            ax.plot(by_snr.index, by_snr.values, marker="o", label=method)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("η_s = I(Z̃;Y) / C(SNR)")
    ax.set_title("F5: Semantic Efficiency vs SNR")
    ax.legend()
    _save(fig, os.path.join(output_dir, "F5_semantic_efficiency.pdf"))


# ── F6: Robustness heatmap (EG-3A) ───────────────────────────────────────────

def plot_eg3_robustness_heatmap(df: pd.DataFrame, output_dir: str, snr_db: float = 15.0):
    """F6: 6 attacks × 8 defenses accuracy heatmap at SNR=15dB."""
    if "attack" not in df.columns or "defense" not in df.columns:
        return
    sub = df[df["snr_db"] == snr_db] if "snr_db" in df.columns else df
    if sub.empty:
        sub = df
    pivot = sub.groupby(["attack", "defense"])["test_accuracy"].mean().unstack(fill_value=0.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, vmin=0, vmax=1)
    ax.set_title(f"F6: Robustness Matrix — Accuracy at SNR={snr_db} dB")
    _save(fig, os.path.join(output_dir, "F6_robustness_heatmap.pdf"))


# ── F7: Convergence trajectories (EG-3A) ─────────────────────────────────────

def plot_eg3_convergence(df: pd.DataFrame, output_dir: str):
    """F7: Accuracy vs round for selected configurations."""
    if "round_t" not in df.columns:
        return
    fig, ax = plt.subplots()
    if "defense" in df.columns:
        for dfn, grp in df.groupby("defense"):
            by_round = grp.groupby("round_t")["test_accuracy"].mean()
            ax.plot(by_round.index, by_round.values, label=dfn)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("F7: Convergence Trajectories")
    ax.legend(fontsize=7)
    _save(fig, os.path.join(output_dir, "F7_convergence.pdf"))


# ── F8: Byzantine fraction sweep (EG-3B) ─────────────────────────────────────

def plot_eg3_byzantine_sweep(df: pd.DataFrame, output_dir: str):
    """F8: Accuracy vs f/N per defense."""
    if "frac" not in df.columns:
        return
    fig, ax = plt.subplots()
    for dfn, grp in df.groupby("defense"):
        by_frac = grp.groupby("frac")["final_accuracy"].mean()
        ax.plot(by_frac.index, by_frac.values, marker="o", label=dfn)
    ax.set_xlabel("Byzantine Fraction f/N")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("F8: Byzantine Fraction Sweep")
    ax.legend()
    _save(fig, os.path.join(output_dir, "F8_byzantine_sweep.pdf"))


# ── F9: Theorem 5 verification (EG-3C) ───────────────────────────────────────

def plot_eg3_theorem5(df: pd.DataFrame, output_dir: str):
    """F9: Gradient bias ‖ĝ−g‖ vs f/N for Coord. Median."""
    if "frac" not in df.columns or "gradient_bias_norm" not in df.columns:
        return
    fig, ax = plt.subplots()
    grp = df.groupby("frac")["gradient_bias_norm"].agg(["mean", "std"])
    ax.errorbar(grp.index, grp["mean"], yerr=grp["std"], marker="o", label="Measured ‖ĝ−g‖")
    ax.axhline(y=1.0, linestyle="--", color="red", label="Theorem 5 bound (‖b‖=1)")
    ax.set_xlabel("Byzantine Fraction f/N")
    ax.set_ylabel("Gradient Bias Norm")
    ax.set_title("F9: Theorem 5 Empirical Verification")
    ax.legend()
    _save(fig, os.path.join(output_dir, "F9_theorem5.pdf"))


# ── F10: Pareto frontier (EG-3D) ─────────────────────────────────────────────

def plot_eg3_pareto(df: pd.DataFrame, output_dir: str):
    """F10: Robustness vs runtime Pareto frontier."""
    if "defense" not in df.columns or "final_accuracy" not in df.columns:
        return
    fig, ax = plt.subplots()
    grp = df.groupby("defense").agg(
        acc=("final_accuracy", "mean"),
        rt=("avg_round_sec", "mean") if "avg_round_sec" in df.columns else ("final_accuracy", "count"),
    ).reset_index()
    ax.scatter(grp["rt"], grp["acc"], s=80)
    for _, row in grp.iterrows():
        ax.annotate(row["defense"], (row["rt"], row["acc"]), fontsize=8)
    ax.set_xlabel("Avg Round Time (s)")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("F10: Robustness-Runtime Pareto Frontier")
    _save(fig, os.path.join(output_dir, "F10_pareto.pdf"))


# ── F11: Non-IID accuracy + FPR (EG-4B) ─────────────────────────────────────

def plot_eg4_noniid(df: pd.DataFrame, output_dir: str):
    """F11: Accuracy and false-positive rate under non-IID partitions."""
    if "partition" not in df.columns:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    width = 0.2
    partitions = df["partition"].unique()
    defenses_u = df["defense"].unique()
    x = np.arange(len(partitions))

    for i, dfn in enumerate(defenses_u):
        accs = [
            df[(df["partition"] == p) & (df["defense"] == dfn)]["final_accuracy"].mean()
            for p in partitions
        ]
        ax1.bar(x + i * width, accs, width, label=dfn)

    ax1.set_xticks(x + width * (len(defenses_u) - 1) / 2)
    ax1.set_xticklabels(partitions)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("F11a: Accuracy under Non-IID")
    ax1.legend(fontsize=7)

    if "false_positive_rate" in df.columns:
        for i, dfn in enumerate(defenses_u):
            fprs = [
                df[(df["partition"] == p) & (df["defense"] == dfn)]["false_positive_rate"].mean()
                for p in partitions
            ]
            ax2.bar(x + i * width, fprs, width, label=dfn)
        ax2.set_xticks(x + width * (len(defenses_u) - 1) / 2)
        ax2.set_xticklabels(partitions)
        ax2.set_ylabel("False Positive Rate")
        ax2.set_title("F11b: FPR under Non-IID")
        ax2.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "F11_noniid.pdf"))


# ── F12: Split-point ablation (EG-5A) ────────────────────────────────────────

def plot_eg5_split_ablation(df: pd.DataFrame, output_dir: str):
    """F12: Accuracy and bytes per split point."""
    if "split_layer" not in df.columns:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    grp = df.groupby("split_layer").agg(
        acc=("final_accuracy", "mean"),
        btx=("bytes_tx", "mean"),
    )
    ax1.bar(grp.index.astype(str), grp["acc"])
    ax1.set_xlabel("Split Layer")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("F12a: Accuracy vs Split Point")

    ax2.bar(grp.index.astype(str), grp["btx"])
    ax2.set_xlabel("Split Layer")
    ax2.set_ylabel("Bytes Transmitted")
    ax2.set_title("F12b: Communication Cost vs Split Point")

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "F12_split_ablation.pdf"))


# ── F13: BCBSA component ablation (EG-5C) ────────────────────────────────────

def plot_eg5_bcbsa_ablation(df: pd.DataFrame, output_dir: str):
    """F13: BCBSA ablation accuracy under A1 and A4."""
    if "variant" not in df.columns or "attack" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot = df.groupby(["variant", "attack"])["final_accuracy"].mean().unstack(fill_value=0.0)
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("BCBSA Variant")
    ax.set_ylabel("Final Accuracy")
    ax.set_title("F13: BCBSA Component Ablation")
    ax.legend(title="Attack")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, os.path.join(output_dir, "F13_bcbsa_ablation.pdf"))
