#!/usr/bin/env python3
"""
SCL Experiment Runner
=====================
Implements the JSAC-grade SCL experiment suite from jsac_experiment_design.md.

Usage examples
--------------
Dry run (synthetic data, 2 rounds, 1 seed):
    python runner.py --exp eg1 --device cpu --dry_run

Full EG-1 (requires real datasets):
    python runner.py --exp eg1 --device cuda --output_dir results/

Full EG-3A robustness matrix:
    python runner.py --exp eg3a --device cuda --output_dir results/

All experiments:
    python runner.py --exp all --device cuda --output_dir results/
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from scl.config import EG1Config, EG2Config, EG3Config, EG4Config, EG5Config
from scl.experiments.eg1_theorem_calibration import run_eg1
from scl.experiments.eg2_semantic_capacity import run_eg2
from scl.experiments.eg3_robustness_matrix import run_eg3a, run_eg3b, run_eg3c, run_eg3d
from scl.experiments.eg4_generalization import run_eg4a, run_eg4b, run_eg4c
from scl.experiments.eg5_ablation import run_eg5a, run_eg5b, run_eg5c, run_eg5d, run_eg5e


def parse_args():
    p = argparse.ArgumentParser(
        description="SCL JSAC Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--exp",
        required=True,
        choices=[
            "eg1", "eg2",
            "eg3a", "eg3b", "eg3c", "eg3d", "eg3",
            "eg4a", "eg4b", "eg4c", "eg4",
            "eg5a", "eg5b", "eg5c", "eg5d", "eg5e", "eg5",
            "all",
        ],
        help="Experiment group to run.",
    )
    p.add_argument("--output_dir", default="results", help="Root output directory.")
    p.add_argument("--device", default="cpu", help="Compute device (cpu / cuda).")
    p.add_argument("--dry_run", action="store_true",
                   help="Minimal run (2 rounds, 1 seed, 2 SNR) to validate pipeline.")
    p.add_argument("--num_rounds", type=int, default=None,
                   help="Override number of rounds (useful for quick tests).")
    p.add_argument("--num_seeds", type=int, default=None,
                   help="Override number of seeds.")
    return p.parse_args()


def _subdir(output_dir: str, name: str) -> str:
    return os.path.join(output_dir, name)


def main():
    args = parse_args()
    dry = args.dry_run
    device = args.device
    out = args.output_dir

    print(f"\nRunning experiment: '{args.exp}'")
    print(f"  output_dir = {out}")
    print(f"  device     = {device}")
    print(f"  dry_run    = {dry}")

    # ── Build configs (optionally override rounds/seeds) ─────────────────
    def _eg1_cfg():
        c = EG1Config()
        if args.num_rounds: c.num_rounds = args.num_rounds
        if args.num_seeds: c.num_seeds = args.num_seeds
        return c

    def _eg2_cfg():
        c = EG2Config()
        if args.num_rounds: c.num_rounds = args.num_rounds
        if args.num_seeds: c.num_seeds = args.num_seeds
        return c

    def _eg3_cfg():
        c = EG3Config()
        if args.num_rounds: c.num_rounds = args.num_rounds
        if args.num_seeds: c.num_seeds = args.num_seeds
        return c

    def _eg4_cfg():
        c = EG4Config()
        if args.num_rounds: c.num_rounds = args.num_rounds
        if args.num_seeds: c.num_seeds = args.num_seeds
        return c

    def _eg5_cfg():
        c = EG5Config()
        if args.num_rounds: c.num_rounds = args.num_rounds
        if args.num_seeds: c.num_seeds = args.num_seeds
        return c

    # ── Dispatch ──────────────────────────────────────────────────────────
    exp = args.exp

    if exp in ("eg1", "all"):
        run_eg1(_eg1_cfg(), _subdir(out, "eg1"), device=device, dry_run=dry)

    if exp in ("eg2", "all"):
        run_eg2(_eg2_cfg(), _subdir(out, "eg2"), device=device, dry_run=dry)

    if exp in ("eg3a", "eg3", "all"):
        run_eg3a(_eg3_cfg(), _subdir(out, "eg3a"), device=device, dry_run=dry)

    if exp in ("eg3b", "eg3", "all"):
        run_eg3b(_eg3_cfg(), _subdir(out, "eg3b"), device=device, dry_run=dry)

    if exp in ("eg3c", "eg3", "all"):
        run_eg3c(_eg3_cfg(), _subdir(out, "eg3c"), device=device, dry_run=dry)

    if exp in ("eg3d", "eg3", "all"):
        run_eg3d(_eg3_cfg(), _subdir(out, "eg3d"), device=device, dry_run=dry)

    if exp in ("eg4a", "eg4", "all"):
        run_eg4a(_eg4_cfg(), _subdir(out, "eg4a"), device=device, dry_run=dry)

    if exp in ("eg4b", "eg4", "all"):
        run_eg4b(_eg4_cfg(), _subdir(out, "eg4b"), device=device, dry_run=dry)

    if exp in ("eg4c", "eg4", "all"):
        run_eg4c(_eg4_cfg(), _subdir(out, "eg4c"), device=device, dry_run=dry)

    if exp in ("eg5a", "eg5", "all"):
        run_eg5a(_eg5_cfg(), _subdir(out, "eg5a"), device=device, dry_run=dry)

    if exp in ("eg5b", "eg5", "all"):
        run_eg5b(_eg5_cfg(), _subdir(out, "eg5b"), device=device, dry_run=dry)

    if exp in ("eg5c", "eg5", "all"):
        run_eg5c(_eg5_cfg(), _subdir(out, "eg5c"), device=device, dry_run=dry)

    if exp in ("eg5d", "eg5", "all"):
        run_eg5d(_eg5_cfg(), _subdir(out, "eg5d"), device=device, dry_run=dry)

    if exp in ("eg5e", "eg5", "all"):
        run_eg5e(_eg5_cfg(), _subdir(out, "eg5e"), device=device, dry_run=dry)

    print("\nDone.")


if __name__ == "__main__":
    main()
