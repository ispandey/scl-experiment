#!/usr/bin/env python3
"""
SCL Experiment Runner
=====================
Implements the JSAC-grade SCL experiment suite from jsac_experiment_design.md.

Hyperparameter priority (highest → lowest):
    CLI flags  >  config.yaml  >  code defaults (scl/config.py)

Usage examples
--------------
Dry run (reduced rounds/seeds to validate pipeline):
    python runner.py --exp eg1 --device cpu --dry_run

Load hyperparameters from YAML (recommended on Colab/Kaggle/Lightning AI):
    python runner.py --exp eg3 --config config.yaml

Full EG-1 on GPU with mixed precision:
    python runner.py --exp eg1 --device cuda --mixed_precision --output_dir results/

Full EG-1 on a specific GPU (e.g. GPU index 2):
    python runner.py --exp eg1 --device cuda --gpu 2 --output_dir results/

Override individual hyperparameters at the CLI:
    python runner.py --exp eg3a --config config.yaml --batch_size 128 --lr 0.05

All experiments:
    python runner.py --exp all --config config.yaml --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Any, Dict, Optional

import pandas as pd

from scl.config import EG1Config, EG2Config, EG3Config, EG4Config, EG5Config
from scl.experiments.eg1_theorem_calibration import run_eg1
from scl.experiments.eg2_semantic_capacity import run_eg2
from scl.experiments.eg3_robustness_matrix import run_eg3a, run_eg3b, run_eg3c, run_eg3d
from scl.experiments.eg4_generalization import run_eg4a, run_eg4b, run_eg4c
from scl.experiments.eg5_ablation import run_eg5a, run_eg5b, run_eg5c, run_eg5d, run_eg5e
from scl.experiments._utils import set_global_opts


def resolve_device(device: str, gpu: Optional[int] = None) -> str:
    """
    Resolve the compute device string, with automatic GPU selection when needed.

    Rules:
    - "cpu"          → "cpu"
    - "cuda:N"       → "cuda:N" (explicit index, honoured as-is)
    - --gpu N        → "cuda:N" (CLI override; takes priority over --device index)
    - "cuda" (bare)  → auto-select the GPU with the most free memory across ALL
                       running processes (uses torch.cuda.mem_get_info which
                       queries the driver, not just the current process).

    When CUDA is requested but unavailable, falls back to CPU with a warning.
    """
    import torch

    if device == "cpu":
        return "cpu"

    if not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available – falling back to CPU.")
        return "cpu"

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        warnings.warn("No CUDA devices found – falling back to CPU.")
        return "cpu"

    # --gpu N takes highest priority
    if gpu is not None:
        if gpu >= n_gpus:
            raise ValueError(
                f"--gpu {gpu} is out of range; only {n_gpus} GPU(s) available "
                f"(indices 0 – {n_gpus - 1})."
            )
        return f"cuda:{gpu}"

    # Explicit cuda:N in --device
    if device.startswith("cuda:"):
        idx = int(device.split(":")[1])
        if idx >= n_gpus:
            raise ValueError(
                f"--device {device} is out of range; only {n_gpus} GPU(s) available."
            )
        return device

    # Auto-select: pick GPU with the most free memory reported by the driver.
    # torch.cuda.mem_get_info(i) returns (free_bytes, total_bytes) and accounts
    # for memory held by *all* processes on that GPU, not just the current one.
    if n_gpus == 1:
        return "cuda:0"

    free_mem = []
    for i in range(n_gpus):
        free_bytes, _ = torch.cuda.mem_get_info(i)
        free_mem.append(free_bytes)

    best = int(max(range(n_gpus), key=lambda i: free_mem[i]))
    print(
        f"  [auto-GPU] Available GPUs and free memory:"
    )
    for i in range(n_gpus):
        marker = " ← selected" if i == best else ""
        props = torch.cuda.get_device_properties(i)
        print(
            f"    GPU {i}: {props.name:30s}  free={free_mem[i]/1024**3:.1f} GB{marker}"
        )
    return f"cuda:{best}"


def _print_device_info(device: str) -> None:
    """Print GPU info when a CUDA device is selected."""
    import torch

    if not device.startswith("cuda"):
        return
    idx = int(device.split(":")[1]) if ":" in device else 0
    props = torch.cuda.get_device_properties(idx)
    total_gb = props.total_memory / 1024 ** 3
    free_bytes, _ = torch.cuda.mem_get_info(idx)
    free_gb = free_bytes / 1024 ** 3
    print(
        f"  GPU {idx}: {props.name}  |  total={total_gb:.1f} GB  "
        f"|  free≈{free_gb:.1f} GB  |  SM×{props.multi_processor_count}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="SCL JSAC Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Experiment selection ──────────────────────────────────────────────
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
    # ── Infrastructure ────────────────────────────────────────────────────
    p.add_argument("--output_dir", default=None,
                   help="Root output directory (default: 'results' or config.yaml value).")
    p.add_argument(
        "--device", default=None,
        help="Compute device: 'cpu', 'cuda' (auto-select GPU), or 'cuda:N'. "
             "Falls back to config.yaml 'device', then 'cpu'.",
    )
    p.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index to use (overrides the index in --device). "
             "Ignored when --device cpu.",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to a YAML config file (e.g. config.yaml). "
             "Values override code defaults; CLI flags override the YAML.",
    )
    # ── Run control ───────────────────────────────────────────────────────
    p.add_argument("--dry_run", action="store_true",
                   help="Minimal run (2 rounds, 1 seed, 2 SNR) to validate pipeline.")
    p.add_argument("--num_rounds", type=int, default=None,
                   help="Override number of rounds (useful for quick tests).")
    p.add_argument("--num_seeds", type=int, default=None,
                   help="Override number of seeds.")
    # ── Hyperparameters (apply to all experiments unless overridden in YAML) ─
    p.add_argument("--batch_size", type=int, default=None,
                   help="Mini-batch size per client.")
    p.add_argument("--lr", type=float, default=None,
                   help="Initial SGD learning rate.")
    p.add_argument("--num_clients", type=int, default=None,
                   help="Total number of federated clients.")
    p.add_argument("--malicious_fraction", type=float, default=None,
                   help="Fraction of clients that are Byzantine (0.0 – 0.5).")
    p.add_argument("--split_layer", type=int, default=None, choices=[1, 2, 3],
                   help="ResNet-18 split point (1=early, 2=mid, 3=late).")
    # ── GPU / precision ───────────────────────────────────────────────────
    p.add_argument(
        "--mixed_precision", action="store_true", default=None,
        help=(
            "Enable automatic mixed-precision (AMP) training for GPU speedup. "
            "Uses bfloat16 on Ampere+ GPUs (A100, RTX 30/40 series), float16 "
            "on older hardware (T4, V100, P100). No effect on CPU."
        ),
    )
    p.add_argument(
        "--data_root", type=str, default=None,
        help=(
            "Root directory for dataset downloads and caching "
            "(default: ~/.cache/scl_data). "
            "Set to a fast local path on Colab (/content/data) or "
            "Kaggle (/kaggle/working/data) to avoid repeated downloads."
        ),
    )
    return p.parse_args()


def _subdir(output_dir: str, name: str) -> str:
    return os.path.join(output_dir, name)


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file; return empty dict on failure."""
    try:
        import yaml
    except ImportError:
        warnings.warn(
            "PyYAML is not installed; --config will be ignored. "
            "Run: pip install PyYAML"
        )
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _apply_section(cfg, section: Dict[str, Any]) -> None:
    """Write values from *section* dict into dataclass *cfg* (skips unknowns)."""
    for key, val in section.items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)


def _apply_cli_overrides(cfg, args) -> None:
    """Apply explicit CLI hyperparameter flags on top of the config object."""
    overrides = {
        "num_rounds": args.num_rounds,
        "num_seeds": args.num_seeds,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_clients": args.num_clients,
        "malicious_fraction": args.malicious_fraction,
        "split_layer": args.split_layer,
    }
    for attr, val in overrides.items():
        if val is not None and hasattr(cfg, attr):
            setattr(cfg, attr, val)


def main():
    args = parse_args()
    dry = args.dry_run

    # ── Load YAML config (if provided) ───────────────────────────────────
    yaml_cfg: Dict[str, Any] = {}
    if args.config:
        yaml_cfg = _load_yaml(args.config)

    # ── Resolve device (CLI > yaml > default 'cpu') ───────────────────────
    raw_device = args.device or yaml_cfg.get("device", "cpu")
    device = resolve_device(raw_device, args.gpu)

    # ── Resolve output dir (CLI > yaml > 'results') ───────────────────────
    out = args.output_dir or yaml_cfg.get("output_dir", "results")

    # ── Resolve mixed_precision and data_root (CLI > yaml) ───────────────
    # argparse stores None (not False) when --mixed_precision is absent because
    # we set default=None; the yaml value fills the gap if present.
    if args.mixed_precision:
        mixed_precision = True
    else:
        mixed_precision = bool(yaml_cfg.get("mixed_precision", False))

    data_root = args.data_root or yaml_cfg.get("data_root") or None

    # Configure dataset root and AMP before any experiment code runs.
    set_global_opts(data_root=data_root, mixed_precision=mixed_precision)

    # Set CUDA memory allocator config to reduce fragmentation before any
    # CUDA context is created.  512 MB is a practical upper bound on the
    # largest single allocation in this workload (smash tensors, gradient
    # vectors); smaller blocks are split more aggressively, keeping the
    # remaining free space contiguous for subsequent allocations.
    if device.startswith("cuda") and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    print(f"\nRunning experiment: '{args.exp}'")
    print(f"  config           = {args.config or '(none)'}")
    print(f"  output_dir       = {out}")
    print(f"  device           = {device}")
    print(f"  dry_run          = {dry}")
    print(f"  mixed_precision  = {mixed_precision}")
    print(f"  data_root        = {data_root or '~/.cache/scl_data (default)'}")
    _print_device_info(device)

    # ── Build configs: code defaults → yaml section → CLI overrides ──────
    def _eg1_cfg():
        c = EG1Config()
        _apply_section(c, yaml_cfg.get("eg1", {}))
        _apply_cli_overrides(c, args)
        return c

    def _eg2_cfg():
        c = EG2Config()
        _apply_section(c, yaml_cfg.get("eg2", {}))
        _apply_cli_overrides(c, args)
        return c

    def _eg3_cfg():
        c = EG3Config()
        _apply_section(c, yaml_cfg.get("eg3", {}))
        _apply_cli_overrides(c, args)
        return c

    def _eg4_cfg():
        c = EG4Config()
        _apply_section(c, yaml_cfg.get("eg4", {}))
        _apply_cli_overrides(c, args)
        return c

    def _eg5_cfg():
        c = EG5Config()
        _apply_section(c, yaml_cfg.get("eg5", {}))
        _apply_cli_overrides(c, args)
        return c

    # ── Dispatch ──────────────────────────────────────────────────────────
    import torch as _torch

    def _cuda_gc():
        """Release unused cached GPU memory between experiment groups."""
        if device.startswith("cuda") and _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    exp = args.exp

    if exp in ("eg1", "all"):
        run_eg1(_eg1_cfg(), _subdir(out, "eg1"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg2", "all"):
        run_eg2(_eg2_cfg(), _subdir(out, "eg2"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg3a", "eg3", "all"):
        run_eg3a(_eg3_cfg(), _subdir(out, "eg3a"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg3b", "eg3", "all"):
        run_eg3b(_eg3_cfg(), _subdir(out, "eg3b"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg3c", "eg3", "all"):
        run_eg3c(_eg3_cfg(), _subdir(out, "eg3c"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg3d", "eg3", "all"):
        run_eg3d(_eg3_cfg(), _subdir(out, "eg3d"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg4a", "eg4", "all"):
        run_eg4a(_eg4_cfg(), _subdir(out, "eg4a"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg4b", "eg4", "all"):
        run_eg4b(_eg4_cfg(), _subdir(out, "eg4b"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg4c", "eg4", "all"):
        run_eg4c(_eg4_cfg(), _subdir(out, "eg4c"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg5a", "eg5", "all"):
        run_eg5a(_eg5_cfg(), _subdir(out, "eg5a"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg5b", "eg5", "all"):
        run_eg5b(_eg5_cfg(), _subdir(out, "eg5b"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg5c", "eg5", "all"):
        run_eg5c(_eg5_cfg(), _subdir(out, "eg5c"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg5d", "eg5", "all"):
        run_eg5d(_eg5_cfg(), _subdir(out, "eg5d"), device=device, dry_run=dry)
        _cuda_gc()

    if exp in ("eg5e", "eg5", "all"):
        run_eg5e(_eg5_cfg(), _subdir(out, "eg5e"), device=device, dry_run=dry)
        _cuda_gc()

    print("\nDone.")


if __name__ == "__main__":
    main()
