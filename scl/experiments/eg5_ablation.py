"""EG-5: Ablation and Sensitivity experiments."""
from __future__ import annotations

import os

import pandas as pd

from scl.config import EG5Config
from scl.experiments._utils import build_trainer, save_tracker
from scl.channels import get_channel
from scl.metrics.communication import compute_bytes_transmitted
import torch


def run_eg5a(
    config: EG5Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """EG-5A: Split-point ablation."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    split_layers = [2] if dry_run else config.split_layers
    snr_list = [15.0] if dry_run else config.snr_list
    attacks = ["none", "weight_poison"]
    defenses = ["fedavg", "bcbsa"]
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds

    for sl in split_layers:
        for snr_db in snr_list:
            for atk in attacks:
                for dfn in defenses:
                    for seed in range(num_seeds):
                        print(f"[EG5A] split={sl} snr={snr_db} atk={atk} dfn={dfn} seed={seed}")
                        trainer, _ = build_trainer(
                            arch="resnet18",
                            split_layer=sl,
                            dataset_name=config.dataset,
                            partition=config.partition,
                            channel_name=config.channel,
                            attack_name=atk,
                            defense_name=dfn,
                            num_clients=config.num_clients,
                            malicious_fraction=config.malicious_fraction,
                            num_rounds=num_rounds,
                            batch_size=config.batch_size,
                            lr=config.lr,
                            device=device,
                            seed=seed,
                        )
                        tracker = trainer.train(snr_db=snr_db, total_rounds=num_rounds)
                        last = tracker.history[-1] if tracker.history else None
                        records.append({
                            "split_layer": sl,
                            "snr_db": snr_db,
                            "attack": atk,
                            "defense": dfn,
                            "seed": seed,
                            "final_accuracy": tracker.final_accuracy(),
                            "bytes_tx": last.bytes_transmitted if last else 0,
                        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg5a.csv"), index=False)
    return df


def run_eg5b(
    config: EG5Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """EG-5B: Channel model comparison."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    channels = ["rayleigh", "awgn"] if dry_run else [
        "rayleigh", "awgn", "rician", "bpsk", "noiseless"
    ]
    snr_list = [15.0] if dry_run else config.snr_list
    num_seeds = 1 if dry_run else 3
    num_rounds = 2 if dry_run else config.num_rounds

    for ch_name in channels:
        for snr_db in snr_list:
            for seed in range(num_seeds):
                print(f"[EG5B] channel={ch_name} snr={snr_db} seed={seed}")
                trainer, _ = build_trainer(
                    arch="resnet18",
                    split_layer=2,
                    dataset_name=config.dataset,
                    partition=config.partition,
                    channel_name=ch_name,
                    attack_name="none",
                    defense_name="fedavg",
                    num_clients=config.num_clients,
                    malicious_fraction=0.0,
                    num_rounds=num_rounds,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    device=device,
                    seed=seed,
                )
                tracker = trainer.train(snr_db=snr_db, total_rounds=num_rounds)
                records.append({
                    "channel": ch_name,
                    "snr_db": snr_db,
                    "seed": seed,
                    "final_accuracy": tracker.final_accuracy(),
                    "semantic_fidelity": tracker.history[-1].semantic_fidelity if tracker.history else 0.0,
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg5b.csv"), index=False)
    return df


def run_eg5c(
    config: EG5Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """EG-5C: BCBSA component ablation."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    variants = ["bcbsa_full", "bcbsa_nosem"] if dry_run else config.bcbsa_variants
    attacks = ["weight_poison", "ipm"]
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds

    for variant in variants:
        for atk in attacks:
            for seed in range(num_seeds):
                print(f"[EG5C] variant={variant} atk={atk} seed={seed}")
                trainer, _ = build_trainer(
                    arch="resnet18",
                    split_layer=2,
                    dataset_name=config.dataset,
                    partition=config.partition,
                    channel_name=config.channel,
                    attack_name=atk,
                    defense_name=variant,
                    num_clients=config.num_clients,
                    malicious_fraction=config.malicious_fraction,
                    num_rounds=num_rounds,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    device=device,
                    seed=seed,
                )
                tracker = trainer.train(snr_db=config.snr_fixed, total_rounds=num_rounds)
                records.append({
                    "variant": variant,
                    "attack": atk,
                    "seed": seed,
                    "final_accuracy": tracker.final_accuracy(),
                    "acceptance_ratio": tracker.history[-1].acceptance_ratio if tracker.history else 1.0,
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg5c.csv"), index=False)
    return df


def run_eg5d(
    config: EG5Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """EG-5D: IB coefficient sensitivity (β × λ heatmap)."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    beta_list = [0.1, 1.0] if dry_run else config.beta_sweep
    lambda_list = [1e-3, 1e-2] if dry_run else config.lambda_sweep
    num_seeds = 1 if dry_run else 3
    num_rounds = 2 if dry_run else config.num_rounds

    for beta in beta_list:
        for lam in lambda_list:
            for seed in range(num_seeds):
                print(f"[EG5D] beta={beta} lambda={lam} seed={seed}")
                trainer, _ = build_trainer(
                    arch="resnet18",
                    split_layer=2,
                    dataset_name=config.dataset,
                    partition=config.partition,
                    channel_name=config.channel,
                    attack_name="none",
                    defense_name="fedavg",
                    num_clients=config.num_clients,
                    malicious_fraction=0.0,
                    num_rounds=num_rounds,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    device=device,
                    seed=seed,
                )
                # Note: beta/lambda affect IB training; in full run pass to SFLServer
                tracker = trainer.train(snr_db=config.snr_fixed, total_rounds=num_rounds)
                records.append({
                    "beta": beta,
                    "lambda": lam,
                    "seed": seed,
                    "final_accuracy": tracker.final_accuracy(),
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg5d.csv"), index=False)
    return df


def run_eg5e(
    config: EG5Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """EG-5E: Compression ratio sensitivity."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    rho_list = [0.5, 0.95] if dry_run else config.compression_sweep
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds

    for rho in rho_list:
        for seed in range(num_seeds):
            print(f"[EG5E] rho={rho} seed={seed}")
            trainer, _ = build_trainer(
                arch="resnet18",
                split_layer=2,
                dataset_name=config.dataset,
                partition=config.partition,
                channel_name=config.channel,
                attack_name="none",
                defense_name="fedavg",
                num_clients=config.num_clients,
                malicious_fraction=0.0,
                num_rounds=num_rounds,
                batch_size=config.batch_size,
                lr=config.lr,
                device=device,
                seed=seed,
                compression_ratio=rho,
            )
            tracker = trainer.train(snr_db=config.snr_fixed, total_rounds=num_rounds)
            last = tracker.history[-1] if tracker.history else None
            records.append({
                "rho": rho,
                "seed": seed,
                "final_accuracy": tracker.final_accuracy(),
                "bytes_tx": last.bytes_transmitted if last else 0,
                "semantic_fidelity": last.semantic_fidelity if last else 0.0,
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg5e.csv"), index=False)
    return df
