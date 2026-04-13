"""EG-4: Generalization experiments."""
from __future__ import annotations

import os

import pandas as pd

from scl.config import EG4Config
from scl.experiments._utils import build_trainer, save_tracker


def run_eg4a(
    config: EG4Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """EG-4A: Multi-dataset generalization."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    datasets = config.datasets[:2] if dry_run else config.datasets
    defenses = config.defenses[:2] if dry_run else ["fedavg", "flame", "bcbsa"]
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds

    for ds in datasets:
        for dfn in defenses:
            for seed in range(num_seeds):
                print(f"[EG4A] ds={ds} dfn={dfn} seed={seed}")
                trainer, _ = build_trainer(
                    arch=config.arch_list[0],
                    split_layer=config.split_layer,
                    dataset_name=ds,
                    partition="iid",
                    channel_name=config.channel,
                    attack_name=config.attack,
                    defense_name=dfn,
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
                    "dataset": ds,
                    "defense": dfn,
                    "seed": seed,
                    "final_accuracy": tracker.final_accuracy(),
                    "auc": tracker.auc_accuracy(),
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg4a.csv"), index=False)
    return df


def run_eg4b(
    config: EG4Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """EG-4B: Data heterogeneity (non-IID)."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    partitions = ["iid", "dir05"] if dry_run else ["iid", "dir05", "dir01"]
    defenses = ["fedavg", "flame", "bcbsa"] if not dry_run else ["fedavg", "bcbsa"]
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds

    for part in partitions:
        for dfn in defenses:
            for seed in range(num_seeds):
                print(f"[EG4B] partition={part} dfn={dfn} seed={seed}")
                trainer, _ = build_trainer(
                    arch=config.arch_list[0],
                    split_layer=config.split_layer,
                    dataset_name="cifar10",
                    partition=part,
                    channel_name=config.channel,
                    attack_name="weight_poison",
                    defense_name=dfn,
                    num_clients=config.num_clients,
                    malicious_fraction=config.malicious_fraction,
                    num_rounds=num_rounds,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    device=device,
                    seed=seed,
                )
                tracker = trainer.train(snr_db=config.snr_fixed, total_rounds=num_rounds)
                last = tracker.history[-1] if tracker.history else None
                records.append({
                    "partition": part,
                    "defense": dfn,
                    "seed": seed,
                    "final_accuracy": tracker.final_accuracy(),
                    "false_positive_rate": last.false_positive_rate if last else 0.0,
                    "acceptance_ratio": last.acceptance_ratio if last else 1.0,
                })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg4b.csv"), index=False)
    return df


def run_eg4c(
    config: EG4Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """EG-4C: Architecture generalization."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    archs = ["resnet18", "mobilenetv2"]
    snr_list = [15.0] if dry_run else [5.0, 15.0, 25.0]
    attacks = ["none", "weight_poison"]
    defenses = ["fedavg", "flame", "bcbsa"]
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds

    for arch in archs:
        for snr_db in snr_list:
            for atk in attacks:
                for dfn in defenses:
                    for seed in range(num_seeds):
                        print(f"[EG4C] arch={arch} snr={snr_db} atk={atk} dfn={dfn} seed={seed}")
                        try:
                            trainer, _ = build_trainer(
                                arch=arch,
                                split_layer=config.split_layer,
                                dataset_name="cifar10",
                                partition="iid",
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
                            records.append({
                                "arch": arch,
                                "snr_db": snr_db,
                                "attack": atk,
                                "defense": dfn,
                                "seed": seed,
                                "final_accuracy": tracker.final_accuracy(),
                            })
                        except Exception as e:
                            print(f"  SKIP: {e}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg4c.csv"), index=False)
    return df
