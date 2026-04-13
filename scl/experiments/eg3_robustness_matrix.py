"""EG-3: Robustness Matrix."""
from __future__ import annotations

import copy
import os
import time
from typing import List

import pandas as pd
import torch

from scl.config import EG3Config
from scl.experiments._utils import build_trainer, save_tracker


def run_eg3a(
    config: EG3Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    EG-3A: Full 6 attacks × 8 defenses × 3 SNR matrix.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    attacks = config.attacks[:2] if dry_run else config.attacks
    defenses = config.defenses[:2] if dry_run else config.defenses
    snr_list = config.snr_list[:1] if dry_run else config.snr_list
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds

    total = len(attacks) * len(defenses) * len(snr_list) * num_seeds
    done = 0

    for atk in attacks:
        for dfn in defenses:
            for snr_db in snr_list:
                seed_accs = []
                for seed in range(num_seeds):
                    done += 1
                    print(
                        f"\n[EG3A] ({done}/{total}) atk={atk} dfn={dfn} "
                        f"snr={snr_db}dB seed={seed}"
                    )
                    trainer, _ = build_trainer(
                        arch="resnet18",
                        split_layer=config.split_layer,
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
                    final_acc = tracker.final_accuracy()
                    seed_accs.append(final_acc)

                    out = os.path.join(output_dir, f"{atk}_{dfn}_snr{snr_db}_s{seed}.json")
                    save_tracker(tracker, out)
                    for m in tracker.history:
                        records.append({
                            "attack": atk,
                            "defense": dfn,
                            "snr_db": snr_db,
                            "seed": seed,
                            **vars(m),
                        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg3a.csv"), index=False)
    print(f"\n[EG3A] Done. Summary → {output_dir}/summary_eg3a.csv")
    return df


def run_eg3b(
    config: EG3Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    EG-3B: Byzantine fraction sweep.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    fractions = [0.10, 0.30] if dry_run else config.fraction_sweep
    attacks = ["weight_poison", "ipm"]
    defenses_b = ["flame", "bcbsa", "median", "fedavg"]
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds

    for frac in fractions:
        for atk in attacks:
            for dfn in defenses_b:
                for seed in range(num_seeds):
                    print(f"[EG3B] frac={frac} atk={atk} dfn={dfn} seed={seed}")
                    trainer, _ = build_trainer(
                        arch="resnet18",
                        split_layer=config.split_layer,
                        dataset_name=config.dataset,
                        partition=config.partition,
                        channel_name=config.channel,
                        attack_name=atk,
                        defense_name=dfn,
                        num_clients=config.num_clients,
                        malicious_fraction=frac,
                        num_rounds=num_rounds,
                        batch_size=config.batch_size,
                        lr=config.lr,
                        device=device,
                        seed=seed,
                    )
                    tracker = trainer.train(snr_db=15.0, total_rounds=num_rounds)
                    for m in tracker.history:
                        records.append({
                            "frac": frac,
                            "attack": atk,
                            "defense": dfn,
                            "seed": seed,
                            **vars(m),
                        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg3b.csv"), index=False)
    return df


def run_eg3c(
    config: EG3Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    EG-3C: Theorem 5 empirical verification — gradient bias of Coord. Median.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    fracs = [0.2, 0.3] if dry_run else [0.2, 0.3, 0.4, 0.5]
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds

    for frac in fracs:
        for seed in range(num_seeds):
            trainer, _ = build_trainer(
                arch="resnet18",
                split_layer=config.split_layer,
                dataset_name=config.dataset,
                partition=config.partition,
                channel_name=config.channel,
                attack_name="weight_poison",
                defense_name="median",
                num_clients=config.num_clients,
                malicious_fraction=frac,
                num_rounds=num_rounds,
                batch_size=config.batch_size,
                lr=config.lr,
                device=device,
                seed=seed,
            )
            tracker = trainer.train(snr_db=15.0, total_rounds=num_rounds)
            bias = tracker.history[-1].gradient_bias_norm if tracker.history else 0.0
            records.append({
                "frac": frac,
                "seed": seed,
                "gradient_bias_norm": bias,
                "final_accuracy": tracker.final_accuracy(),
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg3c.csv"), index=False)
    return df


def run_eg3d(
    config: EG3Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    EG-3D: Runtime-robustness Pareto frontier.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    defenses_d = config.defenses[:3] if dry_run else config.defenses
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds

    for dfn in defenses_d:
        for seed in range(num_seeds):
            t0 = time.time()
            trainer, _ = build_trainer(
                arch="resnet18",
                split_layer=config.split_layer,
                dataset_name=config.dataset,
                partition=config.partition,
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
            tracker = trainer.train(snr_db=15.0, total_rounds=num_rounds)
            wall = time.time() - t0
            records.append({
                "defense": dfn,
                "seed": seed,
                "final_accuracy": tracker.final_accuracy(),
                "total_wall_sec": wall,
                "avg_round_sec": wall / max(1, len(tracker.history)),
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary_eg3d.csv"), index=False)
    return df
