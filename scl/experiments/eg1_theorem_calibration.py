"""EG-1: Theorem 1 & 3 Calibration."""
from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from scl.config import EG1Config
from scl.data.datasets import get_dataset
from scl.metrics.lipschitz import (
    estimate_Ls,
    estimate_Ll,
    estimate_Lg,
    compute_theorem1_bound,
)
from scl.metrics.aggregated import RoundMetrics
from scl.experiments._utils import build_trainer, set_seed, save_tracker


def run_eg1(
    config: EG1Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    EG-1: For each seed and SNR, train under Rayleigh channel with FedAvg
    and measure Theorem 1 bound calibration.

    Returns a summary DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    snr_list = config.snr_list[:2] if dry_run else config.snr_list
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds
    num_clients = 4 if dry_run else config.num_clients
    lip_pairs = (5, 5, 3) if dry_run else (100, 50, 20)  # (Ls, Ll, Lg) n_pairs

    for seed in range(num_seeds):
        for snr_db in snr_list:
            print(f"\n[EG1] seed={seed} snr={snr_db} dB")
            trainer, test_loader = build_trainer(
                arch=config.arch,
                split_layer=config.split_layer,
                dataset_name=config.dataset,
                partition=config.partition,
                channel_name=config.channel,
                attack_name=config.attack,
                defense_name=config.defense,
                num_clients=num_clients,
                malicious_fraction=config.malicious_fraction,
                num_rounds=num_rounds,
                batch_size=config.batch_size,
                lr=config.lr,
                device=device,
                alpha_channel=config.alpha_channel,
                seed=seed,
            )

            from scl.metrics.aggregated import MetricsTracker
            tracker = MetricsTracker()
            for t in range(1, num_rounds + 1):
                m = trainer.run_round(t, snr_db=snr_db)
                tracker.record(m)
                print(f"  Round {t:3d}/{num_rounds} | acc={m.test_accuracy:.3f} | "
                      f"loss={m.train_loss:.4f} | fidelity={m.semantic_fidelity:.3f}")

            # ── Lipschitz estimation (at final round) ──────────────────────
            Ls = Ll = Lg = 0.0
            bound_val = excess_val = 0.0
            try:
                # Collect z samples from one batch via first client
                first_loader = torch.utils.data.DataLoader(
                    trainer.train_datasets[0], batch_size=min(32, config.batch_size), shuffle=True
                )
                x_s, y_s = next(iter(first_loader))
                x_s, y_s = x_s.to(device), y_s.to(device)
                with torch.no_grad():
                    z_s = trainer.client_models[0](x_s)
                criterion = torch.nn.CrossEntropyLoss()
                Ls = estimate_Ls(trainer.server_model, z_s, n_pairs=lip_pairs[0])
                Ll = estimate_Ll(trainer.server_model, criterion, z_s, y_s, n_pairs=lip_pairs[1])
                Lg = estimate_Lg(trainer.server_model, criterion, z_s, y_s, n_pairs=lip_pairs[2])
                d_z = z_s[0].numel()
                bound_val = compute_theorem1_bound(Ls, Ll, d_z, config.alpha_channel, snr_db)
                excess_val = tracker.history[-1].excess_loss if tracker.history else 0.0
                tightness = bound_val / (excess_val + 1e-8)
            except Exception as e:
                tightness = 0.0

            # Update Lipschitz fields on last recorded metric
            if tracker.history:
                tracker.history[-1].Ls_estimate = Ls
                tracker.history[-1].Lg_estimate = Lg
                tracker.history[-1].bound_tightness = tightness

            # Save per-seed-snr tracker
            out_path = os.path.join(output_dir, f"seed{seed}_snr{snr_db}.json")
            save_tracker(tracker, out_path)

            for m in tracker.history:
                records.append({
                    "seed": seed,
                    "snr_db": snr_db,
                    **{k: v for k, v in vars(m).items()},
                    "Ls": Ls,
                    "Ll": Ll,
                    "Lg": Lg,
                    "bound": bound_val,
                })

    df = pd.DataFrame(records)
    summary_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n[EG1] Summary saved to {summary_path}")
    return df
