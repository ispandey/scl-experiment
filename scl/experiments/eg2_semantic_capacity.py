"""EG-2: Semantic Capacity Measurement."""
from __future__ import annotations

import os

import pandas as pd
import torch

from scl.config import EG2Config
from scl.metrics.information import (
    ib_IZtildeY,
    channel_semantic_loss,
    semantic_efficiency,
    estimate_sigma_z,
    capacity_geometry_bound,
)
from scl.experiments._utils import build_trainer, save_tracker


def run_eg2(
    config: EG2Config,
    output_dir: str,
    device: str = "cpu",
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    EG-2: Measure I(Z;Y), I(Z̃;Y), Δ_ch, η_s for SCL vs FedAvg across SNR.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []

    snr_list = config.snr_list[:2] if dry_run else config.snr_list
    num_seeds = 1 if dry_run else config.num_seeds
    num_rounds = 2 if dry_run else config.num_rounds

    # Methods: 'scl' uses IB-augmented training (beta>0); 'fedavg' standard
    method_map = {
        "scl": ("fedavg", 0.1),         # defense name, ib_beta
        "fedavg": ("fedavg", 0.0),
        "nonchannel": ("fedavg", 0.0),  # noiseless channel
    }

    for seed in range(num_seeds):
        for method in config.defenses:
            defense_name, ib_beta = method_map.get(method, ("fedavg", 0.0))
            ch_name = "noiseless" if method == "nonchannel" else config.channel

            for snr_db in snr_list:
                print(f"\n[EG2] seed={seed} method={method} snr={snr_db} dB")
                trainer, test_loader = build_trainer(
                    arch=config.arch,
                    split_layer=config.split_layer,
                    dataset_name=config.dataset,
                    partition=config.partition,
                    channel_name=ch_name,
                    attack_name=config.attack,
                    defense_name=defense_name,
                    num_clients=config.num_clients,
                    malicious_fraction=config.malicious_fraction,
                    num_rounds=num_rounds,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    device=device,
                    alpha_channel=config.alpha_channel,
                    seed=seed,
                )

                tracker = trainer.train(snr_db=snr_db, total_rounds=num_rounds)

                # ── Compute MI metrics after training ──────────────────────
                IZY = IZtildeY_val = delta_ch = eta_s = geom_bound = 0.0
                try:
                    x_p, y_p = next(iter(test_loader))
                    x_p, y_p = x_p.to(device), y_p.to(device)
                    from scl.channels import get_channel
                    ch = get_channel(ch_name, alpha=config.alpha_channel)
                    with torch.no_grad():
                        z = trainer.client_models[0](x_p)
                        z_tilde, _ = ch.forward(z, snr_db)
                        logits_clean = trainer.server_model(z)
                        logits_noisy = trainer.server_model(z_tilde)
                    from scl.data.datasets import num_classes
                    nc = num_classes(config.dataset)
                    IZY = ib_IZtildeY(logits_clean, y_p, nc)
                    IZtildeY_val = ib_IZtildeY(logits_noisy, y_p, nc)
                    delta_ch = max(0.0, IZY - IZtildeY_val)
                    eta_s = semantic_efficiency(IZtildeY_val, snr_db)

                    # Geometry bound
                    sigma_z = estimate_sigma_z(z)
                    geom_bound = capacity_geometry_bound(sigma_z, snr_db, config.alpha_channel)
                except Exception:
                    pass

                out_path = os.path.join(output_dir, f"seed{seed}_{method}_snr{snr_db}.json")
                save_tracker(tracker, out_path)

                for m in tracker.history:
                    records.append({
                        "seed": seed,
                        "method": method,
                        "snr_db": snr_db,
                        **{k: v for k, v in vars(m).items()},
                        "IZY": IZY,
                        "IZtildeY": IZtildeY_val,
                        "delta_ch": delta_ch,
                        "eta_s": eta_s,
                        "geom_bound": geom_bound,
                    })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"\n[EG2] Summary saved to {output_dir}/summary.csv")
    return df
