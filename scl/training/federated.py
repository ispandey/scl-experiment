"""Main federated training loop."""
from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from scl.channels.base import BaseChannel
from scl.defenses.bcbsa import BCBSA
from scl.defenses.fltrust import FLTrust
from scl.metrics.aggregated import MetricsTracker, RoundMetrics
from scl.metrics.communication import compute_bytes_transmitted
from scl.metrics.information import ib_IZtildeY, channel_semantic_loss, semantic_efficiency
from scl.metrics.robustness import (
    compute_acceptance_ratio,
    compute_false_positive_rate,
    compute_gradient_bias_norm,
)
from scl.training.scheduler import WarmupCosineScheduler


class FederatedTrainer:
    """
    Orchestrates one federated round:
      1. Broadcast global server model weights.
      2. Each client: local forward → channel → smash data.
      3. Server: forward/loss/backward → smash grad.
      4. Client: backward → client gradient (server model params).
      5. Apply attacks on malicious clients.
      6. Defense aggregates server-model gradients.
      7. Update server model; each client updates its own model.
      8. Collect all metrics.
    """

    def __init__(
        self,
        client_models: List[nn.Module],
        server_model: nn.Module,
        train_datasets: List[Subset],
        test_loader: DataLoader,
        channel: BaseChannel,
        attack,
        defense,
        honest_ids: List[int],
        malicious_ids: List[int],
        criterion: nn.Module,
        device: str = "cpu",
        batch_size: int = 64,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        warmup_rounds: int = 10,
        total_rounds: int = 100,
        num_classes: int = 10,
        compression_ratio: float = 1.0,
        alpha_channel: float = 1.0,
        # Optional small root dataset for FLTrust
        root_dataset: Optional[Dataset] = None,
    ):
        self.client_models = [m.to(device) for m in client_models]
        self.server_model = server_model.to(device)
        self.train_datasets = train_datasets
        self.test_loader = test_loader
        self.channel = channel
        self.attack = attack
        self.defense = defense
        self.honest_ids = set(honest_ids)
        self.malicious_ids = set(malicious_ids)
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.compression_ratio = compression_ratio
        self.alpha_channel = alpha_channel

        # Per-client optimizers
        self.client_optimizers = [
            torch.optim.SGD(
                m.parameters(), lr=lr, momentum=momentum,
                weight_decay=weight_decay, nesterov=True
            )
            for m in self.client_models
        ]
        # Server optimizer
        self.server_optimizer = torch.optim.SGD(
            server_model.parameters(), lr=lr, momentum=momentum,
            weight_decay=weight_decay, nesterov=True
        )
        # Scheduler applied to server optimizer (clients follow)
        self.scheduler = WarmupCosineScheduler(
            self.server_optimizer,
            warmup_rounds=warmup_rounds,
            total_rounds=total_rounds,
            start_lr=lr * 0.1,
            peak_lr=lr,
        )

        # Root loader for FLTrust
        self.root_loader = None
        if root_dataset is not None:
            self.root_loader = DataLoader(root_dataset, batch_size=32, shuffle=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_attack_type(self) -> str:
        if self.attack is None:
            return "none"
        return type(self.attack).__name__.lower()

    def _is_bcbsa(self) -> bool:
        return isinstance(self.defense, BCBSA)

    def _is_fltrust(self) -> bool:
        return isinstance(self.defense, FLTrust)

    def _client_loader(self, client_id: int) -> DataLoader:
        return DataLoader(
            self.train_datasets[client_id],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    # ------------------------------------------------------------------
    # One round
    # ------------------------------------------------------------------

    def run_round(self, round_t: int, snr_db: float) -> RoundMetrics:
        t_start = time.time()
        metrics = RoundMetrics(round_t=round_t)

        # Update LR
        self.scheduler.step(round_t)

        # Broadcast server model to all clients (copies of state dict)
        global_state = copy.deepcopy(self.server_model.state_dict())

        all_client_ids = list(range(len(self.client_models)))
        n_clients = len(all_client_ids)

        # Collect per-client quantities
        server_grads: List[torch.Tensor] = []
        z_cleans: List[torch.Tensor] = []
        z_tildes: List[torch.Tensor] = []
        train_losses: List[float] = []
        fidelities: List[float] = []
        distortions: List[float] = []
        bytes_txs: List[int] = []
        ber_vals: List[float] = []
        snr_effs: List[float] = []

        # Optional: excess loss measurement
        excess_losses: List[float] = []

        for cid in all_client_ids:
            loader = self._client_loader(cid)
            try:
                x, y = next(iter(loader))
            except StopIteration:
                continue

            x, y = x.to(self.device), y.to(self.device)

            # Label flip attack (A2)
            if cid in self.malicious_ids and self.attack is not None:
                from scl.attacks.label_flip import LabelFlipAttack
                if isinstance(self.attack, LabelFlipAttack):
                    y = self.attack.attack_labels(y)

            client_model = self.client_models[cid]
            client_model.train()

            # ---- Client encoder forward ----
            z_clean = client_model(x)

            # ---- SMASH attack (A3) ----
            z_send = z_clean.detach()
            if cid in self.malicious_ids and self.attack is not None:
                from scl.attacks.smash import SMASHAttack
                if isinstance(self.attack, SMASHAttack):
                    z_send = self.attack.attack_smash(z_send)

            # ---- Channel ----
            z_tilde, ch_info = self.channel.forward(z_send, snr_db)
            snr_effs.append(ch_info.get("snr_effective_db", snr_db))
            ber_vals.append(ch_info.get("ber", 0.0))

            # Fidelity & distortion
            fid = 1.0 - (z_tilde - z_clean.detach()).pow(2).sum().item() / (
                z_clean.detach().pow(2).sum().item() + 1e-8
            )
            dist = (z_tilde - z_clean.detach()).pow(2).mean().item()
            fidelities.append(fid)
            distortions.append(dist)
            bytes_txs.append(compute_bytes_transmitted(z_clean, self.compression_ratio))

            # Excess loss proxy
            with torch.no_grad():
                loss_clean = self.criterion(
                    self.server_model(z_clean.detach()), y
                ).item()
                loss_noisy = self.criterion(
                    self.server_model(z_tilde.detach()), y
                ).item()
            excess_losses.append(max(0.0, loss_noisy - loss_clean))

            # ---- Server forward + backward (smash grad) ----
            self.server_optimizer.zero_grad()
            z_in = z_tilde.detach().requires_grad_(True)
            logits = self.server_model(z_in)
            loss = self.criterion(logits, y)
            loss.backward()
            smash_grad = z_in.grad.detach()
            train_losses.append(loss.item())

            # Collect server gradient (flat)
            srv_grad = torch.cat([
                p.grad.detach().flatten()
                for p in self.server_model.parameters()
                if p.grad is not None
            ])

            # ---- Client backward ----
            self.client_optimizers[cid].zero_grad()
            z_clean.backward(smash_grad)
            nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=10.0)
            self.client_optimizers[cid].step()

            # ---- Weight poisoning (A1) / IPM (A4) / MinMax (A5) ----
            if cid in self.malicious_ids and self.attack is not None:
                from scl.attacks.weight_poison import WeightPoisonAttack
                from scl.attacks.ipm import IPMAttack
                from scl.attacks.minmax import MinMaxAttack

                if isinstance(self.attack, WeightPoisonAttack):
                    srv_grad = self.attack.attack(srv_grad)
                elif isinstance(self.attack, IPMAttack):
                    # IPM needs honest gradients; use current grad as proxy
                    srv_grad = self.attack.attack([srv_grad])
                elif isinstance(self.attack, MinMaxAttack):
                    srv_grad = self.attack.attack(srv_grad)

            server_grads.append(srv_grad)
            z_cleans.append(z_clean.detach())
            z_tildes.append(z_tilde.detach())

        if not server_grads:
            return metrics

        # ---- FLTrust: compute server gradient on root data ----
        server_grad_root = None
        if self._is_fltrust() and self.root_loader is not None:
            self.server_optimizer.zero_grad()
            try:
                xr, yr = next(iter(self.root_loader))
                xr, yr = xr.to(self.device), yr.to(self.device)
                # Use first client model as proxy encoder for root data
                with torch.no_grad():
                    zr = self.client_models[0](xr)
                    zr_tilde, _ = self.channel.forward(zr, snr_db)
                zr_in = zr_tilde.requires_grad_(False)
                lr_logits = self.server_model(zr_in)
                lr_loss = self.criterion(lr_logits, yr)
                lr_loss.backward()
                server_grad_root = torch.cat([
                    p.grad.detach().flatten()
                    for p in self.server_model.parameters()
                    if p.grad is not None
                ])
            except Exception:
                server_grad_root = None
            self.server_optimizer.zero_grad()

        # ---- Defense aggregation ----
        acceptance_ratio = 1.0
        trust_scores = {}

        if self._is_bcbsa():
            agg_grad, acceptance_ratio, trust_scores = self.defense.aggregate(
                server_grads,
                all_client_ids,
                z_cleans,
                z_tildes,
            )
        elif self._is_fltrust():
            agg_grad = self.defense.aggregate(server_grads, server_grad_root)
        elif hasattr(self.defense, "aggregate"):
            try:
                agg_grad = self.defense.aggregate(server_grads)
            except TypeError:
                agg_grad = torch.stack(server_grads).mean(0)
        else:
            agg_grad = torch.stack(server_grads).mean(0)

        # ---- Apply aggregated gradient to server model ----
        self.server_optimizer.zero_grad()
        offset = 0
        for p in self.server_model.parameters():
            n = p.numel()
            p.grad = agg_grad[offset: offset + n].view(p.shape).clone()
            offset += n
        nn.utils.clip_grad_norm_(self.server_model.parameters(), max_norm=10.0)
        self.server_optimizer.step()

        # Broadcast updated server weights (already updated in-place above)

        # ---- Evaluate on test set ----
        self.server_model.eval()
        for cm in self.client_models:
            cm.eval()

        correct = total = 0
        with torch.no_grad():
            for x_t, y_t in self.test_loader:
                x_t, y_t = x_t.to(self.device), y_t.to(self.device)
                z_t = self.client_models[0](x_t)
                z_t_tilde, _ = self.channel.forward(z_t, snr_db)
                logits_t = self.server_model(z_t_tilde)
                preds = logits_t.argmax(dim=1)
                correct += (preds == y_t).sum().item()
                total += y_t.size(0)

        test_acc = correct / total if total > 0 else 0.0

        # ---- Acceptance / FP rate ----
        if trust_scores:
            accepted_ids = [
                cid for cid, t in trust_scores.items() if t > self.defense.tau
            ]
        else:
            accepted_ids = all_client_ids

        fpr = compute_false_positive_rate(accepted_ids, list(self.honest_ids))

        # ---- IZtildeY proxy ----
        if z_tildes and len(z_tildes) > 0:
            try:
                with torch.no_grad():
                    x_probe, y_probe = next(iter(self.test_loader))
                    x_probe, y_probe = x_probe.to(self.device), y_probe.to(self.device)
                    z_probe = self.client_models[0](x_probe)
                    zt_probe, _ = self.channel.forward(z_probe, snr_db)
                    logits_probe = self.server_model(zt_probe)
                IZtildeY = ib_IZtildeY(logits_probe, y_probe, self.num_classes)
                eta_s = semantic_efficiency(IZtildeY, snr_db)
            except Exception:
                IZtildeY = 0.0
                eta_s = 0.0
        else:
            IZtildeY = 0.0
            eta_s = 0.0

        # ---- Populate metrics ----
        metrics.test_accuracy = test_acc
        metrics.train_loss = sum(train_losses) / len(train_losses) if train_losses else 0.0
        metrics.excess_loss = sum(excess_losses) / len(excess_losses) if excess_losses else 0.0
        metrics.semantic_fidelity = sum(fidelities) / len(fidelities) if fidelities else 0.0
        metrics.distortion_Dk = sum(distortions) / len(distortions) if distortions else 0.0
        metrics.snr_effective_db = sum(snr_effs) / len(snr_effs) if snr_effs else snr_db
        metrics.ber = sum(ber_vals) / len(ber_vals) if ber_vals else 0.0
        metrics.IZtildeY_proxy = IZtildeY
        metrics.semantic_efficiency = eta_s
        metrics.acceptance_ratio = acceptance_ratio
        metrics.false_positive_rate = fpr
        metrics.bytes_transmitted = int(sum(bytes_txs) / len(bytes_txs)) if bytes_txs else 0
        metrics.compression_ratio = self.compression_ratio
        metrics.wall_clock_sec = time.time() - t_start

        return metrics

    def train(self, snr_db: float, total_rounds: int = 100) -> MetricsTracker:
        """Run the full training loop."""
        tracker = MetricsTracker()
        for t in range(1, total_rounds + 1):
            m = self.run_round(t, snr_db)
            tracker.record(m)
        return tracker
