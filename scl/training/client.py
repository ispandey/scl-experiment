"""SFL client: local forward pass, channel transmission, gradient computation."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scl.channels.base import BaseChannel
from scl.metrics.communication import compute_bytes_transmitted, topk_compress


class SFLClient:
    """
    Manages the client-side split federated learning operations.

    Responsibilities:
    1. Run forward pass through the client-side encoder to produce smash data z.
    2. Pass z through the (simulated) channel to obtain z̃.
    3. Optionally apply compression to z before transmission.
    4. Receive ∂L/∂z̃ from server and backpropagate through the client model.
    """

    def __init__(
        self,
        client_id: int,
        client_model: nn.Module,
        dataset: Dataset,
        channel: BaseChannel,
        attack=None,         # attack object or None
        device: str = "cpu",
        batch_size: int = 64,
        compression_ratio: float = 1.0,
    ):
        self.client_id = client_id
        self.client_model = client_model.to(device)
        self.dataset = dataset
        self.channel = channel
        self.attack = attack
        self.device = device
        self.batch_size = batch_size
        self.compression_ratio = compression_ratio

        self._loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )
        self._iter = None

    def _next_batch(self):
        if self._iter is None:
            self._iter = iter(self._loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._loader)
            return next(self._iter)

    def forward(
        self,
        snr_db: float,
        apply_label_flip: bool = False,
        num_classes: int = 10,
    ) -> Dict:
        """
        Run client forward pass.

        Returns a dict with keys:
            x, y, z_clean, z_tilde, channel_info, bytes_tx
        """
        x, y = self._next_batch()
        x = x.to(self.device)
        y = y.to(self.device)

        # Label flipping (A2) — modify labels before use
        if apply_label_flip and self.attack is not None:
            from scl.attacks.label_flip import LabelFlipAttack
            if isinstance(self.attack, LabelFlipAttack):
                y = self.attack.attack_labels(y)

        # Client-side forward
        self.client_model.train()
        z_clean = self.client_model(x)

        # Optional compression
        if self.compression_ratio < 1.0:
            z_clean = topk_compress(z_clean, self.compression_ratio)

        # SMASH attack (A3) — perturb z before sending
        z_send = z_clean
        if self.attack is not None:
            from scl.attacks.smash import SMASHAttack
            if isinstance(self.attack, SMASHAttack):
                z_send = self.attack.attack_smash(z_clean.detach())

        # Channel
        z_tilde, channel_info = self.channel.forward(z_send.detach(), snr_db)

        # Byte count
        bytes_tx = compute_bytes_transmitted(z_clean, self.compression_ratio)

        return {
            "x": x,
            "y": y,
            "z_clean": z_clean,     # client encoder output (requires grad)
            "z_tilde": z_tilde,     # noisy smash data sent to server
            "z_send": z_send,       # after optional attack (before channel)
            "channel_info": channel_info,
            "bytes_tx": bytes_tx,
        }

    def backward(
        self,
        z_clean: torch.Tensor,
        smash_grad: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict:
        """
        Backprop through client model.

        Args:
            z_clean:    The z tensor from forward() (has grad_fn).
            smash_grad: ∂L/∂z̃ from the server.
            optimizer:  Client-side optimizer.

        Returns:
            dict with 'client_grad_norm'.
        """
        optimizer.zero_grad()
        z_clean.backward(smash_grad)
        torch.nn.utils.clip_grad_norm_(self.client_model.parameters(), max_norm=10.0)
        optimizer.step()

        total_norm = 0.0
        for p in self.client_model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return {"client_grad_norm": total_norm ** 0.5}
