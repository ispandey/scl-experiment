"""SFL server: forward pass, loss computation, gradient broadcasting."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scl.metrics.information import ib_IZtildeY


class SFLServer:
    """
    Manages the server-side split federated learning operations.

    Responsibilities:
    1. Receive z̃ from clients and compute server-side forward pass.
    2. Compute task loss (cross-entropy).
    3. Optionally add IB regularisation (for SCL method).
    4. Backpropagate to produce ∂L/∂z̃ to send back to clients.
    5. Evaluate the full pipeline on the test set.
    """

    def __init__(
        self,
        server_model: nn.Module,
        criterion: nn.Module,
        device: str = "cpu",
        ib_beta: float = 0.0,   # IB weight β for SCL (0 = standard SFL)
        num_classes: int = 10,
    ):
        self.server_model = server_model.to(device)
        self.criterion = criterion
        self.device = device
        self.ib_beta = ib_beta
        self.num_classes = num_classes

    def forward_loss(
        self,
        z_tilde: torch.Tensor,
        y: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute task loss (and optional IB term).

        Args:
            z_tilde: Smash data from client (requires grad for backprop).
            y:       Labels.
            mu, logvar: VAE outputs for IB regularisation (optional).

        Returns:
            (loss, logits, smash_grad=∂L/∂z̃)
        """
        z_in = z_tilde.detach().requires_grad_(True)
        logits = self.server_model(z_in)
        loss = self.criterion(logits, y)

        # IB regularisation
        if self.ib_beta > 0 and mu is not None and logvar is not None:
            from scl.metrics.information import ib_IXZ
            kl = torch.tensor(ib_IXZ(mu, logvar))
            loss = loss + self.ib_beta * kl

        loss.backward()
        smash_grad = z_in.grad.detach()

        return loss, logits, smash_grad

    def collect_server_gradient(self, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        After calling forward_loss on a clean root dataset, return server-model
        gradient for FLTrust (concatenated, flattened).
        """
        grads = []
        for p in self.server_model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
        return torch.cat(grads) if grads else torch.zeros(1)

    def get_param_gradient(self) -> torch.Tensor:
        """Return current gradient of server model parameters as flat vector."""
        grads = []
        for p in self.server_model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().flatten())
        return torch.cat(grads) if grads else torch.zeros(1)

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        client_model: nn.Module,
        channel,
        snr_db: float,
    ) -> Dict:
        """Full pipeline evaluation: client_enc → channel → server_dec."""
        client_model.eval()
        self.server_model.eval()

        correct = total = 0
        total_loss = 0.0
        n_batches = 0

        for x, y in test_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            z = client_model(x)
            z_tilde, _ = channel.forward(z.detach(), snr_db)
            logits = self.server_model(z_tilde)
            loss = self.criterion(logits, y)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item()
            n_batches += 1

        acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        return {"test_accuracy": acc, "test_loss": avg_loss}
