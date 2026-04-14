"""Shared utilities for experiment runners."""
from __future__ import annotations

import copy
import os
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from scl.attacks import get_attack
from scl.channels import get_channel
from scl.data.datasets import get_dataset, num_classes
from scl.data.partition import partition_dataset
from scl.defenses import get_defense
from scl.defenses.bcbsa import BCBSA
from scl.models.resnet import build_resnet18_split
from scl.models.mobilenet import build_mobilenetv2_split
from scl.training.federated import FederatedTrainer


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _is_cuda(device: str) -> bool:
    return device.startswith("cuda")


def _pin_memory(device: str) -> bool:
    """Return True when DataLoaders should pin host memory for fast H→D transfer."""
    return _is_cuda(device)


def _num_workers() -> int:
    """Number of DataLoader worker processes (bounded so we don't over-subscribe)."""
    cpu_count = os.cpu_count() or 1
    return min(4, cpu_count)


def setup_cuda(device: str) -> None:
    """
    Apply one-time CUDA optimisations when a GPU device is selected.

    - cudnn.benchmark=True:  auto-tunes convolution kernels for fixed input sizes
                             (CIFAR/TinyImageNet batches are constant-size).
    - cudnn.deterministic:   left at its default (False) for maximum throughput;
                             reproducibility is controlled via set_seed().
    """
    if not _is_cuda(device):
        return
    torch.backends.cudnn.benchmark = True


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_models(
    arch: str,
    split_layer: int,
    num_cls: int,
    device: str,
) -> Tuple[nn.Module, nn.Module]:
    if arch == "resnet18":
        client_m, server_m = build_resnet18_split(split_layer, num_cls)
    elif arch == "mobilenetv2":
        client_m, server_m = build_mobilenetv2_split(num_cls=num_cls)
    else:
        raise ValueError(f"Unknown arch '{arch}'")
    return client_m.to(device), server_m.to(device)


def build_trainer(
    *,
    arch: str,
    split_layer: int,
    dataset_name: str,
    partition: str,
    channel_name: str,
    attack_name: str,
    defense_name: str,
    num_clients: int,
    malicious_fraction: float,
    num_rounds: int,
    batch_size: int,
    lr: float,
    device: str,
    alpha_channel: float = 1.0,
    compression_ratio: float = 1.0,
    seed: int = 0,
) -> Tuple[FederatedTrainer, DataLoader]:
    """Build a fully-configured FederatedTrainer for one experiment cell."""
    setup_cuda(device)
    set_seed(seed)

    nc = num_classes(dataset_name)
    train_ds = get_dataset(dataset_name, train=True)
    test_ds = get_dataset(dataset_name, train=False)
    pin = _pin_memory(device)
    nw = _num_workers()
    test_loader = DataLoader(
        test_ds, batch_size=128, shuffle=False,
        num_workers=nw, pin_memory=pin,
    )

    client_datasets = partition_dataset(
        train_ds, num_clients, partition, num_classes=nc, seed=seed
    )

    n_malicious = int(num_clients * malicious_fraction)
    malicious_ids = list(range(num_clients - n_malicious, num_clients))
    honest_ids = list(range(num_clients - n_malicious))

    channel = get_channel(channel_name, alpha=alpha_channel)
    attack = get_attack(attack_name)
    defense = get_defense(defense_name)

    # Build one client model per client (shared-init, independent optimisers)
    client_m_proto, server_m = build_models(arch, split_layer, nc, device)
    client_models = [copy.deepcopy(client_m_proto) for _ in range(num_clients)]

    # Small root dataset for FLTrust (50 samples from test set)
    root_ds = Subset(test_ds, list(range(min(50, len(test_ds)))))

    trainer = FederatedTrainer(
        client_models=client_models,
        server_model=server_m,
        train_datasets=client_datasets,
        test_loader=test_loader,
        channel=channel,
        attack=attack,
        defense=defense,
        honest_ids=honest_ids,
        malicious_ids=malicious_ids,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        batch_size=batch_size,
        lr=lr,
        warmup_rounds=min(10, num_rounds // 10),
        total_rounds=num_rounds,
        num_classes=nc,
        compression_ratio=compression_ratio,
        alpha_channel=alpha_channel,
        root_dataset=root_ds,
    )
    return trainer, test_loader


def save_tracker(tracker, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    tracker.save(path)
