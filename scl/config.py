"""SCL experiment configuration dataclasses."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EG1Config:
    """EG-1: Theorem 1 & 3 Calibration"""
    dataset: str = "cifar10"
    partition: str = "iid"
    arch: str = "resnet18"
    split_layer: int = 2
    channel: str = "rayleigh"
    snr_list: List[float] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    attack: str = "none"
    defense: str = "fedavg"
    num_rounds: int = 100
    num_seeds: int = 5
    num_clients: int = 50
    malicious_fraction: float = 0.0
    batch_size: int = 64
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    alpha_channel: float = 1.0


@dataclass
class EG2Config:
    """EG-2: Semantic Capacity Measurement"""
    dataset: str = "cifar10"
    partition: str = "iid"
    arch: str = "resnet18"
    split_layer: int = 2
    channel: str = "rayleigh"
    snr_list: List[float] = field(default_factory=lambda: [0, 5, 10, 15, 20, 25, 30])
    attack: str = "none"
    defenses: List[str] = field(default_factory=lambda: ["scl", "fedavg", "nonchannel"])
    num_rounds: int = 100
    num_seeds: int = 5
    num_clients: int = 50
    malicious_fraction: float = 0.0
    batch_size: int = 64
    lr: float = 0.1
    alpha_channel: float = 1.0


@dataclass
class EG3Config:
    """EG-3: Robustness Matrix"""
    dataset: str = "cifar10"
    partition: str = "iid"
    arch: str = "resnet18"
    split_layer: int = 2
    channel: str = "rayleigh"
    snr_list: List[float] = field(default_factory=lambda: [5, 15, 25])
    attacks: List[str] = field(
        default_factory=lambda: [
            "none", "weight_poison", "label_flip", "smash", "ipm", "minmax"
        ]
    )
    defenses: List[str] = field(
        default_factory=lambda: [
            "fedavg", "median", "krum", "flame", "fltrust", "dnc", "bcbsa", "bcbsa_nosem"
        ]
    )
    num_rounds: int = 100
    num_seeds: int = 5
    num_clients: int = 50
    malicious_fraction: float = 0.30
    batch_size: int = 64
    lr: float = 0.1
    alpha_channel: float = 1.0
    # EG-3B
    fraction_sweep: List[float] = field(default_factory=lambda: [0.10, 0.20, 0.30, 0.40, 0.50])


@dataclass
class EG4Config:
    """EG-4: Generalization"""
    datasets: List[str] = field(default_factory=lambda: ["cifar10", "cifar100", "tinyimagenet", "mnist"])
    partition_types: List[str] = field(default_factory=lambda: ["iid", "dir05", "dir01"])
    arch_list: List[str] = field(default_factory=lambda: ["resnet18", "mobilenetv2"])
    split_layer: int = 2
    channel: str = "rayleigh"
    snr_fixed: float = 15.0
    attack: str = "weight_poison"
    defenses: List[str] = field(default_factory=lambda: ["fedavg", "flame", "bcbsa", "fedprox"])
    num_rounds: int = 100
    num_seeds: int = 3
    num_clients: int = 50
    malicious_fraction: float = 0.30
    batch_size: int = 64
    lr: float = 0.1


@dataclass
class EG5Config:
    """EG-5: Ablation and Sensitivity"""
    dataset: str = "cifar10"
    partition: str = "iid"
    channel: str = "rayleigh"
    snr_list: List[float] = field(default_factory=lambda: [5, 15, 25])
    snr_fixed: float = 15.0
    num_rounds: int = 100
    num_seeds: int = 5
    num_clients: int = 50
    malicious_fraction: float = 0.30
    batch_size: int = 64
    lr: float = 0.1
    # EG-5A
    split_layers: List[int] = field(default_factory=lambda: [1, 2, 3])
    # EG-5C
    bcbsa_variants: List[str] = field(
        default_factory=lambda: [
            "bcbsa_full", "bcbsa_nofid", "bcbsa_nodist", "bcbsa_nosem",
            "bcbsa_notemp", "flame"
        ]
    )
    # EG-5D
    beta_sweep: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0])
    lambda_sweep: List[float] = field(default_factory=lambda: [1e-3, 5e-3, 1e-2, 5e-2])
    # EG-5E
    compression_sweep: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.85, 0.95])
