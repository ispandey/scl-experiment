"""Split ResNet-18 with three configurable split points."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Helper: build ResNet-18 body
# ---------------------------------------------------------------------------

def _resnet18_body():
    """Return the torchvision ResNet-18 without the final avgpool/fc."""
    m = tvm.resnet18(weights=None)
    return m


# ---------------------------------------------------------------------------
# Client encoder
# ---------------------------------------------------------------------------

class ResNet18Client(nn.Module):
    """
    Client-side of ResNet-18, split after a chosen layer.

    split_layer:
        1 -> outputs after layer1  (64  × 32×32 for 32×32 input)
        2 -> outputs after layer2  (128 × 16×16 for 32×32 input)  [PRIMARY]
        3 -> outputs after layer3  (256 × 8×8  for 32×32 input)
    """

    def __init__(self, split_layer: int = 2):
        super().__init__()
        if split_layer not in (1, 2, 3):
            raise ValueError("split_layer must be 1, 2, or 3")
        self.split_layer = split_layer
        base = _resnet18_body()

        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )
        self.layer1 = base.layer1
        if split_layer >= 2:
            self.layer2 = base.layer2
        if split_layer >= 3:
            self.layer3 = base.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        if self.split_layer >= 2:
            x = self.layer2(x)
        if self.split_layer >= 3:
            x = self.layer3(x)
        return x

    @property
    def smash_dim(self) -> int:
        """Number of elements in the smash data tensor (for CIFAR 32×32 input)."""
        dims = {1: 64 * 8 * 8, 2: 128 * 4 * 4, 3: 256 * 2 * 2}
        return dims[self.split_layer]


# ---------------------------------------------------------------------------
# Server decoder
# ---------------------------------------------------------------------------

class ResNet18Server(nn.Module):
    """
    Server-side of ResNet-18, continuing from split_layer.
    """

    def __init__(self, split_layer: int = 2, num_classes: int = 10):
        super().__init__()
        if split_layer not in (1, 2, 3):
            raise ValueError("split_layer must be 1, 2, or 3")
        self.split_layer = split_layer
        base = _resnet18_body()

        layers = []
        if split_layer <= 1:
            layers.append(base.layer2)
        if split_layer <= 2:
            layers.append(base.layer3)
        if split_layer <= 3:
            layers.append(base.layer4)
        self.layers = nn.Sequential(*layers)
        self.avgpool = base.avgpool

        in_features = 512  # ResNet-18 layer4 always outputs 512 channels
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.layers(z)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_resnet18_split(
    split_layer: int = 2,
    num_classes: int = 10,
) -> Tuple[ResNet18Client, ResNet18Server]:
    """Return (client_model, server_model) for a given split layer."""
    return ResNet18Client(split_layer), ResNet18Server(split_layer, num_classes)
