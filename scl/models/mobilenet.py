"""Split MobileNetV2 at block 4→5 for edge-realistic experiments."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as tvm


class MobileNetV2Client(nn.Module):
    """
    Client-side of MobileNetV2, using the first `num_blocks_client` inverted
    residual blocks plus the initial conv stem.

    Default: 5 blocks → smash dim ≈ 32 × 14 × 14 = 6272 (for 224×224 input).
    For CIFAR (32×32): output ~ 32 × 4 × 4 = 512.
    """

    def __init__(self, num_blocks_client: int = 5):
        super().__init__()
        base = tvm.mobilenet_v2(weights=None)
        features = base.features
        # features[0] is ConvBNActivation stem; [1:] are InvertedResidual blocks
        # We take features[0 .. num_blocks_client] inclusive
        self.features = nn.Sequential(*list(features[: num_blocks_client + 1]))
        self.num_blocks_client = num_blocks_client

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class MobileNetV2Server(nn.Module):
    """
    Server-side of MobileNetV2, starting from block `num_blocks_client + 1`.
    """

    def __init__(self, num_blocks_client: int = 5, num_classes: int = 10):
        super().__init__()
        base = tvm.mobilenet_v2(weights=None)
        features = base.features
        self.features = nn.Sequential(*list(features[num_blocks_client + 1 :]))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        last_channel = base.last_channel  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.features(z)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_mobilenetv2_split(
    num_blocks_client: int = 5, num_classes: int = 10
) -> Tuple[MobileNetV2Client, MobileNetV2Server]:
    return MobileNetV2Client(num_blocks_client), MobileNetV2Server(num_blocks_client, num_classes)
