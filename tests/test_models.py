"""Tests for model implementations (split ResNet-18 and MobileNetV2)."""
import torch
import pytest

from scl.models.resnet import ResNet18Client, ResNet18Server, build_resnet18_split
from scl.models.mobilenet import MobileNetV2Client, MobileNetV2Server, build_mobilenetv2_split


@pytest.mark.parametrize("split_layer", [1, 2, 3])
def test_resnet18_shapes(split_layer):
    """Verify smash data shape and server output shape for each split point."""
    client, server = build_resnet18_split(split_layer, num_classes=10)
    client.eval()
    server.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        z = client(x)
        logits = server(z)

    expected_smash_dims = {1: (2, 64, 8, 8), 2: (2, 128, 4, 4), 3: (2, 256, 2, 2)}
    assert z.shape == expected_smash_dims[split_layer], f"Got {z.shape}"
    assert logits.shape == (2, 10)


def test_resnet18_smash_dim_property():
    for sl in [1, 2, 3]:
        client = ResNet18Client(sl)
        expected = {1: 64 * 8 * 8, 2: 128 * 4 * 4, 3: 256 * 2 * 2}
        assert client.smash_dim == expected[sl]


def test_mobilenetv2_forward():
    client, server = build_mobilenetv2_split(num_classes=10)
    client.eval()
    server.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        z = client(x)
        logits = server(z)

    assert logits.shape == (2, 10)


def test_invalid_split_layer():
    with pytest.raises(ValueError):
        ResNet18Client(split_layer=0)
    with pytest.raises(ValueError):
        ResNet18Server(split_layer=4)
