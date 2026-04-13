"""Tests for channel models."""
import math
import torch
import pytest

from scl.channels import get_channel
from scl.channels.rayleigh import RayleighSemanticChannel
from scl.channels.awgn import AWGNSemanticChannel
from scl.channels.rician import RicianSemanticChannel
from scl.channels.digital import DigitalBPSKChannel
from scl.channels.noiseless import NoiselessChannel


@pytest.fixture
def z():
    torch.manual_seed(0)
    return torch.randn(4, 16, 4, 4)  # batch=4, smash-like shape


def test_rayleigh_shape(z):
    ch = RayleighSemanticChannel()
    z_tilde, info = ch(z, snr_db=15.0)
    assert z_tilde.shape == z.shape
    assert "snr_effective_db" in info


def test_awgn_shape(z):
    ch = AWGNSemanticChannel()
    z_tilde, info = ch(z, snr_db=10.0)
    assert z_tilde.shape == z.shape


def test_rician_shape(z):
    ch = RicianSemanticChannel()
    z_tilde, info = ch(z, snr_db=10.0)
    assert z_tilde.shape == z.shape
    assert "kappa_rice" in info


def test_digital_shape(z):
    ch = DigitalBPSKChannel()
    z_tilde, info = ch(z, snr_db=10.0)
    assert z_tilde.shape == z.shape
    assert "ber" in info
    assert 0.0 <= info["ber"] <= 0.5


def test_noiseless(z):
    ch = NoiselessChannel()
    z_tilde, info = ch(z, snr_db=15.0)
    assert torch.allclose(z_tilde, z)
    assert info["sigma2_eps"] == 0.0


def test_factory():
    for name in ["rayleigh", "awgn", "rician", "bpsk", "noiseless", "none"]:
        ch = get_channel(name)
        z = torch.randn(2, 8)
        z_tilde, _ = ch(z, snr_db=15.0)
        assert z_tilde.shape == z.shape


def test_high_snr_low_noise(z):
    """At very high SNR, AWGN noise should be negligible."""
    ch = AWGNSemanticChannel()
    z_tilde, _ = ch(z, snr_db=60.0)
    mse = (z_tilde - z).pow(2).mean().item()
    assert mse < 1e-3
