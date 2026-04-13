"""
Channel factory.

Usage:
    from scl.channels import get_channel
    ch = get_channel('rayleigh')
    z_tilde, info = ch(z, snr_db=15.0)
"""
from .base import BaseChannel
from .rayleigh import RayleighSemanticChannel
from .awgn import AWGNSemanticChannel
from .rician import RicianSemanticChannel
from .digital import DigitalBPSKChannel
from .noiseless import NoiselessChannel

_REGISTRY = {
    "rayleigh": RayleighSemanticChannel,
    "awgn": AWGNSemanticChannel,
    "rician": RicianSemanticChannel,
    "digital": DigitalBPSKChannel,
    "bpsk": DigitalBPSKChannel,
    "noiseless": NoiselessChannel,
    "none": NoiselessChannel,
}


def get_channel(name: str, **kwargs) -> BaseChannel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown channel '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
