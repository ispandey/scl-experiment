"""
Defense factory.

Usage:
    from scl.defenses import get_defense
    defense = get_defense('bcbsa')
"""
from .fedavg import FedAvg
from .median import CoordMedian
from .krum import Krum
from .flame import FLAME
from .fltrust import FLTrust
from .dnc import DnC
from .bcbsa import BCBSA, bcbsa_full, bcbsa_nofid, bcbsa_nodist, bcbsa_nosem, bcbsa_notemp

_REGISTRY = {
    "fedavg": FedAvg,
    "median": CoordMedian,
    "krum": Krum,
    "flame": FLAME,
    "fltrust": FLTrust,
    "dnc": DnC,
    "bcbsa": bcbsa_full,
    "bcbsa_full": bcbsa_full,
    "bcbsa_nofid": bcbsa_nofid,
    "bcbsa_nodist": bcbsa_nodist,
    "bcbsa_nosem": bcbsa_nosem,
    "bcbsa_notemp": bcbsa_notemp,
}


def get_defense(name: str, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown defense '{name}'. Choose from {list(_REGISTRY)}")
    factory = _REGISTRY[name]
    # Factory functions (bcbsa_*) take no kwargs; classes accept kwargs
    try:
        return factory(**kwargs) if kwargs else factory()
    except TypeError:
        return factory()
