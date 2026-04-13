"""
Attack factory.

Usage:
    from scl.attacks import get_attack
    atk = get_attack('weight_poison')
    g_adv = atk.attack(gradient)
"""
from .weight_poison import WeightPoisonAttack
from .label_flip import LabelFlipAttack
from .smash import SMASHAttack
from .ipm import IPMAttack
from .minmax import MinMaxAttack

_REGISTRY = {
    "none": None,
    "weight_poison": WeightPoisonAttack,
    "label_flip": LabelFlipAttack,
    "smash": SMASHAttack,
    "ipm": IPMAttack,
    "minmax": MinMaxAttack,
}


def get_attack(name: str, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown attack '{name}'. Choose from {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    return cls(**kwargs) if cls is not None else None
