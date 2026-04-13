"""Tests for attack implementations."""
import torch
import pytest

from scl.attacks import get_attack
from scl.attacks.weight_poison import WeightPoisonAttack
from scl.attacks.label_flip import LabelFlipAttack
from scl.attacks.smash import SMASHAttack
from scl.attacks.ipm import IPMAttack
from scl.attacks.minmax import MinMaxAttack


def make_grad(seed=0):
    torch.manual_seed(seed)
    return torch.randn(100)


def test_weight_poison():
    atk = WeightPoisonAttack(scale=5.0)
    g = make_grad()
    g_adv = atk.attack(g)
    assert g_adv.shape == g.shape
    # Adversarial gradient should differ from honest
    assert not torch.allclose(g_adv, g)


def test_label_flip():
    atk = LabelFlipAttack(num_classes=10)
    y = torch.tensor([0, 1, 9, 5])
    y_adv = atk.attack_labels(y)
    assert y_adv.tolist() == [1, 2, 0, 6]


def test_smash_attack():
    atk = SMASHAttack(epsilon=0.3)
    z = torch.randn(4, 32)
    z_adv = atk.attack_smash(z)
    assert z_adv.shape == z.shape
    assert not torch.allclose(z_adv, z)


def test_ipm():
    atk = IPMAttack(gamma=10.0)
    grads = [torch.ones(50), torch.ones(50) * 2]
    g_adv = atk.attack(grads)
    assert g_adv.shape == (50,)
    # Should be negative (reversed direction)
    assert g_adv.mean().item() < 0


def test_minmax():
    atk = MinMaxAttack(lam=1.0, lr=0.01, steps=2)
    g = make_grad()
    g_adv = atk.attack(g, global_grad=g.clone())
    assert g_adv.shape == g.shape


def test_factory_none():
    atk = get_attack("none")
    assert atk is None


def test_factory_all():
    for name in ["weight_poison", "label_flip", "smash", "ipm", "minmax"]:
        atk = get_attack(name)
        assert atk is not None
