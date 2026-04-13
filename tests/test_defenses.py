"""Tests for defense/aggregation implementations."""
import torch
import pytest

from scl.defenses import get_defense
from scl.defenses.fedavg import FedAvg
from scl.defenses.median import CoordMedian
from scl.defenses.krum import Krum
from scl.defenses.flame import FLAME
from scl.defenses.fltrust import FLTrust
from scl.defenses.dnc import DnC
from scl.defenses.bcbsa import BCBSA, bcbsa_full, bcbsa_nosem


def honest_grads(n=5, d=50, seed=0):
    torch.manual_seed(seed)
    return [torch.ones(d) + 0.01 * torch.randn(d) for _ in range(n)]


def poison_grads(n=2, d=50, seed=42):
    torch.manual_seed(seed)
    return [-5.0 * torch.ones(d) + torch.randn(d) for _ in range(n)]


def test_fedavg():
    grads = honest_grads(5) + poison_grads(2)
    agg = FedAvg().aggregate(grads)
    assert agg.shape == (50,)


def test_median():
    grads = honest_grads(5) + poison_grads(2)
    agg = CoordMedian().aggregate(grads)
    assert agg.shape == (50,)
    # Median should be close to honest gradient (≈1.0)
    assert abs(agg.mean().item() - 1.0) < 0.5


def test_krum():
    grads = honest_grads(5) + poison_grads(2)
    agg = Krum(f=2).aggregate(grads)
    assert agg.shape == (50,)


def test_flame():
    grads = honest_grads(5) + poison_grads(2)
    agg = FLAME().aggregate(grads)
    assert agg.shape == (50,)


def test_fltrust():
    grads = honest_grads(5) + poison_grads(2)
    server_grad = torch.ones(50)
    agg = FLTrust().aggregate(grads, server_gradient=server_grad)
    assert agg.shape == (50,)


def test_dnc():
    grads = honest_grads(6) + poison_grads(2)
    agg = DnC().aggregate(grads)
    assert agg.shape == (50,)


def test_bcbsa():
    d = 50
    n_honest, n_mal = 5, 2
    torch.manual_seed(0)
    grads = honest_grads(n_honest, d) + poison_grads(n_mal, d)
    z_cleans = [torch.randn(8, 4) for _ in range(n_honest + n_mal)]
    z_tildes = [z + 0.01 * torch.randn_like(z) for z in z_cleans]
    client_ids = list(range(n_honest + n_mal))

    bcbsa = bcbsa_full()
    agg, acc_ratio, trusts = bcbsa.aggregate(grads, client_ids, z_cleans, z_tildes)

    assert agg.shape == (d,)
    assert 0.0 <= acc_ratio <= 1.0
    assert len(trusts) == n_honest + n_mal


def test_bcbsa_nosem_aggregates():
    d = 50
    torch.manual_seed(1)
    grads = honest_grads(4, d) + poison_grads(2, d)
    z_cleans = [torch.randn(4, 4) for _ in range(6)]
    z_tildes = [z + 0.1 * torch.randn_like(z) for z in z_cleans]
    client_ids = list(range(6))
    bcbsa = bcbsa_nosem()
    agg, _, _ = bcbsa.aggregate(grads, client_ids, z_cleans, z_tildes)
    assert agg.shape == (d,)


def test_factory():
    for name in ["fedavg", "median", "krum", "flame", "fltrust", "dnc",
                 "bcbsa", "bcbsa_nofid", "bcbsa_nodist", "bcbsa_nosem", "bcbsa_notemp"]:
        defense = get_defense(name)
        assert defense is not None
