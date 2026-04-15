"""Tests for data loading and partitioning."""
import numpy as np
import pytest

from scl.data.datasets import get_dataset, num_classes, SyntheticDataset
from scl.data.partition import iid_partition, dirichlet_partition, partition_dataset


@pytest.fixture
def synthetic_cifar10():
    return SyntheticDataset(500, (3, 32, 32), 10)


def test_synthetic_dataset(synthetic_cifar10):
    ds = synthetic_cifar10
    assert len(ds) == 500
    x, y = ds[0]
    assert x.shape == (3, 32, 32)
    assert 0 <= y < 10


def test_get_dataset_raises_when_offline():
    """get_dataset should raise RuntimeError (not fall back silently) when offline."""
    with pytest.raises(RuntimeError, match="Failed to load dataset 'cifar10'"):
        get_dataset("cifar10", train=True)


def test_iid_partition(synthetic_cifar10):
    splits = iid_partition(synthetic_cifar10, num_clients=5, seed=42)
    assert len(splits) == 5
    total = sum(len(s) for s in splits)
    assert total == len(synthetic_cifar10)
    # All indices unique
    all_idx = [i for s in splits for i in s]
    assert len(set(all_idx)) == len(all_idx)


def test_dirichlet_partition(synthetic_cifar10):
    splits = dirichlet_partition(
        synthetic_cifar10, num_clients=5, alpha=0.5, num_classes=10, seed=42
    )
    assert len(splits) == 5
    total = sum(len(s) for s in splits)
    assert total == len(synthetic_cifar10)


def test_partition_dataset_iid(synthetic_cifar10):
    subsets = partition_dataset(synthetic_cifar10, num_clients=4, partition="iid")
    assert len(subsets) == 4


def test_partition_dataset_dirichlet(synthetic_cifar10):
    for scheme in ("dir05", "dir01"):
        subsets = partition_dataset(synthetic_cifar10, num_clients=4, partition=scheme)
        assert len(subsets) == 4


def test_num_classes():
    assert num_classes("cifar10") == 10
    assert num_classes("cifar100") == 100
    assert num_classes("mnist") == 10
    assert num_classes("tinyimagenet") == 200
