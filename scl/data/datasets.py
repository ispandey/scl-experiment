"""Dataset loaders for CIFAR-10/100, TinyImageNet, MNIST, and a synthetic fallback."""
from __future__ import annotations

import os
import warnings
from typing import Optional

import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _cifar10_train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def _cifar10_test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


def _cifar100_train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])


def _cifar100_test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])


def _tinyimagenet_train_transform():
    return T.Compose([
        T.Resize(64),
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])


def _tinyimagenet_test_transform():
    return T.Compose([
        T.Resize(64),
        T.ToTensor(),
        T.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
    ])


def _mnist_train_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ])


def _mnist_test_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ])


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """Synthetic random dataset for offline / test environments."""

    def __init__(self, num_samples: int, input_shape: tuple, num_classes: int):
        self.data = torch.randn(num_samples, *input_shape)
        self.targets = torch.randint(0, num_classes, (num_samples,)).tolist()
        self.num_classes = num_classes

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def _make_synthetic(name: str, train: bool) -> Dataset:
    specs = {
        "cifar10":       ((3, 32, 32), 10, 50000, 10000),
        "cifar100":      ((3, 32, 32), 100, 50000, 10000),
        "tinyimagenet":  ((3, 64, 64), 200, 100000, 10000),
        "mnist":         ((1, 28, 28), 10, 60000, 10000),
    }
    shape, nc, n_train, n_test = specs.get(name, ((3, 32, 32), 10, 5000, 1000))
    n = n_train if train else n_test
    return SyntheticDataset(n, shape, nc)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

_DATA_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "scl_data")


def get_dataset(name: str, train: bool = True, data_root: Optional[str] = None) -> Dataset:
    """
    Load a dataset by name.  Falls back to synthetic data if download fails.

    Args:
        name: 'cifar10' | 'cifar100' | 'tinyimagenet' | 'mnist'
        train: train split if True, test split otherwise
        data_root: directory to cache downloads (defaults to ~/.cache/scl_data)

    Returns:
        A PyTorch Dataset with a `.targets` attribute (list of int labels).
    """
    root = data_root or _DATA_ROOT
    os.makedirs(root, exist_ok=True)

    loaders = {
        "cifar10": _load_cifar10,
        "cifar100": _load_cifar100,
        "tinyimagenet": _load_tinyimagenet,
        "mnist": _load_mnist,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(loaders)}")

    try:
        return loaders[name](root, train)
    except Exception as exc:
        warnings.warn(
            f"Failed to load '{name}' ({exc}). Falling back to synthetic dataset.",
            UserWarning,
        )
        return _make_synthetic(name, train)


def _load_cifar10(root: str, train: bool) -> Dataset:
    transform = _cifar10_train_transform() if train else _cifar10_test_transform()
    return torchvision.datasets.CIFAR10(root, train=train, transform=transform, download=True)


def _load_cifar100(root: str, train: bool) -> Dataset:
    transform = _cifar100_train_transform() if train else _cifar100_test_transform()
    return torchvision.datasets.CIFAR100(root, train=train, transform=transform, download=True)


def _load_mnist(root: str, train: bool) -> Dataset:
    transform = _mnist_train_transform() if train else _mnist_test_transform()
    return torchvision.datasets.MNIST(root, train=train, transform=transform, download=True)


def _load_tinyimagenet(root: str, train: bool) -> Dataset:
    tiny_root = os.path.join(root, "tiny-imagenet-200")
    if not os.path.isdir(tiny_root):
        _download_tinyimagenet(root)
    split = "train" if train else "val"
    split_dir = os.path.join(tiny_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"TinyImageNet split dir not found: {split_dir}")
    transform = _tinyimagenet_train_transform() if train else _tinyimagenet_test_transform()
    ds = torchvision.datasets.ImageFolder(split_dir, transform=transform)
    # add `.targets` list for compatibility
    ds.targets = [lbl for _, lbl in ds.samples]
    return ds


def _download_tinyimagenet(root: str):
    import urllib.request
    import zipfile

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    os.remove(zip_path)


def num_classes(name: str) -> int:
    return {"cifar10": 10, "cifar100": 100, "tinyimagenet": 200, "mnist": 10}[name]
