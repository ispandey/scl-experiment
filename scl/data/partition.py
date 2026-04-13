"""Client data partitioning: IID and Dirichlet non-IID."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from torch.utils.data import Dataset, Subset


def iid_partition(dataset: Dataset, num_clients: int, seed: int = 42) -> List[List[int]]:
    """Uniform random partition across K clients."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = rng.permutation(n).tolist()
    per_client = n // num_clients
    splits = []
    for k in range(num_clients):
        start = k * per_client
        end = start + per_client if k < num_clients - 1 else n
        splits.append(indices[start:end])
    return splits


def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    num_classes: int = 10,
    seed: int = 42,
) -> List[List[int]]:
    """
    Dirichlet(alpha) partition as specified in the design document.

    Args:
        dataset:     Dataset with a `.targets` attribute (list/array of int labels).
        num_clients: Number of clients K.
        alpha:       Dirichlet concentration (0.1 = severe, 0.5 = mild, large = IID).
        num_classes: Total number of label classes.
        seed:        Random seed for reproducibility.

    Returns:
        List of K index lists.
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    label_indices: dict = {c: np.where(targets == c)[0] for c in range(num_classes)}
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(label_indices[c])).astype(int)
        proportions[-1] = len(label_indices[c]) - proportions[:-1].sum()
        idx = label_indices[c].copy()
        rng.shuffle(idx)
        splits = np.split(idx, np.cumsum(proportions)[:-1])
        for k, s in enumerate(splits):
            client_indices[k].extend(s.tolist())

    return client_indices


def make_client_datasets(
    dataset: Dataset,
    client_indices: List[List[int]],
) -> List[Subset]:
    """Wrap raw index lists into torch Subsets."""
    return [Subset(dataset, idxs) for idxs in client_indices]


def partition_dataset(
    dataset: Dataset,
    num_clients: int,
    partition: str = "iid",
    num_classes: int = 10,
    seed: int = 42,
) -> List[Subset]:
    """
    High-level partitioning interface.

    Args:
        partition: 'iid' | 'dir05' | 'dir01' | 'dir<alpha>'
    """
    if partition == "iid":
        idxs = iid_partition(dataset, num_clients, seed)
    elif partition in ("dir05", "dir0.5"):
        idxs = dirichlet_partition(dataset, num_clients, 0.5, num_classes, seed)
    elif partition in ("dir01", "dir0.1"):
        idxs = dirichlet_partition(dataset, num_clients, 0.1, num_classes, seed)
    elif partition.startswith("dir"):
        alpha = float(partition[3:])
        idxs = dirichlet_partition(dataset, num_clients, alpha, num_classes, seed)
    else:
        raise ValueError(f"Unknown partition scheme '{partition}'")
    return make_client_datasets(dataset, idxs)
