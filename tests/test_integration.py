"""Integration test: end-to-end federated training round."""
import torch
import pytest

from scl.models.resnet import build_resnet18_split
from scl.channels import get_channel
from scl.attacks import get_attack
from scl.defenses import get_defense
from scl.data.datasets import SyntheticDataset
from scl.data.partition import iid_partition, make_client_datasets
from scl.training.federated import FederatedTrainer
from torch.utils.data import DataLoader


@pytest.fixture
def tiny_trainer():
    torch.manual_seed(42)
    num_clients = 4
    num_classes = 10

    # Tiny synthetic datasets
    train_ds = SyntheticDataset(200, (3, 32, 32), num_classes)
    test_ds = SyntheticDataset(40, (3, 32, 32), num_classes)

    splits = iid_partition(train_ds, num_clients, seed=0)
    client_datasets = make_client_datasets(train_ds, splits)

    test_loader = DataLoader(test_ds, batch_size=20, shuffle=False)

    import copy
    client_proto, server_model = build_resnet18_split(split_layer=2, num_classes=num_classes)
    client_models = [copy.deepcopy(client_proto) for _ in range(num_clients)]

    channel = get_channel("awgn")
    defense = get_defense("fedavg")

    trainer = FederatedTrainer(
        client_models=client_models,
        server_model=server_model,
        train_datasets=client_datasets,
        test_loader=test_loader,
        channel=channel,
        attack=None,
        defense=defense,
        honest_ids=list(range(num_clients)),
        malicious_ids=[],
        criterion=torch.nn.CrossEntropyLoss(),
        device="cpu",
        batch_size=16,
        lr=0.01,
        warmup_rounds=1,
        total_rounds=3,
        num_classes=num_classes,
    )
    return trainer


def test_single_round(tiny_trainer):
    metrics = tiny_trainer.run_round(round_t=1, snr_db=15.0)
    assert 0.0 <= metrics.test_accuracy <= 1.0
    assert metrics.train_loss > 0.0
    assert metrics.wall_clock_sec > 0.0


def test_training_loop(tiny_trainer):
    tracker = tiny_trainer.train(snr_db=15.0, total_rounds=3)
    assert len(tracker) == 3
    assert all(0.0 <= m.test_accuracy <= 1.0 for m in tracker.history)


@pytest.mark.parametrize("defense_name", ["fedavg", "median", "bcbsa"])
def test_defense_variants(defense_name):
    torch.manual_seed(0)
    num_clients = 4
    nc = 10

    train_ds = SyntheticDataset(80, (3, 32, 32), nc)
    test_ds = SyntheticDataset(20, (3, 32, 32), nc)
    splits = iid_partition(train_ds, num_clients, seed=0)
    client_datasets = make_client_datasets(train_ds, splits)
    test_loader = DataLoader(test_ds, batch_size=20)

    import copy
    client_proto, server_model = build_resnet18_split(2, nc)
    client_models = [copy.deepcopy(client_proto) for _ in range(num_clients)]

    channel = get_channel("awgn")
    defense = get_defense(defense_name)

    trainer = FederatedTrainer(
        client_models=client_models,
        server_model=server_model,
        train_datasets=client_datasets,
        test_loader=test_loader,
        channel=channel,
        attack=None,
        defense=defense,
        honest_ids=list(range(num_clients)),
        malicious_ids=[],
        criterion=torch.nn.CrossEntropyLoss(),
        device="cpu",
        batch_size=8,
        lr=0.01,
        warmup_rounds=1,
        total_rounds=2,
        num_classes=nc,
    )
    metrics = trainer.run_round(round_t=1, snr_db=15.0)
    assert 0.0 <= metrics.test_accuracy <= 1.0


def test_attack_weight_poison():
    """Verify that a weight poisoning attack produces valid (but degraded) training."""
    torch.manual_seed(5)
    num_clients = 4
    nc = 10

    train_ds = SyntheticDataset(80, (3, 32, 32), nc)
    test_ds = SyntheticDataset(20, (3, 32, 32), nc)
    splits = iid_partition(train_ds, num_clients, seed=0)
    client_datasets = make_client_datasets(train_ds, splits)
    test_loader = DataLoader(test_ds, batch_size=20)

    import copy
    client_proto, server_model = build_resnet18_split(2, nc)
    client_models = [copy.deepcopy(client_proto) for _ in range(num_clients)]

    channel = get_channel("awgn")
    attack = get_attack("weight_poison")
    defense = get_defense("fedavg")

    trainer = FederatedTrainer(
        client_models=client_models,
        server_model=server_model,
        train_datasets=client_datasets,
        test_loader=test_loader,
        channel=channel,
        attack=attack,
        defense=defense,
        honest_ids=[0, 1, 2],
        malicious_ids=[3],
        criterion=torch.nn.CrossEntropyLoss(),
        device="cpu",
        batch_size=8,
        lr=0.01,
        warmup_rounds=1,
        total_rounds=2,
        num_classes=nc,
    )
    metrics = trainer.run_round(round_t=1, snr_db=15.0)
    assert 0.0 <= metrics.test_accuracy <= 1.0
