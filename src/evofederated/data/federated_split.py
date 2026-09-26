"""Construcción federada con test oficial separado (Fashion-MNIST/CIFAR-10).

- Train oficial -> partición Dirichlet por cliente -> split local train/val.
- Test oficial -> partición Dirichlet independiente (misma alpha, seed+offset)
  para asignar test por cliente sin mezclar con train.
- Sintético -> genera train y test separados con distinta seed.
"""
import numpy as np
import torch
from torch.utils.data import Subset

from .datasets import (
    get_dataset,
    make_synthetic_dataset,
    dirichlet_partition,
    get_input_dim_and_n_classes,
)
from ..clients.client import FederatedClient


def _labels_of(dataset):
    if hasattr(dataset, "tensors"):
        return dataset.tensors[1].numpy()
    if hasattr(dataset, "targets"):
        t = dataset.targets
        return np.array(t) if not isinstance(t, torch.Tensor) else t.numpy()
    return np.array([dataset[i][1] for i in range(len(dataset))])


def build_federated_clients(cfg, seed: int | None = None):
    seed = cfg.seed if seed is None else int(seed)
    name = cfg.dataset.name.lower()
    n_clients = int(cfg.dataset.n_clients)
    alpha = float(cfg.dataset.alpha)

    if name == "synthetic":
        train_base = make_synthetic_dataset(
            n_samples=cfg.dataset.synthetic_n_samples,
            n_features=cfg.dataset.synthetic_n_features,
            n_informative=cfg.dataset.synthetic_n_informative,
            n_classes=cfg.dataset.synthetic_n_classes,
            seed=seed,
        )
        test_base = make_synthetic_dataset(
            n_samples=max(500, cfg.dataset.synthetic_n_samples // 3),
            n_features=cfg.dataset.synthetic_n_features,
            n_informative=cfg.dataset.synthetic_n_informative,
            n_classes=cfg.dataset.synthetic_n_classes,
            seed=seed + 10_000,
        )
        train_labels = train_base.tensors[1].numpy()
        test_labels = test_base.tensors[1].numpy()
        train_part = dirichlet_partition(
            train_labels, n_clients, alpha, seed,
            cfg.dataset.min_samples_per_client, cfg.dataset.val_ratio, 0.0,
        )
        # test oficial separado: todo a test, sin val
        test_part = dirichlet_partition(
            test_labels, n_clients, alpha, seed + 7_777,
            1, 0.0, 0.0,
        )
        input_dim, n_classes = get_input_dim_and_n_classes(train_base)
        clients = {}
        for cid in range(n_clients):
            tr = train_part.splits[cid]["train"]
            va = train_part.splits[cid]["val"]
            te = test_part.splits[cid]["train"] + test_part.splits[cid]["val"] + test_part.splits[cid]["test"]
            clients[cid] = FederatedClient(
                cid,
                Subset(train_base, tr) if tr else None,
                Subset(train_base, va) if va else None,
                Subset(test_base, te) if te else None,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers,
            )
        return clients, {"input_dim": input_dim, "n_classes": n_classes,
                         "train_partition": train_part, "test_partition": test_part}

    # imagen: train y test oficiales separados
    train_base = get_dataset(name, root=cfg.dataset.data_root, train=True, download=True)
    test_base = get_dataset(name, root=cfg.dataset.data_root, train=False, download=True)
    train_labels = _labels_of(train_base)
    test_labels = _labels_of(test_base)
    train_part = dirichlet_partition(
        train_labels, n_clients, alpha, seed,
        cfg.dataset.min_samples_per_client, cfg.dataset.val_ratio, 0.0,
    )
    test_part = dirichlet_partition(
        test_labels, n_clients, alpha, seed + 7_777, 1, 0.0, 0.0,
    )
    input_dim, n_classes = get_input_dim_and_n_classes(train_base)
    clients = {}
    for cid in range(n_clients):
        tr = train_part.splits[cid]["train"]
        va = train_part.splits[cid]["val"]
        te = test_part.splits[cid]["train"] + test_part.splits[cid]["val"] + test_part.splits[cid]["test"]
        clients[cid] = FederatedClient(
            cid,
            Subset(train_base, tr) if tr else None,
            Subset(train_base, va) if va else None,
            Subset(test_base, te) if te else None,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
        )
    return clients, {"input_dim": input_dim, "n_classes": n_classes,
                     "train_partition": train_part, "test_partition": test_part}
