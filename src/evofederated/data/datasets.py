"""Dataset loading + Dirichlet Non-IID partition + heterogeneity visuals."""
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
import pandas as pd

try:
    from torchvision import datasets, transforms
    HAS_TV = True
except ImportError:
    HAS_TV = False

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Base datasets
# ---------------------------------------------------------------------------

def get_dataset(name: str, root: str = "./data", train: bool = True, download: bool = True):
    name = name.lower()
    if name == "fashion-mnist":
        if not HAS_TV:
            raise ImportError("torchvision required for Fashion-MNIST")
        tf = transforms.Compose([transforms.ToTensor()])
        return datasets.FashionMNIST(root=root, train=train, download=download, transform=tf)
    elif name == "cifar10":
        if not HAS_TV:
            raise ImportError("torchvision required for CIFAR-10")
        if train:
            tf = transforms.Compose([transforms.ToTensor()])
        else:
            tf = transforms.Compose([transforms.ToTensor()])
        return datasets.CIFAR10(root=root, train=train, download=download, transform=tf)
    elif name == "synthetic":
        raise ValueError("Use make_synthetic_dataset() for synthetic; get_dataset only for image datasets")
    else:
        raise ValueError(f"Unknown dataset: {name}")


def to_tensor_dataset(ds, name: str):
    """Carga una vez en memoria (evita PIL/ToTensor por muestra). CIFAR-10 se normaliza por canal."""
    X = torch.as_tensor(ds.data)
    y = torch.as_tensor(ds.targets if not isinstance(ds.targets, torch.Tensor) else ds.targets.numpy()).long()
    if name.lower() == "cifar10":
        X = X.permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
        X = (X - mean) / std
    else:  # fashion-mnist: N x 28 x 28 uint8
        X = (X.float() / 255.0).unsqueeze(1)
    return TensorDataset(X, y)


def make_synthetic_dataset(
    n_samples: int = 5000,
    n_features: int = 64,
    n_informative: int = 30,
    n_classes: int = 10,
    n_redundant: int = 0,
    n_clusters_per_class: int = 2,
    class_sep: float = 0.8,
    flip_y: float = 0.05,
    seed: int = 42,
    standardize: bool = True,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=seed,
    )
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


# ---------------------------------------------------------------------------
# Dirichlet partition
# ---------------------------------------------------------------------------

@dataclass
class FederatedPartition:
    n_clients: int
    alpha: float
    seed: int
    min_samples_per_client: int = 10
    # mapping client_id -> indices list
    client_indices: Dict[int, List[int]] = field(default_factory=dict)
    # mapping client_id -> {'train':..., 'val':..., 'test':...}
    splits: Dict[int, Dict[str, List[int]]] = field(default_factory=dict)
    n_classes: int = 10
    class_distribution: Optional[np.ndarray] = None  # shape (n_clients, n_classes)

    def to_dict(self):
        return {
            "n_clients": self.n_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "min_samples_per_client": self.min_samples_per_client,
            "n_classes": self.n_classes,
            "client_indices": {str(k): v for k, v in self.client_indices.items()},
            "splits": {str(k): {sk: sv for sk, sv in v.items()} for k, v in self.splits.items()},
        }


def dirichlet_partition(
    labels: np.ndarray,
    n_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples_per_client: int = 10,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> FederatedPartition:
    """
    Partition indices via Dirichlet over classes.

    labels: np array shape (N,) with 0..C-1
    """
    rng = np.random.RandomState(seed)
    n_classes = int(labels.max() + 1)
    n = len(labels)

    # group indices per class
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(n_classes)}
    for c in class_indices:
        rng.shuffle(class_indices[c])

    # For each class, sample Dirichlet proportions for clients
    # We'll iteratively assign.
    client_indices: Dict[int, List[int]] = {i: [] for i in range(n_clients)}

    for c in range(n_classes):
        idxs = class_indices[c]
        if len(idxs) == 0:
            continue
        # Sample Dirichlet
        proportions = rng.dirichlet([alpha] * n_clients)
        # Adjust to avoid very tiny assignments? Keep proportions as sampled
        # Convert to counts
        counts = (proportions * len(idxs)).astype(int)
        # fix rounding remainder
        remainder = len(idxs) - counts.sum()
        # distribute remainder to largest proportions
        if remainder > 0:
            order = np.argsort(proportions)[::-1]
            for i in range(remainder):
                counts[order[i % n_clients]] += 1
        # ensure shuffle before slicing
        # We already shuffled per class, now allocate slices
        start = 0
        for client_id in range(n_clients):
            cnt = int(counts[client_id])
            if cnt > 0:
                client_indices[client_id].extend(idxs[start:start+cnt])
                start += cnt

    # Enforce min_samples_per_client by rebalancing if needed (simple retry logic)
    # If any client has < min, we redistribute from largest client
    # Do a few iterations
    for _ in range(10):
        sizes = [len(client_indices[i]) for i in range(n_clients)]
        smallest = int(np.argmin(sizes))
        if sizes[smallest] >= min_samples_per_client:
            break
        largest = int(np.argmax(sizes))
        if largest == smallest:
            break
        # move one random sample
        move_idx = int(rng.choice(client_indices[largest]))
        client_indices[largest].remove(move_idx)
        client_indices[smallest].append(move_idx)

    # Shuffle per client
    for cid in client_indices:
        rng.shuffle(client_indices[cid])

    # Build train/val/test splits per client
    splits: Dict[int, Dict[str, List[int]]] = {}
    for cid in range(n_clients):
        idxs = client_indices[cid]
        rng2 = np.random.RandomState(seed + cid + 999)
        rng2.shuffle(idxs)
        n_total = len(idxs)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_test - n_val
        # ensure at least 1 test/val if possible? keep as computed
        splits[cid] = {
            "train": idxs[:n_train],
            "val": idxs[n_train:n_train+n_val],
            "test": idxs[n_train+n_val:],
        }

    # class distribution matrix
    class_dist = np.zeros((n_clients, n_classes), dtype=int)
    for cid in range(n_clients):
        for idx in client_indices[cid]:
            class_dist[cid, labels[idx]] += 1

    return FederatedPartition(
        n_clients=n_clients,
        alpha=alpha,
        seed=seed,
        min_samples_per_client=min_samples_per_client,
        client_indices=client_indices,
        splits=splits,
        n_classes=n_classes,
        class_distribution=class_dist,
    )


def heterogeneity_metrics(class_dist: np.ndarray):
    """
    Compute heterogeneity statistics from client x class count matrix.
    Returns dict with metrics.
    """
    # normalize per client to proportions
    row_sums = class_dist.sum(axis=1, keepdims=True).clip(min=1)
    props = class_dist / row_sums
    # global proportion
    global_counts = class_dist.sum(axis=0)
    global_props = global_counts / global_counts.sum()

    # TV distance per client vs global, etc.
    # Jensen etc. simplified
    # Coefficient of variation of client sizes
    sizes = class_dist.sum(axis=1).astype(float)
    cv_size = float(sizes.std() / (sizes.mean() + 1e-9))

    # Mean pairwise Jensen-Shannon? Use L1 for simplicity
    # Pairwise L1 between clients
    n_clients = class_dist.shape[0]
    pairwise_l1 = []
    for i in range(n_clients):
        for j in range(i+1, n_clients):
            pairwise_l1.append(np.abs(props[i] - props[j]).sum() / 2)  # TV

    # KL-like entropy? Use Gini? Simplified: std of props per class
    std_per_class = props.std(axis=0).mean()

    return {
        "cv_size": cv_size,
        "mean_pairwise_TV": float(np.mean(pairwise_l1)) if pairwise_l1 else 0.0,
        "max_pairwise_TV": float(np.max(pairwise_l1)) if pairwise_l1 else 0.0,
        "mean_std_props": float(std_per_class),
        "client_sizes": sizes.tolist(),
        "global_props": global_props.tolist(),
    }


# ---------------------------------------------------------------------------
# Helpers to create federated client datasets without leaking raw data
# ---------------------------------------------------------------------------

def create_client_subsets(base_dataset: Dataset, partition: FederatedPartition, split: str = "train") -> Dict[int, Subset]:
    """Create Subset per client for a given split (train/val/test). Server should not aggregate raw data."""
    out = {}
    for cid, s in partition.splits.items():
        idxs = s[split]
        out[cid] = Subset(base_dataset, idxs)
    return out


def get_input_dim_and_n_classes(dataset: Dataset):
    """Infer input_dim (flattened) and n_classes."""
    # Try to infer from first sample
    x, y = dataset[0]
    if isinstance(x, torch.Tensor):
        # image Tensor: shape C x H x W
        input_dim = int(np.prod(x.shape))
    else:
        input_dim = int(np.prod(np.array(x).shape))
    # n_classes: look at labels if possible
    # For TensorDataset
    if hasattr(dataset, "tensors"):
        n_classes = int(dataset.tensors[1].max().item() + 1)
    elif hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            n_classes = int(targets.max().item() + 1)
        else:
            # list
            n_classes = int(max(targets) + 1)
    elif hasattr(dataset, "classes"):
        n_classes = len(dataset.classes)
    else:
        # fallback: iterate unique?
        # approximate
        n_classes = 10
    return input_dim, n_classes
