"""Utilidades compartidas: nodos de juguete con 3 grupos de clases (heterogeneidad conocida)."""
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from evofederated.clients.client import FederatedClient

torch.set_num_threads(4)

N_PER_GROUP, N_GROUPS, N_CLASSES = 4, 3, 6


def make_toy_clients(seed=0):
    rng = np.random.RandomState(seed)
    clients, group_of = {}, []

    def split(classes, n):
        y = rng.choice(classes, size=n)
        x = rng.randn(n, 1, 8, 8).astype("float32") * 0.3 + y[:, None, None, None].astype("float32")
        return TensorDataset(torch.tensor(x), torch.tensor(y))

    for cid in range(N_PER_GROUP * N_GROUPS):
        g = cid // N_PER_GROUP
        cls = [2 * g, 2 * g + 1]
        clients[cid] = FederatedClient(cid, split(cls, 40), split(cls, 12), split(cls, 12), batch_size=16)
        group_of.append(g)
    return clients, np.array(group_of)


@pytest.fixture
def toy():
    return make_toy_clients()
