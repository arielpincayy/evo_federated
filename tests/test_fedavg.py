"""FedAvg: promedio ponderado, rondas reproducibles, sin fuga de datos."""
import numpy as np
import torch
from torch.utils.data import TensorDataset

from evofederated.clients.client import FederatedClient
from evofederated.clients.server import FederatedServer
from evofederated.federated.fedavg import fedavg_buffers, fedavg_mean, run_fedavg, evaluate_global_on_split
from evofederated.models.genome import Genome, genome_to_model
from evofederated.federated.characterization import flatten_params


def _toy_clients():
    torch.manual_seed(0)
    clients = {}
    for cid in range(3):
        X = torch.randn(24, 6)
        y = torch.randint(0, 2, (24,))
        tr = TensorDataset(X[:16], y[:16])
        va = TensorDataset(X[16:20], y[16:20])
        te = TensorDataset(X[20:], y[20:])
        clients[cid] = FederatedClient(cid, tr, va, te, batch_size=8)
    return clients


def test_fedavg_weighted_mean():
    server = FederatedServer(_toy_clients())
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    out = server.fedavg_aggregate({0: a, 1: b}, {0: 1.0, 1: 3.0})
    np.testing.assert_allclose(out, np.array([2.5, 3.5]))


def test_fedavg_batchnorm_buffers():
    buffers = {
        0: {
            "running_mean": torch.tensor([1.0, 3.0]),
            "running_var": torch.tensor([2.0, 4.0]),
            "num_batches_tracked": torch.tensor(5, dtype=torch.long),
        },
        1: {
            "running_mean": torch.tensor([5.0, 7.0]),
            "running_var": torch.tensor([6.0, 8.0]),
            "num_batches_tracked": torch.tensor(11, dtype=torch.long),
        },
    }
    out = fedavg_buffers(buffers, {0: 1.0, 1: 3.0})
    torch.testing.assert_close(out["running_mean"], torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(out["running_var"], torch.tensor([5.0, 7.0]))
    assert out["num_batches_tracked"].item() == 11


def test_fedavg_rounds_reproducible_and_no_raw_access():
    clients = _toy_clients()
    genome_cfg = {"l_max": 2, "n_neurons_choices": (8, 16), "activations": ("relu",),
                  "use_batchnorm": False, "use_dropout": False}
    genome = Genome(L=1, hidden_sizes=[8, 8], activation="relu", dropout=0.0, use_bn=False, l_max=2)
    template = genome_to_model(genome, 6, 2)
    flat_init = flatten_params(template)
    sched = [[0, 1], [1, 2]]
    torch.manual_seed(123)
    r1 = run_fedavg(genome, genome_to_model, clients, 6, 2, flat_init, sched,
                    local_epochs=1, lr=1e-3, device="cpu")
    torch.manual_seed(123)
    r2 = run_fedavg(genome, genome_to_model, clients, 6, 2, flat_init, sched,
                    local_epochs=1, lr=1e-3, device="cpu")
    np.testing.assert_allclose(r1["global_params"], r2["global_params"])
    assert r1["n_local_trainings"] == 4
    assert r1["n_rounds"] == 2
    assert r1["communication_messages"] == 8
    assert r1["communication_bytes"] == 8 * r1["state_bytes"]
    # servidor no guarda datasets crudos
    server = FederatedServer(clients)
    server.collect_client_info()
    assert "n_train" in server.client_info[0]
    assert not hasattr(server, "train_dataset")
