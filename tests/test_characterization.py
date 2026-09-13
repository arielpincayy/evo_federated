import numpy as np
import torch
from torch.utils.data import TensorDataset
from evofederated.clients.client import FederatedClient
from evofederated.federated.characterization import characterize_clients, flatten_params
from evofederated.data.datasets import make_synthetic_dataset

def test_characterization_dimensions():
    # small synthetic
    full = make_synthetic_dataset(n_samples=200, n_features=10, n_informative=5, n_classes=3, seed=0)
    # split into 3 clients manually
    labels = full.tensors[1].numpy()
    clients = {}
    # simple split: first 70, next 70, next 60
    for i, (start,end) in enumerate([(0,70),(70,140),(140,200)]):
        ds = TensorDataset(full.tensors[0][start:end], full.tensors[1][start:end])
        clients[i] = FederatedClient(i, ds, None, ds, batch_size=16)
    genome_cfg = {"l_max":2, "n_neurons_choices":(16,32), "activations":("relu",), "use_batchnorm":False, "use_dropout":False}
    train_cfg = {"lr":1e-3, "optimizer":"adam", "epochs":2}
    res = characterize_clients(clients, decoder_input_dim=10, decoder_n_classes=3, genome_config=genome_cfg, train_config=train_cfg, k_epochs=2, seed=42, device="cpu")
    assert "probe_genome" in res
    assert "theta0" in res
    assert "deltas" in res
    n_params = res["n_params"]
    for cid, delta in res["deltas"].items():
        assert delta.shape[0] == n_params
        assert delta.shape == res["theta0"].shape
    # deltas should not all be zero (training happened)
    norms = [np.linalg.norm(d) for d in res["deltas"].values()]
    assert any(n>1e-6 for n in norms)
    # All clients received same initial: theta0 consistent
    # Check that flatten works

def test_same_probe_for_all_clients():
    full = make_synthetic_dataset(n_samples=150, n_features=8, n_informative=4, n_classes=2, seed=1)
    clients = {}
    for i in range(2):
        ds = TensorDataset(full.tensors[0][i*75:(i+1)*75], full.tensors[1][i*75:(i+1)*75])
        clients[i] = FederatedClient(i, ds, None, ds, batch_size=16)
    cfg = {"l_max":2, "n_neurons_choices":(16,32), "activations":("relu","tanh"), "use_batchnorm":True, "use_dropout":True}
    train_cfg = {"lr":1e-3, "optimizer":"adam", "epochs":1}
    res1 = characterize_clients(clients, 8, 2, cfg, train_cfg, k_epochs=1, seed=123, device="cpu")
    res2 = characterize_clients(clients, 8, 2, cfg, train_cfg, k_epochs=1, seed=123, device="cpu")
    np.testing.assert_allclose(res1["theta0"], res2["theta0"])
    assert res1["probe_genome"].to_dict() == res2["probe_genome"].to_dict()
    res3 = characterize_clients(clients, 8, 2, cfg, train_cfg, k_epochs=1, seed=999, device="cpu")
    # different seed => different probe (likely)
    assert res1["probe_genome"].to_dict() != res3["probe_genome"].to_dict()

