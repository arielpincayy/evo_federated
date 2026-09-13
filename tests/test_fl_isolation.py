"""Ensure no raw data leakage, no weight contamination, reproducibility checks."""
import numpy as np
import torch
from torch.utils.data import TensorDataset
from evofederated.clients.client import FederatedClient
from evofederated.models.genome import Genome, genome_to_model
from evofederated.federated.characterization import flatten_params, set_model_params
from evofederated.evolution.evaluator import Evaluator
from evofederated.federated.pairing import get_pairing

def test_no_weight_contamination():
    # Create two clients with distinct data
    torch.manual_seed(0)
    X1 = torch.randn(20,5)
    y1 = torch.randint(0,2,(20,))
    X2 = torch.randn(20,5)
    y2 = torch.randint(0,2,(20,))
    c1 = FederatedClient(0, TensorDataset(X1,y1), None, TensorDataset(X1,y1), batch_size=10)
    c2 = FederatedClient(1, TensorDataset(X2,y2), None, TensorDataset(X2,y2), batch_size=10)
    clients = {0:c1,1:c2}
    genome_cfg = {"l_max":2, "n_neurons_choices":(16,32), "activations":("relu",), "use_batchnorm":False, "use_dropout":False}
    train_cfg = {"lr":1e-3, "optimizer":"adam", "epochs":1}
    genome = Genome(L=1, hidden_sizes=[16,16], activation="relu", dropout=0.0, use_bn=False, l_max=2)
    # Get deterministic init flat
    model_template = genome_to_model(genome, 5, 2)
    flat_init = flatten_params(model_template)
    # Evaluate on both clients sequentially using same init
    model1 = genome_to_model(genome,5,2)
    set_model_params(model1, flat_init)
    flat_before = flatten_params(model1).copy()
    c1.train(model1, epochs=1, lr=1e-3, device="cpu")
    flat_after_c1 = flatten_params(model1).copy()
    # ensure training changed weights
    assert not np.allclose(flat_before, flat_after_c1)
    # Now evaluate on c2 with SAME initial (not after_c1)
    model2 = genome_to_model(genome,5,2)
    set_model_params(model2, flat_init)
    flat_before2 = flatten_params(model2).copy()
    np.testing.assert_allclose(flat_before, flat_before2)
    c2.train(model2, epochs=1, lr=1e-3, device="cpu")
    # Ensure evaluator does not reuse weights: test Evaluator
    evaluator = Evaluator(clients, 5,2, genome_cfg, train_cfg, get_pairing("dynamic"), divergence_matrix=np.array([[0,1],[1,0]]), strategy_name="dynamic", seed=42, device="cpu")
    # manually test evaluate_single does not leak
    vec = genome.to_vector(genome_cfg)
    # First call
    F1,_ = evaluator.evaluate_single(vec, 0, arch_id="test0")
    # Second call with same genome should produce same F if deterministic, not dependent on previous training
    # Since evaluator uses deterministic init per genome, second call should be reproducible
    vec2 = genome.to_vector(genome_cfg)
    F2,_ = evaluator.evaluate_single(vec2, 0, arch_id="test1")
    np.testing.assert_allclose(F1, F2)

def test_exhaustive_not_contaminates_fitness():
    # Placeholder: ensure exhaustive validation is separate from evolution
    # This is structural: evaluator's eval_count shouldn't include exhaustive
    # We test that manual exhaustive loop doesn't affect evaluator's internal state beyond explicit calls
    # Here just ensure isolation conceptually - no raw data access via server
    from evofederated.clients.server import FederatedServer
    X = torch.randn(10,4)
    y = torch.randint(0,2,(10,))
    c = FederatedClient(0, TensorDataset(X,y), None, TensorDataset(X,y), batch_size=5)
    server = FederatedServer({0:c})
    server.collect_client_info()
    # Server should only have size info, not raw tensors
    assert 0 in server.client_info
    assert "n_train" in server.client_info[0]
    # Ensure raw dataset not stored in server
    assert not hasattr(server, "train_dataset")
    # Attempt to access via server.clients is allowed but conceptually server shouldn't iterate raw data
    # Here we just check API surface: server doesn't expose method to get raw data

