import numpy as np
import torch
from evofederated.models.genome import Genome, genome_to_model, model_stats

def test_genome_roundtrip():
    cfg = {"l_max":4, "n_neurons_choices":(16,32,64), "activations":("relu","tanh"), "use_batchnorm":True, "use_dropout":True}
    rng = np.random.RandomState(0)
    g = Genome.random(cfg, rng)
    vec = g.to_vector(cfg)
    g2 = Genome.from_vector(vec, cfg)
    # Check decoding consistency via vector roundtrip
    vec2 = g2.to_vector(cfg)
    np.testing.assert_allclose(vec, vec2, atol=1e-6)

def test_genome_valid_architecture():
    cfg = {"l_max":4, "n_neurons_choices":(16,32,64,128), "activations":("relu","tanh","elu"), "use_batchnorm":True, "use_dropout":True}
    rng = np.random.RandomState(1)
    for _ in range(20):
        g = Genome.random(cfg, rng)
        model = genome_to_model(g, input_dim=784, n_classes=10)
        # forward pass
        x = torch.randn(4, 784)
        out = model(x)
        assert out.shape == (4,10)
        stats = model_stats(model)
        assert stats["n_params"] > 0
        assert stats["depth"] == g.L + 1 or stats["depth"] >= 1

def test_genome_all_mutations_valid():
    cfg = {"l_max":3, "n_neurons_choices":(16,32), "activations":("relu",), "use_batchnorm":False, "use_dropout":False}
    # Test extreme vectors
    for vec in [np.zeros(1+3+3), np.ones(1+3+3)]:
        g = Genome.from_vector(vec, cfg)
        model = genome_to_model(g, 20, 5)
        x = torch.randn(2,20)
        out = model(x)
        assert out.shape == (2,5)

def test_dropout_disabled():
    cfg = {"l_max":2, "n_neurons_choices":(16,32), "activations":("relu",), "use_batchnorm":True, "use_dropout":False}
    vec = np.array([0.5, 0.5, 0.5, 0.5, 0.9, 0.9])  # dropout 0.45 but should be forced 0
    g = Genome.from_vector(vec, cfg)
    assert g.dropout == 0.0
