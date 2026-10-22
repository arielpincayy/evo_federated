import numpy as np
import torch

from evofederated.models.cnn_genome import CNNGenome, CNNSpace, cnn_to_model, max_n_params, n_params


def test_random_vectors_decode_to_valid_models():
    space = CNNSpace()
    rng = np.random.RandomState(0)
    for _ in range(20):
        g = CNNGenome.from_vector(rng.rand(space.n_var), space)
        assert 1 <= g.n_blocks <= space.max_blocks and len(g.channels) == g.n_blocks
        assert g.kernel in space.kernels and g.activation in space.activations
        out = cnn_to_model(g, (3, 32, 32), 10)(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 10)


def test_out_of_range_vector_is_clipped_and_max_params_bounds_all():
    space = CNNSpace()
    g = CNNGenome.from_vector(np.full(space.n_var, 5.0), space)
    assert g.n_blocks == space.max_blocks
    ref = max_n_params(space, (3, 32, 32), 10)
    rng = np.random.RandomState(1)
    for _ in range(20):
        m = cnn_to_model(CNNGenome.from_vector(rng.rand(space.n_var), space), (3, 32, 32), 10)
        assert n_params(m) <= ref
