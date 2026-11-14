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


FC_SPACE = CNNSpace(max_blocks=3, channels=(8, 16, 32, 64), kernels=(3, 5), activations=("relu", "elu"), max_fc=2, fc_widths=(32, 64, 128, 256), pool=2)


def test_fc_space_decodes_valid_models_and_max_params_bounds_all():
    assert FC_SPACE.n_var == 3 + 2 + 4
    ref = max_n_params(FC_SPACE, (3, 32, 32), 10)
    rng = np.random.RandomState(0)
    for _ in range(20):
        g = CNNGenome.from_vector(rng.rand(FC_SPACE.n_var), FC_SPACE)
        assert len(g.fc) <= FC_SPACE.max_fc and all(w in FC_SPACE.fc_widths for w in g.fc) and g.pool == 2
        m = cnn_to_model(g, (3, 32, 32), 10)
        assert m(torch.randn(2, 3, 32, 32)).shape == (2, 10) and n_params(m) <= ref
    assert len(CNNGenome.from_vector(np.ones(FC_SPACE.n_var), FC_SPACE).fc) == 2


def test_off_slots_are_skipped_and_first_conv_never_off():
    v = np.array([0.0, 0.1, 0.9, 0.05, 0.9, 0.5, 0.5, 0.5, 0.9])  # conv_1 en 0 (sigue activo), conv_2 apagado, conv_3 activo, fc_1 apagado, fc_2 activo
    g = CNNGenome.from_vector(v, FC_SPACE)
    assert g.channels == [8, 64] and g.n_blocks == 2 and g.fc == [256]


def test_probe_architecture_is_unchanged():
    # la sonda de los estudios de clusters (fc=[], pool=1) debe seguir teniendo 1610 parámetros: mismos Δ y clusters
    assert n_params(cnn_to_model(CNNGenome(2, [8, 16], 3, "relu", 0.0, True), (3, 32, 32), 10)) == 1610
