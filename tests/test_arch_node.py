import numpy as np

from evofederated.metrics.arch_node import perm_p, selection_gain, stable_interaction


def _data(inter_sd, seed=0, A=20, N=60, noise=0.05):
    rng = np.random.RandomState(seed)
    base = rng.randn(A, 1) * 0.05 + rng.randn(1, N) * 0.1 + 0.5  # efectos aditivos de arquitectura y nodo
    Z = rng.randn(A, N) * inter_sd  # interacción verdadera, común a réplicas y splits
    mk = lambda: np.stack([base + Z + rng.randn(A, N) * noise for _ in range(2)])  # ruido independiente por (réplica, split)
    return mk(), mk(), np.arange(N) % 5


def test_no_interaction_gives_zero_variance_negative_gain_and_large_p():
    Yv, Yt, lab = _data(0.0)
    assert abs(stable_interaction(Yv, Yt)) < 5e-4
    assert perm_p(Yv, Yt, 500) > 0.05
    gn, gc = selection_gain(Yv, Yt, lab)
    assert gn < 0 and gc < 0


def test_planted_interaction_is_recovered():
    Yv, Yt, lab = _data(0.1)
    assert abs(stable_interaction(Yv, Yt) - 0.1 ** 2) < 0.003
    assert perm_p(Yv, Yt, 500) < 0.01
    assert selection_gain(Yv, Yt, lab)[0] > 0
