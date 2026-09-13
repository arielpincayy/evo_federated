import numpy as np
from evofederated.federated.pairing import get_pairing

def test_random_never_self():
    D = np.array([[0,1,0.5],[1,0,0.2],[0.5,0.2,0]])
    strat = get_pairing("random")
    rng = np.random.RandomState(0)
    for i in range(3):
        for _ in range(20):
            j = strat.select_counterpart(i, D, rng)
            assert j != i

def test_fixed_most_divergent():
    D = np.array([[0,1,0.5],[1,0,0.2],[0.5,0.2,0]])
    strat = get_pairing("fixed")
    rng = np.random.RandomState(0)
    assert strat.select_counterpart(0, D, rng)==1
    assert strat.select_counterpart(1, D, rng)==0
    assert strat.select_counterpart(2, D, rng)==0

def test_dynamic_probabilities():
    D = np.array([[0,1,0.5],[1,0,0.2],[0.5,0.2,0]])
    strat = get_pairing("dynamic", tau=100.0) # high tau => almost deterministic max
    rng = np.random.RandomState(0)
    # with high tau, from 0 should almost always pick 1 (max distance 1.0)
    picks = [strat.select_counterpart(0, D, rng) for _ in range(50)]
    # majority should be 1
    assert picks.count(1) > 40
    # low tau => more diverse
    strat_low = get_pairing("dynamic", tau=0.1)
    rng2 = np.random.RandomState(0)
    picks2 = [strat_low.select_counterpart(0, D, rng2) for _ in range(100)]
    # should have both 1 and 2 appearing
    assert 1 in picks2 and 2 in picks2

def test_dynamic_never_self_with_uniform_D():
    D = np.zeros((4,4))
    strat = get_pairing("dynamic", tau=5.0)
    rng = np.random.RandomState(0)
    for i in range(4):
        for _ in range(10):
            j = strat.select_counterpart(i, D, rng)
            assert j != i
