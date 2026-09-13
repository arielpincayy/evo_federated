import numpy as np
from evofederated.federated.divergence import compute_divergence_matrix, cosine_distance, l2_normalized_distance

def test_cosine_properties():
    a = np.array([1.0,0.0])
    b = np.array([0.0,1.0])
    assert np.isclose(cosine_distance(a,b),1.0)  # orthogonal => 1
    assert np.isclose(cosine_distance(a,a),0.0)
    assert np.isclose(cosine_distance(a,-a),2.0)

def test_zero_vector_handling():
    a = np.zeros(5)
    b = np.random.randn(5)
    # both zero => 0
    assert cosine_distance(a,a)==0.0
    # one zero => 1
    assert cosine_distance(a,b)==1.0
    assert l2_normalized_distance(a,a)==0.0
    assert l2_normalized_distance(a,b)==1.0

def test_divergence_matrix_properties():
    deltas = {0: np.array([1.0,0]), 1: np.array([0.0,1.0]), 2: np.array([1.0,0])}
    D = compute_divergence_matrix(deltas, metric="cosine")
    # symmetric
    assert np.allclose(D, D.T)
    # diagonal zero
    assert np.allclose(np.diag(D), 0.0)
    # D[0,1] should be 1 (orthogonal)
    assert np.isclose(D[0,1],1.0)
    assert np.isclose(D[0,2],0.0)

def test_l2_normalized():
    deltas = {0: np.array([1,0.]), 1: np.array([2,0.])}
    D = compute_divergence_matrix(deltas, metric="l2_norm")
    # same direction, normalized equal => 0
    assert np.isclose(D[0,1],0.0, atol=1e-6)
