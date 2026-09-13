import numpy as np
from evofederated.metrics.hypervolume import compute_hv

def test_hv_known_values():
    # Simple 2D minimization case
    # Points: (0.2,0.8), (0.5,0.5), (0.8,0.2) with ref (1.1,1.1)
    # Known HV? We'll just test monotonicity and basic
    F = np.array([[0.2,0.8],[0.5,0.5],[0.8,0.2]])
    ref = np.array([1.1,1.1])
    hv = compute_hv(F, ref)
    assert hv > 0
    # Adding a dominated point should not increase HV
    F2 = np.array([[0.2,0.8],[0.5,0.5],[0.8,0.2],[0.6,0.6]])
    hv2 = compute_hv(F2, ref)
    assert np.isclose(hv, hv2) # dominated point doesn't affect

    # Better front should have larger HV
    F_worse = np.array([[0.6,0.9],[0.9,0.6]])
    hv_worse = compute_hv(F_worse, ref)
    assert hv > hv_worse

def test_hv_empty():
    assert compute_hv(np.array([]).reshape(0,2), np.array([0.1,0.1])) == 0.0
    assert compute_hv(None, np.array([0.1,0.1])) == 0.0

def test_hv_minimization_orientation():
    # For our NAS: objectives are -F1 in [-1,0], ref (0.1,0.1)
    # Better solutions have more negative values => closer to -1
    F_good = np.array([[-0.9,-0.9],[-0.8,-0.7]])
    F_bad = np.array([[-0.2,-0.3]])
    ref = np.array([0.1,0.1])
    hv_good = compute_hv(F_good, ref)
    hv_bad = compute_hv(F_bad, ref)
    assert hv_good > hv_bad

def test_hv_ref_inclusive():
    # ref = (0,0) vs (0.1,0.1) should give different but positive
    F = np.array([[-0.5,-0.5]])
    hv1 = compute_hv(F, np.array([0.0,0.0]))
    hv2 = compute_hv(F, np.array([0.1,0.1]))
    assert hv1>0 and hv2>0
    assert hv2>hv1
