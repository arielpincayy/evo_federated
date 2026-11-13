import numpy as np

from evofederated.metrics.transfer import permutation_p, transfer_gap


def _block(n=40, c=4, hi=0.8, lo=0.3):
    lab = np.arange(n) % c
    A = np.where(lab[:, None] == lab[None, :], hi, lo).astype(float)
    np.fill_diagonal(A, np.nan)
    return np.nan_to_num(A), lab


def test_block_matrix_has_positive_gap_and_small_p():
    A, lab = _block()
    assert abs(transfer_gap(A, lab) - 0.5) < 1e-9
    assert permutation_p(A, lab, n_perm=200) < 0.02


def test_constant_matrix_has_zero_gap():
    lab = np.arange(20) % 4
    A = np.full((20, 20), 0.5)
    assert transfer_gap(A, lab) == 0.0
