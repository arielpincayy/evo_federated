"""Ganancia de transferencia intra- vs inter-cluster sobre la matriz A[s, j] (modelo entrenado en s, evaluado en j)."""
import numpy as np


def transfer_gap(A: np.ndarray, labels: np.ndarray) -> float:
    """media_s( media_{j en cluster(s), j!=s} A[s,j] - media_{j fuera} A[s,j] ). Ignora fuentes sin vecinos intra o inter."""
    same = labels[:, None] == labels[None, :]
    off = ~np.eye(len(labels), dtype=bool)
    ins, out = same & off, ~same
    ok = ins.any(1) & out.any(1)
    a_in = np.where(ins, A, 0).sum(1)[ok] / ins.sum(1)[ok]
    a_out = np.where(out, A, 0).sum(1)[ok] / out.sum(1)[ok]
    return float((a_in - a_out).mean())


def permutation_p(A: np.ndarray, labels: np.ndarray, n_perm: int = 10_000, seed: int = 0) -> float:
    """p unilateral: P(gap con etiquetas permutadas >= gap observado). Conserva los tamaños de cluster."""
    A = np.nan_to_num(A)
    rng = np.random.RandomState(seed)
    obs = transfer_gap(A, labels)
    perm = np.array([transfer_gap(A, rng.permutation(labels)) for _ in range(n_perm)])
    return float((1 + (perm >= obs).sum()) / (1 + n_perm))
