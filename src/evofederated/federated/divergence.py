"""Matriz de divergencia entre vectores de aprendizaje."""
import numpy as np
from typing import Dict


def cosine_distance(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        # if both near zero, distance 0 (identical no-update); if one zero, max distance 1
        if nu < eps and nv < eps:
            return 0.0
        return 1.0
    cos = np.dot(u, v) / (nu * nv + eps)
    cos = np.clip(cos, -1.0, 1.0)
    return 1.0 - cos


def l2_normalized_distance(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        if nu < eps and nv < eps:
            return 0.0
        # distance to zero vector
        # normalize non-zero, other is zero => distance sqrt(2)?? but normalized L2 between unit vectors
        # We'll return 1.0 for simplicity
        return 1.0
    un = u / (nu + eps)
    vn = v / (nv + eps)
    return float(np.linalg.norm(un - vn))


def compute_divergence_matrix(
    deltas: Dict[int, np.ndarray],
    metric: str = "cosine",
    eps: float = 1e-9,
) -> np.ndarray:
    """
    deltas: dict client_id -> vector shape P
    Returns matrix D shape N x N.
    Ordering by sorted client ids.
    """
    cids = sorted(deltas.keys())
    n = len(cids)
    mat = np.zeros((n, n), dtype=float)
    for i, cid_i in enumerate(cids):
        for j, cid_j in enumerate(cids):
            if i == j:
                mat[i, j] = 0.0
            elif j < i:
                mat[i, j] = mat[j, i]
            else:
                u = deltas[cid_i]
                v = deltas[cid_j]
                if metric == "cosine":
                    d = cosine_distance(u, v, eps)
                elif metric == "l2_norm":
                    d = l2_normalized_distance(u, v, eps)
                else:
                    raise ValueError(f"Unknown metric {metric}")
                mat[i, j] = d
    # Ensure symmetry
    # Return matrix + mapping
    return mat


def divergence_stats(D: np.ndarray):
    n = D.shape[0]
    triu = D[np.triu_indices(n, k=1)]
    return {
        "mean": float(triu.mean()) if len(triu) else 0.0,
        "std": float(triu.std()) if len(triu) else 0.0,
        "min": float(triu.min()) if len(triu) else 0.0,
        "max": float(triu.max()) if len(triu) else 0.0,
        "median": float(np.median(triu)) if len(triu) else 0.0,
    }


def most_divergent_pairs(D: np.ndarray, top_k: int = 10):
    n = D.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, D[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]
