"""Similarity utilities for learning-update vectors Δ.

Computes cosine similarity matrix between clients' update vectors.
Main signal comes from model updates, not dataset statistics.
"""
from typing import Dict
import numpy as np


def cosine_similarity(u: np.ndarray, v: np.ndarray, eps: float = 1e-9) -> float:
    """Cosine similarity in [-1, 1].

    Handles zero vectors:
      both zero -> 1.0 (identical, no update)
      one zero -> 0.0 (neutral, not similar nor opposite)
    """
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps and nv < eps:
        return 1.0
    if nu < eps or nv < eps:
        return 0.0
    cos = float(np.dot(u, v) / (nu * nv + eps))
    return float(np.clip(cos, -1.0, 1.0))


def compute_similarity_matrix(
    deltas: Dict[int, np.ndarray],
    eps: float = 1e-9,
) -> np.ndarray:
    """Compute S[i,j] = cosine_similarity(Δ_i, Δ_j).

    Ordering by sorted client ids.
    Diagonal is 1.0 (self-similarity).
    """
    cids = sorted(deltas.keys())
    n = len(cids)
    S = np.zeros((n, n), dtype=float)
    for i, ci in enumerate(cids):
        for j, cj in enumerate(cids):
            if i == j:
                S[i, j] = 1.0
            elif j < i:
                S[i, j] = S[j, i]
            else:
                S[i, j] = cosine_similarity(deltas[ci], deltas[cj], eps=eps)
    return S


def compute_distance_matrix_from_similarity(S: np.ndarray) -> np.ndarray:
    """Distance = 1 - similarity (range 0..2). Diagonal 0."""
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D


def similarity_stats(S: np.ndarray) -> dict:
    """Stats over upper triangle (excluding diagonal)."""
    n = S.shape[0]
    if n < 2:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    triu = S[np.triu_indices(n, k=1)]
    return {
        "mean": float(triu.mean()),
        "std": float(triu.std()),
        "min": float(triu.min()),
        "max": float(triu.max()),
        "median": float(np.median(triu)),
    }
