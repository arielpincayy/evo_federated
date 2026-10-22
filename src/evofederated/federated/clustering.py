"""Agrupación de nodos por similitud de sus vectores Δ (probe) y baselines de agrupación."""
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def _unit_rows(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-12)


def cluster_delta(delta: np.ndarray, n_clusters: int, method: str = "kmeans", seed: int = 0) -> np.ndarray:
    """Clusters por similitud coseno de Δ (filas = nodos ordenados). Devuelve etiquetas 0..C-1, todas no vacías."""
    n = delta.shape[0]
    c = min(n_clusters, n)
    u = _unit_rows(delta)
    if method == "kmeans":  # k-means esférico: k-means sobre filas normalizadas
        return KMeans(n_clusters=c, n_init=10, random_state=seed).fit_predict(u)
    if method == "average":
        from scipy.cluster.hierarchy import fcluster, linkage
        z = linkage(u, method="average", metric="cosine")
        lab = fcluster(z, t=c, criterion="maxclust") - 1
        if len(np.unique(lab)) != c:  # maxclust puede devolver menos por empates
            raise ValueError(f"average linkage produjo {len(np.unique(lab))} clusters, se pidieron {c}")
        return lab
    raise ValueError(f"método desconocido: {method}")


def cluster_labels_oracle(class_dist: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Baseline con información de etiquetas (no privado): k-means sobre la distribución de clases."""
    p = class_dist / np.maximum(class_dist.sum(axis=1, keepdims=True), 1)
    return KMeans(n_clusters=min(n_clusters, len(p)), n_init=10, random_state=seed).fit_predict(p)


def cluster_random(n: int, n_clusters: int, seed: int = 0) -> np.ndarray:
    """Particiona nodos al azar en C grupos no vacíos de tamaño casi igual."""
    rng = np.random.RandomState(seed)
    lab = np.arange(n) % min(n_clusters, n)
    rng.shuffle(lab)
    return lab


def cluster_quality(labels: np.ndarray, class_dist: np.ndarray, oracle: Optional[np.ndarray] = None) -> Dict[str, float]:
    """eta2: fracción de la varianza de la distribución de clases explicada por los clusters; ARI vs oráculo."""
    p = class_dist / np.maximum(class_dist.sum(axis=1, keepdims=True), 1)
    total = ((p - p.mean(axis=0)) ** 2).sum()
    within = sum(((p[labels == c] - p[labels == c].mean(axis=0)) ** 2).sum() for c in np.unique(labels))
    out = {"n_clusters": int(len(np.unique(labels))), "eta2_label_dist": float(1 - within / total) if total > 0 else 0.0}
    if oracle is not None:
        out["ari_vs_label_clusters"] = float(adjusted_rand_score(oracle, labels))
    return out
