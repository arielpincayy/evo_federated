"""Agrupación de nodos por similitud de sus vectores Δ (probe) y baselines de agrupación."""
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


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


def choose_k(delta: np.ndarray, k_max: int = 10, seed: int = 0):
    """k en 2..k_max con mayor silueta coseno de k-means esférico sobre Δ (no supervisado). Devuelve (k, {k: silueta})."""
    u = _unit_rows(delta)
    sil = {}
    for k in range(2, min(k_max, len(u) - 1) + 1):
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(u)
        if len(np.unique(lab)) == k:
            sil[k] = float(silhouette_score(u, lab, metric="cosine"))
    return max(sil, key=sil.get), sil


def cluster_dbscan(z: np.ndarray, q: float, min_samples: int = 5):
    """DBSCAN con distancia coseno. eps = cuantil `q` de la distancia al `min_samples`-ésimo vecino (sin mirar etiquetas).

    El ruido (-1) se asigna al cluster de centroide más cercano por coseno, para que todo nodo tenga cluster.
    Devuelve (etiquetas 0..k-1, nº de nodos de ruido). Si no hay ningún cluster, todo es ruido y devuelve ceros (k=1: degenerado).
    """
    u = _unit_rows(z)
    d = np.clip(1.0 - u @ u.T, 0.0, None)
    eps = float(np.quantile(np.sort(d, axis=1)[:, min_samples], q))  # columna 0 es el propio nodo
    lab = DBSCAN(eps=max(eps, 1e-12), min_samples=min_samples, metric="precomputed").fit_predict(d)
    noise = lab < 0
    if noise.all():
        return np.zeros(len(u), dtype=int), int(noise.sum())
    cent = _unit_rows(np.stack([u[lab == c].mean(axis=0) for c in range(lab.max() + 1)]))
    lab[noise] = np.argmax(u[noise] @ cent.T, axis=1)
    return lab, int(noise.sum())


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
