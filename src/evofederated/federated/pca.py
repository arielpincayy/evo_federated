"""PCA sobre updates Δ_i antes de cosine similarity (A) — independiente de PCA visualización (B)."""
import time
from typing import Dict, Tuple, Any
import numpy as np


def _stack_deltas(deltas: Dict[int, np.ndarray]) -> Tuple[np.ndarray, list]:
    cids = sorted(deltas.keys())
    mat = np.stack([deltas[c] for c in cids], axis=0)  # N x D
    return mat, cids


def apply_pca(
    deltas: Dict[int, np.ndarray],
    mode: str = "none",
    variance: float = 0.95,
    n_components: int = 4,
    seed: int = 42,
) -> Tuple[Dict[int, np.ndarray], dict]:
    """
    Aplica PCA a matriz X (N x D) donde cada fila es Δ_i.
    mode = "none" | "variance" | "fixed"
    - variance: n_components = variance (float 0..1) explicado
    - fixed:    n_components = int fijo
    Retorna dict reducidos {cid: z_i} y info dict.
    PCA se ajusta únicamente sobre updates de esa ejecución (no leakage).
    """
    t0 = time.time()
    X, cids = _stack_deltas(deltas)  # N x D
    N, D = X.shape
    max_components = min(N - 1, D) if N > 1 else 0

    info = {
        "mode": mode,
        "N": int(N),
        "D_original": int(D),
        "D_reduced": int(D),
        "variance_target": None,
        "n_components_requested": None,
        "n_components_actual": None,
        "explained_variance_ratio": [],
        "explained_variance_total": 1.0,
        "pca_time": 0.0,
        "note": "",
        "max_valid": int(max_components),
    }

    if mode == "none" or mode is None or not mode:
        info["mode"] = "none"
        info["D_reduced"] = int(D)
        info["n_components_actual"] = int(D)
        info["pca_time"] = time.time() - t0
        return deltas, info

    if max_components <= 0:
        # N=1 no se puede PCA
        info["note"] = "N<=1 no PCA posible, retorno raw"
        info["pca_time"] = time.time() - t0
        return deltas, info

    try:
        from sklearn.decomposition import PCA
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    if mode == "variance":
        target = float(variance)
        info["variance_target"] = target
        info["n_components_requested"] = target
        if not has_sklearn:
            # fallback SVD manually variance based
            U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
            eig = (S ** 2) / (N - 1)
            total = eig.sum()
            ratios = eig / total if total > 0 else np.zeros_like(eig)
            cumsum = np.cumsum(ratios)
            d = int(np.searchsorted(cumsum, target) + 1)
            d = max(1, min(d, max_components))
            Z = (U[:, :d] * S[:d])
            reduced = {cid: Z[i] for i, cid in enumerate(cids)}
            info["D_reduced"] = int(d)
            info["n_components_actual"] = int(d)
            info["explained_variance_ratio"] = ratios[:d].tolist()
            info["explained_variance_total"] = float(cumsum[d - 1]) if d <= len(cumsum) else 1.0
            info["pca_time"] = time.time() - t0
            return reduced, info
        else:
            # sklearn PCA with variance retains min components covering variance
            # limit to max_components
            pca = PCA(n_components=target, svd_solver="full", random_state=seed)
            Z = pca.fit_transform(X)  # N x d
            d = Z.shape[1]
            # cap if needed (sklearn already limited to min(N,D))
            if d > max_components:
                Z = Z[:, :max_components]
                d = max_components
            reduced = {cid: Z[i] for i, cid in enumerate(cids)}
            info["D_reduced"] = int(d)
            info["n_components_actual"] = int(d)
            info["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist() if hasattr(pca, "explained_variance_ratio_") else []
            info["explained_variance_total"] = float(np.sum(pca.explained_variance_ratio_)) if hasattr(pca, "explained_variance_ratio_") else 1.0
            info["pca_time"] = time.time() - t0
            return reduced, info

    elif mode == "fixed":
        req = int(n_components)
        info["n_components_requested"] = req
        # adapt to valid
        d = min(req, max_components)
        if req != d:
            info["note"] = f"requested {req} > max_valid {max_components}, adapted to {d}"
        else:
            info["note"] = ""
        if not has_sklearn:
            U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
            Z = (U[:, :d] * S[:d])
            eig = (S ** 2) / (N - 1)
            total = eig.sum()
            ratios = eig / total if total > 0 else np.zeros_like(eig)
            reduced = {cid: Z[i] for i, cid in enumerate(cids)}
            info["D_reduced"] = int(d)
            info["n_components_actual"] = int(d)
            info["explained_variance_ratio"] = ratios[:d].tolist()
            info["explained_variance_total"] = float(np.sum(ratios[:d]))
            info["pca_time"] = time.time() - t0
            return reduced, info
        else:
            pca = PCA(n_components=d, random_state=seed)
            Z = pca.fit_transform(X)
            reduced = {cid: Z[i] for i, cid in enumerate(cids)}
            info["D_reduced"] = int(Z.shape[1])
            info["n_components_actual"] = int(Z.shape[1])
            info["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist() if hasattr(pca, "explained_variance_ratio_") else []
            info["explained_variance_total"] = float(np.sum(pca.explained_variance_ratio_)) if hasattr(pca, "explained_variance_ratio_") else 1.0
            info["pca_time"] = time.time() - t0
            return reduced, info
    else:
        raise ValueError(f"unknown PCA mode: {mode}")


def compute_similarity_preservation(S_raw: np.ndarray, S_pca: np.ndarray) -> dict:
    """Calcula métricas preservación entre S_raw y S_pca (solo off-diagonal)."""
    n = S_raw.shape[0]
    if n < 2 or S_raw.shape != S_pca.shape:
        return {}
    idx = np.triu_indices(n, k=1)
    a = S_raw[idx].ravel()
    b = S_pca[idx].ravel()
    # Pearson
    try:
        pearson = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 1e-12 and np.std(b) > 1e-12 else 0.0
    except Exception:
        pearson = 0.0
    # Spearman/Kendall
    try:
        from scipy.stats import spearmanr, kendalltau
        spearman = float(spearmanr(a, b).correlation) if len(a) > 1 else 0.0
        kendall = float(kendalltau(a, b).correlation) if len(a) > 1 else 0.0
        if np.isnan(spearman): spearman = 0.0
        if np.isnan(kendall): kendall = 0.0
    except Exception:
        spearman = kendall = 0.0
    mae = float(np.mean(np.abs(a - b)))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    # ranking preservation: for each node, top-1 neighbour preserved?
    nearest_raw = np.argmax(S_raw - np.eye(n) * 10, axis=1)
    nearest_pca = np.argmax(S_pca - np.eye(n) * 10, axis=1)
    nn_preserved = float(np.mean(nearest_raw == nearest_pca))
    # pairing preservation: max matching pairs equal?
    try:
        from .matching import max_similarity_matching
        pairs_raw, _ = max_similarity_matching(S_raw)
        pairs_pca, _ = max_similarity_matching(S_pca)
        set_raw = set(tuple(sorted(p)) for p in pairs_raw)
        set_pca = set(tuple(sorted(p)) for p in pairs_pca)
        pair_preserved = float(len(set_raw & set_pca) / max(1, len(set_raw)))
    except Exception:
        pair_preserved = 0.0
    return {
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "mae": mae,
        "rmse": rmse,
        "nn_preserved": nn_preserved,
        "pairing_preserved": pair_preserved,
    }
