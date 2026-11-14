"""Interacción arquitectura x nodo sobre matrices Y[réplica, arquitectura, nodo] (accuracy o logit). Ver docs/ARCH_NODE_DESIGN.md."""
import numpy as np


def double_center(Y: np.ndarray) -> np.ndarray:
    return Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + Y.mean()


def _pairs(n_r: int):
    return [(r, q) for r in range(n_r) for q in range(n_r) if r != q]


def stable_interaction(Yv: np.ndarray, Yt: np.ndarray) -> float:
    """V = media de R_val[r] * R_test[r'] (r != r'): varianza de la interacción que se repite con otro entrenamiento y otra muestra del nodo."""
    Rv, Rt = (np.stack([double_center(y) for y in Y]) for Y in (Yv, Yt))
    return float(np.mean([(Rv[r] * Rt[q]).mean() for r, q in _pairs(len(Rv))]))


def perm_p(Yv: np.ndarray, Yt: np.ndarray, n_perm: int = 10_000, seed: int = 0) -> float:
    """p unilateral (V > 0): permuta los nodos de R_test (rompe la correspondencia nodo a nodo)."""
    Rv, Rt = (np.stack([double_center(y) for y in Y]) for Y in (Yv, Yt))
    pairs, rng = _pairs(len(Rv)), np.random.RandomState(seed)
    obs = np.mean([(Rv[r] * Rt[q]).mean() for r, q in pairs])
    n = Rv.shape[2]
    perm = [np.mean([(Rv[r] * Rt[q][:, p]).mean() for r, q in pairs]) for p in (rng.permutation(n) for _ in range(n_perm))]
    return float((1 + (np.array(perm) >= obs).sum()) / (1 + n_perm))


def effects_sd(Yv: np.ndarray, Yt: np.ndarray):
    """SD del efecto de arquitectura (medias por arquitectura) y del de nodo (medias por nodo), con todo promediado."""
    M = (Yv.mean(0) + Yt.mean(0)) / 2
    return float(M.mean(1).std()), float(M.mean(0).std())


def resid_var(Yv: np.ndarray, Yt: np.ndarray) -> float:
    return float(np.mean([(double_center(y) ** 2).mean() for Y in (Yv, Yt) for y in Y]))


def selection_gain(Yv: np.ndarray, Yt: np.ndarray, labels: np.ndarray):
    """(ganancia por nodo, ganancia por cluster) en las unidades de Y: elegir con val de la réplica r, evaluar en test de r' != r,
    frente a la arquitectura global elegida con todos los nodos."""
    cols, gn, gc = np.arange(Yv.shape[2]), [], []
    for r, q in _pairs(len(Yv)):
        glob = Yt[q][Yv[r].mean(1).argmax()].mean()
        node = Yt[q][Yv[r].argmax(0), cols].mean()
        a_cl = np.empty(len(cols), dtype=int)
        for c in np.unique(labels):
            m = labels == c
            a_cl[m] = Yv[r][:, m].mean(1).argmax()
        gn.append(node - glob)
        gc.append(Yt[q][a_cl, cols].mean() - glob)
    return float(np.mean(gn)), float(np.mean(gc))


def boot_ci(stat, n_nodes: int, n_boot: int = 2000, seed: int = 0):
    """IC95 por bootstrap de nodos; `stat(idx)` recalcula el estadístico con los nodos remuestreados."""
    rng = np.random.RandomState(seed)
    return tuple(np.percentile([stat(rng.randint(0, n_nodes, n_nodes)) for _ in range(n_boot)], [2.5, 97.5]))
