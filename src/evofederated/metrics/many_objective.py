"""Métricas many-objective: HV, IGD/IGD+, spread, nondominated, knee, parallel coords data."""
import numpy as np
from typing import List, Dict
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
try:
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    HAS_IGD = True
except ImportError:
    HAS_IGD = False


def compute_hv(F: np.ndarray, ref_point: np.ndarray) -> float:
    if F is None or len(F) == 0:
        return 0.0
    # exact HV may be heavy for N>10 ; pymoo uses WFG; for N=32 still works but slow.
    # Fallback Monte Carlo if too heavy (n_obj > 10 and len>50)
    n_obj = F.shape[1] if len(F.shape) >1 else 1
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    pf = F[nds] if len(nds) >0 else F
    if len(pf) == 0:
        return 0.0
    # For many-objective, if very heavy try approximation: if n_obj>10 and len(pf)>100, sample
    try:
        hv = HV(ref_point=np.array(ref_point, dtype=float))
        # pymoo HV for n_obj up to ~10 exact; beyond may be slower but we attempt
        # Limit PF size for HV to 200 to avoid blowup
        if len(pf) > 200:
            # keep random subset
            rng = np.random.RandomState(0)
            idx = rng.choice(len(pf), 200, replace=False)
            pf = pf[idx]
        val = hv.do(pf)
        return float(val) if not np.isnan(val) else 0.0
    except Exception as e:
        # Monte Carlo approximation fallback
        try:
            return monte_carlo_hv(pf, ref_point, n_samples=10000)
        except Exception:
            return 0.0


def monte_carlo_hv(pf: np.ndarray, ref: np.ndarray, n_samples: int = 5000) -> float:
    """Simple MC HV: sample uniformly in [0, ref] and count dominated."""
    if pf is None or len(pf)==0:
        return 0.0
    # pf in minimization [0,1], ref 1.1, volume total = prod(ref)
    pf = np.array(pf)
    ref = np.array(ref)
    n_obj = len(ref)
    # ideal 0
    lo = np.zeros(n_obj)
    vol = np.prod(ref - lo)
    rng = np.random.RandomState(1)
    samples = rng.rand(n_samples, n_obj) * (ref - lo) + lo  # uniform in [0, ref]
    # dominated if exists pf point <= sample in all objs
    dominated = 0
    for s in samples:
        # if any pf dominates s (pf <= s in all dims)
        if np.any(np.all(pf <= s, axis=1)):
            dominated += 1
    return float(dominated / n_samples * vol)


def compute_igd(pf: np.ndarray, ref_pf: np.ndarray, plus: bool = False) -> float:
    if pf is None or ref_pf is None or len(pf)==0 or len(ref_pf)==0:
        return float("nan")
    try:
        if plus and HAS_IGD:
            ind = IGDPlus(ref_pf)
        elif HAS_IGD:
            ind = IGD(ref_pf)
        else:
            return float("nan")
        return float(ind.do(pf))
    except Exception:
        return float("nan")


def get_nondominated(F: np.ndarray):
    if F is None or len(F)==0:
        return np.array([], dtype=bool), np.array([])
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    mask = np.zeros(len(F), dtype=bool)
    mask[nds] = True
    return mask, F[nds]


def count_nondominated(F: np.ndarray) -> int:
    mask,_ = get_nondominated(F)
    return int(mask.sum())


def compute_spread(F: np.ndarray) -> float:
    """Delta spread (Deb) simplified for 2 objectives; for many use std of distances."""
    if F is None or len(F) < 2:
        return 0.0
    # For many-objective, compute average distance to nearest neighbor normalized
    from scipy.spatial.distance import cdist
    try:
        dists = cdist(F, F)
        np.fill_diagonal(dists, np.inf)
        nn = dists.min(axis=1)
        return float(nn.mean() / (nn.std() + 1e-9)) if nn.std()>0 else float(nn.mean())
    except Exception:
        return 0.0


def coverage_metric(F: np.ndarray, ref: np.ndarray = None) -> dict:
    """Cobertura simple: range per objective y volumen."""
    if F is None or len(F)==0:
        return {}
    mins = F.min(axis=0).tolist()
    maxs = F.max(axis=0).tolist()
    ranges = (F.max(axis=0) - F.min(axis=0)).tolist()
    return {"mins": mins, "maxs": maxs, "ranges": ranges, "mean_range": float(np.mean(ranges))}


def find_knee_points(F: np.ndarray, method: str = "tradeoff") -> List[int]:
    """
    Identifica knee points sencillos y defendibles.
    - Para 2D: max distance to line ideal-nadir.
    - Para N>2: sort by mean objective (compromise) + max tradeoff (min max regret).
    Retorna índices en F.
    """
    if F is None or len(F)==0:
        return []
    # filter nondominated only
    mask, pf = get_nondominated(F)
    indices = np.where(mask)[0]
    if len(pf)==0:
        return []
    if len(pf)==1:
        return [int(indices[0])]
    # Normalize pf to [0,1] for fairness
    f_min = pf.min(axis=0)
    f_max = pf.max(axis=0)
    denom = (f_max - f_min).clip(min=1e-9)
    pf_norm = (pf - f_min) / denom
    # distance to ideal (0) weighted: compromise = smallest L2 distance to ideal (balanced)
    # But also more principled: knee = max distance to hyperplane? Use bend angle approximation.
    # Simple: compute distance to line for 2D, else to hyperplane defined by extremes (convex).
    # For many-objective, approximate knee as point with minimal sum (most balanced) or max marginal tradeoff.
    if pf_norm.shape[1]==2:
        # line from (0,1) to (1,0)? Actually ideal (0,0) to nadir (1,1) diagonal? Use ideal (0,0) to worst (1,1)
        # but more common is line between extremes; we use distance to diagonal.
        # Simplified: max distance to line x=y? Not.
        # Use max distance to convex hull line connecting extremes: farthest from line between min per obj points.
        # For 2D, knee is point farthest from line connecting (min f0) and (min f1) points.
        # Find extreme points: idx min f0, idx min f1
        idx_min0 = np.argmin(pf_norm[:,0])
        idx_min1 = np.argmin(pf_norm[:,1])
        p0 = pf_norm[idx_min0]
        p1 = pf_norm[idx_min1]
        # line p0->p1
        line_vec = p1 - p0
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-9:
            dists = np.linalg.norm(pf_norm - p0, axis=1)
        else:
            # distance = |(p - p0) cross line| / len
            cross = np.abs((pf_norm[:,0]-p0[0])*line_vec[1] - (pf_norm[:,1]-p0[1])*line_vec[0])
            dists = cross / line_len
        knee_rel = int(np.argmax(dists))
        return [int(indices[knee_rel])]
    else:
        # Many: knee approx = point closest to ideal in L2 (best balanced) and also far from extremes
        # Also compute max tradeoff: sort by weighted distance
        # Return 2-3 candidates: best balanced, best median
        l2 = np.linalg.norm(pf_norm, axis=1)
        idx_best_balanced = int(np.argmin(l2))
        # second: minimal max objective (minimize worst) - Chebyshev
        cheby = pf_norm.max(axis=1)
        idx_min_cheby = int(np.argmin(cheby))
        # third: minimal sum
        sums = pf_norm.sum(axis=1)
        idx_min_sum = int(np.argmin(sums))
        # deduplicate
        chosen = []
        for idx in [idx_best_balanced, idx_min_cheby, idx_min_sum]:
            g_idx = int(indices[idx])
            if g_idx not in chosen:
                chosen.append(g_idx)
        return chosen[:3]


def objective_statistics(F: np.ndarray, accuracies: np.ndarray = None) -> dict:
    """
    Estadísticas agregadas aunque algoritmo sea many-objective:
    mean/median/worst/best/std/gap etc desde accuracies
    F = 1 - acc, so acc = 1 - F
    """
    if F is None or len(F)==0:
        return {}
    # Convert back to accuracy if needed
    if accuracies is None:
        accuracies = 1.0 - F  # per individual per objective
        # need per individual aggregated? Use mean across objectives per individual then stats across population?
        # For per-generation stats we compute over PF or whole pop? Use pf
        mask, pf = get_nondominated(F)
        if len(pf)==0:
            pf = F
        acc_pf = 1 - pf
        # mean per individual then across PF
        ind_means = acc_pf.mean(axis=1)
        return {
            "mean_client_accuracy": float(np.mean(acc_pf)),  # overall mean across objectives and individuals
            "median_client_accuracy": float(np.median(acc_pf)),
            "mean_individual_accuracy": float(np.mean(ind_means)),
            "median_individual_accuracy": float(np.median(ind_means)),
            "worst_client_accuracy": float(np.min(acc_pf)),
            "best_client_accuracy": float(np.max(acc_pf)),
            "std_between_clients": float(acc_pf.std()),
            "gap": float(np.max(acc_pf) - np.min(acc_pf)),
            "n_nd": int(mask.sum()),
        }
    else:
        return {}
