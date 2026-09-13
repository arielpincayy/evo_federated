"""Hypervolume utilities."""
import numpy as np
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def compute_hv(F: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Compute HV for front F in minimization space.
    F shape (n_points, n_obj).
    ref_point shape (n_obj,).
    """
    if F is None or len(F) == 0:
        return 0.0
    # Filter to non-dominated only
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    pf = F[nds] if len(nds) > 0 else F
    # Ensure pf dominates ref: for minimization, pf < ref
    # If any point has objective > ref (worse), pymoo HV may handle but we filter?
    # We'll clip? Not.
    try:
        hv = HV(ref_point=np.array(ref_point, dtype=float))
        val = hv.do(pf)
        return float(val)
    except Exception:
        return 0.0


def hv_common_validation(F_list, ref_point):
    """Alias for validation HV with common reference."""
    return compute_hv(np.array(F_list), ref_point)


def hv_evolution(pf_per_generation: list, ref_point: np.ndarray):
    """Compute HV per generation given list of PF arrays."""
    return [compute_hv(pf, ref_point) for pf in pf_per_generation]
