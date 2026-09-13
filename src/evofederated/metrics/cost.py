"""Cost metrics."""
import numpy as np


def compute_saving(e_method: int, e_full: int) -> float:
    if e_full == 0:
        return 0.0
    return 1.0 - (e_method / e_full)


def cost_metrics(eval_count: int, n_generations: int, pop_size: int, n_clients: int, strategy: str, characterization_cost: int = 0):
    """
    Compute saving vs FULL.
    FULL approx R*G*N (+ char)
    Proposed approx 2*R*G (+ char + controls)
    """
    e_full = pop_size * n_generations * n_clients
    e_method = eval_count  # actual including char? We'll add char externally
    saving = compute_saving(e_method, e_full)
    return {
        "eval_count": eval_count,
        "e_full_theoretical": e_full,
        "saving_vs_full": saving,
        "saving_percent": saving * 100,
        "evals_per_gen": eval_count / max(1, n_generations),
    }
