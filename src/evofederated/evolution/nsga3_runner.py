"""NSGA-III runner con reference directions, normalization, niching."""
import time
import numpy as np
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from pymoo.core.callback import Callback

from .many_objective import ManyObjectiveProblem, ManyObjectiveEvaluator


def get_nsga3_ref_dirs(n_obj, pop_size_hint=20, seed=42):
    """
    Retorna ref_dirs para NSGA-III.
    Para NSGA-III pop_size típicamente = n_dirs (o >= n_dirs).
    Decisión documentada similar a MOEA/D.
    """
    if n_obj <= 3:
        for p in [12,8,6,4,3,2,1]:
            from math import comb
            n_dirs = comb(p + n_obj -1, n_obj-1)
            if n_dirs <= pop_size_hint*1.5 and n_dirs >= pop_size_hint*0.5:
                return get_reference_directions("das-dennis", n_obj, n_partitions=p), p, n_dirs
        return get_reference_directions("das-dennis", n_obj, n_partitions=4), 4, 0
    elif n_obj <= 5:
        for p in [6,5,4,3,2,1]:
            from math import comb
            n_dirs = comb(p + n_obj -1, n_obj-1)
            if n_dirs <= pop_size_hint*2:
                return get_reference_directions("das-dennis", n_obj, n_partitions=p), p, n_dirs
        return get_reference_directions("das-dennis", n_obj, n_partitions=2),2,0
    else:
        from math import comb
        n_dirs_p1 = n_obj
        n_dirs_p2 = comb(n_obj + 1, n_obj -1) if n_obj <= 12 else 1e9
        # Energia: para N=8, p2=36; for N=16 p2=136, might be large but feasible
        if pop_size_hint >= n_dirs_p2 and n_obj <= 10:
            return get_reference_directions("das-dennis", n_obj, n_partitions=2),2,n_dirs_p2
        else:
            return get_reference_directions("das-dennis", n_obj, n_partitions=1),1,n_dirs_p1


def run_nsga3(
    evaluator: ManyObjectiveEvaluator,
    n_var: int,
    pop_size: int = 20,
    n_generations: int = 10,
    seed: int = 42,
    hv_ref: np.ndarray = None,
    verbose: bool = False,
):
    n_obj = evaluator.n_obj
    ref_dirs, n_partitions, n_dirs = get_nsga3_ref_dirs(n_obj, pop_size_hint=pop_size, seed=seed)
    actual_pop = len(ref_dirs)
    # NSGA-III pop_size can be larger than n_dirs, but keep equal for simplicity
    # If hint differs, adapt
    if actual_pop != pop_size:
        print(f"[NSGA-III] N={n_obj} pop_hint {pop_size} -> adapted pop {actual_pop} (p={n_partitions})")
        pop_size = actual_pop

    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(eta=30, prob=1.0),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    problem = ManyObjectiveProblem(n_var=n_var, evaluator=evaluator)

    if hv_ref is None:
        hv_ref = np.ones(n_obj) * 1.1

    hv_history = []
    start_time = time.time()

    class NSGA3Callback(Callback):
        def notify(self, algorithm):
            cur_gen = int(algorithm.n_gen)
            evaluator.set_generation(cur_gen)
            F = algorithm.pop.get("F")
            if F is not None and len(F) > 0:
                try:
                    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
                    pf = F[nds]
                except Exception:
                    pf = F
                try:
                    hv = HV(ref_point=hv_ref)
                    hv_val = hv.do(pf) if len(pf) else 0.0
                except Exception:
                    hv_val = 0.0
                n_nd = int(len(pf)) if pf is not None else 0
            else:
                hv_val = 0.0
                n_nd = 0
            elapsed = time.time() - start_time
            recs = [r for r in evaluator.generation_history if r["generation"] == cur_gen]
            if recs:
                mean_acc = float(np.mean([r["mean_accuracy"] for r in recs]))
                worst_acc = float(np.mean([r["worst_accuracy"] for r in recs]))
                std_acc = float(np.mean([r["std_accuracy"] for r in recs]))
            else:
                mean_acc = worst_acc = std_acc = 0.0
            hv_history.append({
                "generation": cur_gen,
                "hv": float(hv_val),
                "n_nd": n_nd,
                "n_pop": int(len(F)) if F is not None else 0,
                "mean_accuracy": mean_acc,
                "worst_accuracy": worst_acc,
                "std_accuracy": std_acc,
                "eval_count": int(evaluator.eval_count),
                "elapsed": float(elapsed),
                "pop_size": int(pop_size),
                "n_obj": int(n_obj),
            })

    callback = NSGA3Callback()

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_generations),
        seed=seed,
        callback=callback,
        verbose=verbose,
    )

    return {
        "result": res,
        "hv_history": hv_history,
        "evaluator": evaluator,
        "ref_dirs": ref_dirs,
        "n_partitions": n_partitions,
        "pop_size_actual": pop_size,
        "hv_ref": hv_ref,
    }
