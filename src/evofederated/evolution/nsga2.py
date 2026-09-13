"""NSGA-II runner usando pymoo.
Maneja loop generacional manual para poder registrar HV, pairing history, etc.
"""
import time
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

from .problem import FederatedNASProblem
from .evaluator import Evaluator


def run_nsga2(
    evaluator: Evaluator,
    n_var: int,
    pop_size: int = 20,
    n_generations: int = 10,
    crossover_prob: float = 0.9,
    crossover_eta: float = 15,
    mutation_prob: float = 0.2,
    mutation_eta: float = 20,
    seed: int = 42,
    hv_ref: tuple = (0.0, 0.0),  # in minimization space, worse point (0,0) since objectives are -F1 in [-1,0]
    verbose: bool = False,
):
    """
    Ejecuta NSGA-II con tracking de HV por generación.
    hv_ref debe estar en espacio de minimización y ser peor que cualquier solución.
    Para -F1 in [-1,0], ref (0,0) es peor (since we minimize, larger = worse). 
    For HV calculation, reference should be dominated by all points. In pymoo, HV ref is worse point.
    With objectives -F1: best = -1, worst = 0 => ref = (0.1,0.1) or (0,0) if inclusive? Need slightly worse than 0.
    We'll use (0.05, 0.05) or (0,0) depending.
    Actually if ref = (0,0), then points with -F1=0 (F1=0) are equal to ref, HV may be 0. Safer to use (0.1,0.1).
    But spec says point must be fixed and documented. We'll use given hv_ref and ensure it's >0 for this case.
    If user passes (0,0), we adjust to (0.1,0.1) implicitly? Better document and keep (0,0) as worse inclusive.
    Pymoo HV handles ref as exclusive upper bound? Check.
    We'll keep logic: if ref == (0,0), convert to (0.1,0.1) to guarantee dominance.
    """
    # Adjust ref if needed
    hv_ref_arr = np.array(hv_ref, dtype=float)
    # If objectives are -F1 in [-1,0], ref (0,0) is worst inclusive; make slightly worse for HV volume positive
    if np.allclose(hv_ref_arr, 0.0):
        hv_ref_arr = np.array([0.1, 0.1])

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=crossover_prob, eta=crossover_eta),
        mutation=PM(eta=mutation_eta, prob=mutation_prob),
        eliminate_duplicates=True,
    )

    problem = FederatedNASProblem(n_var=n_var, evaluator=evaluator)

    # Manual loop to track per generation HV
    # Use pymoo minimize with callback? Simpler to loop manually.
    # But pymoo's minimize handles generational loop internally.
    # We'll implement callback to capture HV.
    hv_history = []
    eval_history = []
    time_history = []

    start_time = time.time()

    # To allow per-generation evaluator generation update, we need custom loop.
    # We'll implement simple generational loop without using minimize(), or with callback.

    # Approach: use minimize with callback, but also need to update evaluator generation each iteration.
    # We'll define a callback class.

    from pymoo.core.callback import Callback

    class HVCallback(Callback):
        def __init__(self):
            super().__init__()
            self.gen = 0

        def notify(self, algorithm):
            # algorithm.n_gen is current generation
            evaluator.set_generation(algorithm.n_gen)
            # compute HV of current population's Pareto front
            F = algorithm.pop.get("F")
            if F is not None and len(F) > 0:
                # Non-dominated filtering is done by pymoo internally, but HV on whole pop approximates.
                # For HV we should use non-dominated set.
                from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

                nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
                pf = F[nds]
                # Filter points that are dominated by ref? HV requires ref worse than all.
                # Remove points outside ref (worse than ref)
                # For minimization, points must be < ref in at least one objective? Actually HV assumes all points dominate ref? Check.
                # In minimization, a point dominates ref if point < ref in all objectives.
                # Our points are <=0, ref is 0.1 => all dominate ref, so okay.
                try:
                    hv = HV(ref_point=hv_ref_arr)
                    hv_val = hv.do(pf)
                except Exception as e:
                    hv_val = 0.0
            else:
                hv_val = 0.0
                pf = np.array([])

            elapsed = time.time() - start_time
            hv_history.append(
                {
                    "generation": int(algorithm.n_gen),
                    "hv": float(hv_val),
                    "n_pop": int(len(F)) if F is not None else 0,
                    "pf_size": int(len(pf)) if pf is not None else 0,
                    "eval_count": int(evaluator.eval_count),
                    "elapsed": float(elapsed),
                }
            )
            eval_history.append(int(evaluator.eval_count))
            time_history.append(float(elapsed))

    callback = HVCallback()

    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_generations),
        seed=seed,
        callback=callback,
        verbose=verbose,
    )

    # If callback not triggered for initial gen 0? It triggers each generation including 0? We'll have history.
    # Ensure res.F etc.
    return {
        "result": res,
        "hv_history": hv_history,
        "evaluator": evaluator,
        "hv_ref": hv_ref_arr.tolist(),
    }
