"""MOEA/D runner estándar con descomposición Tchebycheff (adecuada para conflictivos/many)."""
import time
import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.decomposition.weighted_sum import WeightedSum
from pymoo.decomposition.pbi import PBI
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from .many_objective import ManyObjectiveProblem, ManyObjectiveEvaluator
from pymoo.core.callback import Callback


def get_moead_ref_dirs(n_obj, pop_size_hint=20, seed=42):
    """
    Retorna ref_dirs adecuados para N objetivos.
    Documenta decisiones: para N <=5 usamos particiones que aproximen pop_size_hint,
    para N>5 usamos n_partitions=1 => pop = N (mínimo viable), o ajuste.
    """
    if n_obj <= 3:
        # try to get close to pop_size_hint
        # n_dirs = C(p + n_obj -1, n_obj-1)
        for p in [12, 8, 6, 4, 3, 2, 1]:
            from math import comb
            n_dirs = comb(p + n_obj - 1, n_obj - 1)
            if n_dirs <= pop_size_hint * 1.5 and n_dirs >= pop_size_hint * 0.5:
                return get_reference_directions("das-dennis", n_obj, n_partitions=p), p, n_dirs
        return get_reference_directions("das-dennis", n_obj, n_partitions=4), 4, 0
    elif n_obj <= 5:
        for p in [6,5,4,3,2,1]:
            from math import comb
            n_dirs = comb(p + n_obj - 1, n_obj - 1)
            if n_dirs <= pop_size_hint * 2:
                return get_reference_directions("das-dennis", n_obj, n_partitions=p), p, n_dirs
        return get_reference_directions("das-dennis", n_obj, n_partitions=2), 2, 0
    else:
        # many-objective N>=8: minimal pop = N with p=1
        # if pop_size_hint larger, try p=2 if not too huge
        from math import comb
        n_dirs_p1 = n_obj  # p=1 => N
        n_dirs_p2 = comb(n_obj + 1, n_obj - 1)  # = n_obj*(n_obj+1)/2 huge
        if pop_size_hint >= n_dirs_p2 and n_obj <= 12:
            return get_reference_directions("das-dennis", n_obj, n_partitions=2), 2, n_dirs_p2
        else:
            # p=1 minimal
            return get_reference_directions("das-dennis", n_obj, n_partitions=1), 1, n_dirs_p1


def run_moead(
    evaluator: ManyObjectiveEvaluator,
    n_var: int,
    pop_size: int = 20,
    n_generations: int = 10,
    decomposition: str = "tchebycheff",
    n_neighbors: int = 20,
    prob_neighbor_mating: float = 0.9,
    seed: int = 42,
    hv_ref: np.ndarray = None,
    verbose: bool = False,
):
    n_obj = evaluator.n_obj
    # get ref_dirs appropriately
    ref_dirs, n_partitions, n_dirs = get_moead_ref_dirs(n_obj, pop_size_hint=pop_size, seed=seed)
    actual_pop = len(ref_dirs)
    if actual_pop != pop_size:
        print(f"[MOEA/D] N={n_obj} pop_hint {pop_size} -> adapted pop {actual_pop} (p={n_partitions}, n_dirs={n_dirs})")
        pop_size = actual_pop

    # choose decomposition
    if decomposition == "tchebycheff":
        decomp = Tchebicheff()
    elif decomposition == "weighted_sum":
        decomp = WeightedSum()
    elif decomposition == "pbi":
        decomp = PBI(theta=5.0)
    else:
        decomp = Tchebicheff()

    # n_neighbors cannot exceed pop_size
    n_neighbors = min(n_neighbors, pop_size)
    if n_neighbors < 2:
        n_neighbors = min(2, pop_size)

    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=n_neighbors,
        decomposition=decomp,
        prob_neighbor_mating=prob_neighbor_mating,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=1.0, eta=20),
        mutation=PM(eta=20),
    )

    problem = ManyObjectiveProblem(n_var=n_var, evaluator=evaluator)

    # HV tracking per generation
    # ref point for minimization 1-acc: worst =1.0 per obj, use 1.1 or 1.0 slightly worse
    if hv_ref is None:
        hv_ref = np.ones(n_obj) * 1.1  # dominates all feasible [0,1]

    hv_history = []
    start_time = time.time()

    class MOEADCallback(Callback):
        def __init__(self):
            super().__init__()
            self.gen = 0
        def notify(self, algorithm):
            # algorithm.n_gen is current gen index (starts 1)
            cur_gen = int(algorithm.n_gen)
            evaluator.set_generation(cur_gen)
            F = algorithm.pop.get("F")
            if F is not None and len(F) > 0:
                # filter nondominated for HV
                try:
                    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
                    pf = F[nds]
                except Exception:
                    pf = F
                try:
                    hv = HV(ref_point=hv_ref)
                    hv_val = hv.do(pf) if len(pf) else 0.0
                except Exception as e:
                    hv_val = 0.0
                n_nd = int(len(pf)) if pf is not None else 0
            else:
                hv_val = 0.0
                n_nd = 0
                pf = np.array([])
            elapsed = time.time() - start_time
            # aggregated mean accuracy etc from evaluator history for this gen
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

    callback = MOEADCallback()

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
        "decomposition": decomposition,
        "n_neighbors": n_neighbors,
    }
