"""Single-objective GA runner with per-generation tracking."""
import time
import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem


class SimilarityNASProblem(Problem):
    def __init__(self, n_var: int, evaluator):
        super().__init__(n_var=n_var, n_obj=1, n_constr=0, xl=0.0, xu=1.0)
        self.evaluator = evaluator

    def _evaluate(self, X, out, *args, **kwargs):
        F = self.evaluator.evaluate_batch(X)
        out["F"] = F


class GenerationTracker(Callback):
    def __init__(self, evaluator, start_time):
        super().__init__()
        self.evaluator = evaluator
        self.start_time = start_time
        self.history = []

    def notify(self, algorithm):
        gen = int(algorithm.n_gen)
        self.evaluator.set_generation(gen)
        # compute population stats for current generation if available
        pop = algorithm.pop
        F = pop.get("F") if pop is not None else None
        n = len(F) if F is not None else 0
        # also get X? not needed
        # Gather recent generation_history entries for this gen
        # evaluator.generation_history contains entries with generation field
        # For this gen, filter
        recs = [r for r in self.evaluator.generation_history if r["generation"] == gen]
        if recs:
            mean_fits = [r["mean_accuracy"] for r in recs]
            best = float(np.max(mean_fits)) if mean_fits else 0.0
            mean = float(np.mean(mean_fits)) if mean_fits else 0.0
            median = float(np.median(mean_fits)) if mean_fits else 0.0
            std = float(np.std(mean_fits)) if mean_fits else 0.0
            worst = float(np.min(mean_fits)) if mean_fits else 0.0
        else:
            # fallback from F (which is 1 - accuracy)
            if F is not None and len(F):
                accs = 1 - F.flatten()
                best = float(np.max(accs))
                mean = float(np.mean(accs))
                median = float(np.median(accs))
                std = float(np.std(accs))
                worst = float(np.min(accs))
            else:
                best = mean = median = std = worst = 0.0

        elapsed = time.time() - self.start_time
        # representatives for this gen
        reps_cids, reps_pos = self.evaluator._get_representatives(gen)
        # n_evals this gen: pop_size * n_reps
        evals = int(self.evaluator.eval_count)
        rec = {
            "generation": gen,
            "best_fitness": best,
            "mean_fitness": mean,
            "median_fitness": median,
            "std_fitness": std,
            "worst_fitness": worst,
            "n_pop": n,
            "n_representatives": len(reps_cids),
            "representatives": str(reps_cids),
            "representatives_pos": str(reps_pos),
            "eval_count": evals,
            "elapsed": float(elapsed),
        }
        # per-rep mean accuracy across population if available
        if recs:
            # aggregate per rep acc across population
            for cid in self.evaluator.client_ids_sorted:
                col = f"acc_rep_{cid}"
                vals = [r[col] for r in recs if col in r]
                if vals:
                    rec[f"mean_acc_rep_{cid}"] = float(np.mean(vals))
        self.history.append(rec)


def run_single_objective_ga(
    evaluator,
    n_var: int,
    pop_size: int = 10,
    n_generations: int = 10,
    crossover_prob: float = 0.9,
    crossover_eta: float = 15,
    mutation_prob: float = 0.3,
    mutation_eta: float = 20,
    seed: int = 42,
    verbose: bool = False,
):
    algorithm = GA(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=crossover_prob, eta=crossover_eta),
        mutation=PM(eta=mutation_eta, prob=mutation_prob),
        eliminate_duplicates=True,
    )
    problem = SimilarityNASProblem(n_var=n_var, evaluator=evaluator)
    start_time = time.time()
    tracker = GenerationTracker(evaluator, start_time)
    # ensure evaluator starts at gen 0
    evaluator.set_generation(0)
    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_generations),
        seed=seed,
        callback=tracker,
        verbose=verbose,
    )
    elapsed = time.time() - start_time
    # If no generations recorded due to callback timing, ensure last gen captured
    # callback is called each generation; for n_generations we should have n_generations entries?
    # pymoo first call at gen 1? Let's ensure we have entries. If history empty, create from evaluator data.
    return {
        "result": res,
        "history": tracker.history,
        "evaluator": evaluator,
        "elapsed": elapsed,
    }
