"""Bucle de búsqueda: NSGA-II (o random search con el mismo coste) sobre ClusterSampleEvaluator.

Cada generación:
  * NSGA-II reevalúa padres e hijos con los mismos representantes (si no, comparar fitness medidos con
    representantes distintos sesgaría la selección);
  * random search evalúa el mismo número de individuos por generación, todos nuevos (mismo coste).
Registra por generación: HV, error del estimador (rep vs todos los nodos), éxito de hijos y diversidad.
"""
import json
import os
import pickle
from math import comb

import cloudpickle  # los algoritmos de pymoo no se serializan con pickle estándar
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import kendalltau

HV_REF = np.array([0.0, 1.05])  # (-acc, n_params/n_params_max): peor que cualquier solución


def auto_partitions(k: int, max_dirs: int) -> int:
    """Mayor nº de particiones p de Das-Dennis con C(k+p-1, p) <= max_dirs (al menos 1)."""
    p = 1
    while comb(k + p, p + 1) <= max_dirs:
        p += 1
    return p


def n_dirs(k: int, p: int) -> int:
    return comb(k + p - 1, p)


def generations_for_budget(budget: int, pop: int) -> int:
    """Generaciones G con pop + (G-1)*2*pop ~ budget evaluaciones (padres + hijos se reevalúan desde la 2ª generación)."""
    return max(1, round(1 + (budget - pop) / (2 * pop)))


def genome_duplicates(space):
    """Duplicados a nivel de arquitectura decodificada (no del vector): dos vectores con el mismo genoma son la misma arquitectura."""
    from ..models.cnn_genome import CNNGenome
    return ElementwiseDuplicateElimination(cmp_func=lambda a, b: CNNGenome.from_vector(a.X, space).to_dict() == CNNGenome.from_vector(b.X, space).to_dict())


def _nd(F: np.ndarray) -> np.ndarray:
    return NonDominatedSorting().do(F, only_non_dominated_front=True)


def _hv(F: np.ndarray) -> float:
    return float(HV(ref_point=HV_REF).do(F)) if len(F) else 0.0


def output_front(records: list, multi: bool = False):
    """Frente del algoritmo con TODA la información que tiene (acc estimada por representantes).

    `multi=False`: no dominadas por (acc_rep, parámetros). `multi=True`: no dominadas en los C objetivos `obj_acc` (maximizar)."""
    if multi:
        F = -np.array([json.loads(r["obj_acc"]) for r in records])
    else:
        F = np.array([[-r["acc_rep"], r["n_params_norm"]] for r in records])
    return [records[i] for i in _nd(F)]


def _gen_stats(recs: list) -> dict:
    a = np.array([r["acc_rep"] for r in recs])
    out = {"acc_rep_mean": float(a.mean())}
    if "mon_acc_mean" in recs[0]:
        m = np.array([r["mon_acc_mean"] for r in recs])
        out.update({"est_bias": float((a - m).mean()), "est_rmse": float(np.sqrt(((a - m) ** 2).mean())),
                    "est_kendall": float(kendalltau(a, m)[0]) if len(recs) > 2 and a.std() > 0 and m.std() > 0 else float("nan")})
    return out


def run_search(ev, space, pop_size: int, n_generations: int, search: str = "nsga2", seed: int = 42,
               crossover_prob: float = 0.9, crossover_eta: float = 15, mutation_prob: float = 0.2,
               mutation_eta: float = 20, ref_partitions: int = 2, checkpoint: str = None, log=print,
               reeval_parents: bool = True, eval_budget: int = 0):
    """Devuelve (history de evaluaciones, métricas por generación).

    search: nsga2 (2 objetivos), nsga3 (un objetivo por cluster; direcciones de Das-Dennis con `ref_partitions` particiones y
    población = nº de direcciones) o random (mismo nº de evaluaciones por generación que el método evolutivo equivalente).
    `reeval_parents=False` (MOEA v2): solo se evalúan los hijos (la aleatoria sortea `pop_size` por generación), los duplicados se eliminan a nivel de
    arquitectura decodificada y la búsqueda para al alcanzar `eval_budget` evaluaciones reales (`n_generations` es el tope)."""
    rng = np.random.RandomState(seed)
    gen_rows = []
    multi = ev.objectives != "mean_params"
    evo = search in ("nsga2", "nsga3")
    if evo:
        ops = dict(sampling=FloatRandomSampling(), crossover=SBX(prob=crossover_prob, eta=crossover_eta),
                   mutation=PM(prob=mutation_prob, eta=mutation_eta), eliminate_duplicates=True if reeval_parents else genome_duplicates(space))
        if search == "nsga3":
            ref_dirs = get_reference_directions("das-dennis", ev.n_objectives(), n_partitions=ref_partitions)
            algo = NSGA3(ref_dirs=ref_dirs, pop_size=len(ref_dirs), **ops)
        else:
            algo = NSGA2(pop_size=pop_size, **ops)
        algo.setup(Problem(n_var=space.n_var, n_obj=ev.n_objectives(), xl=0.0, xu=1.0), termination=("n_gen", n_generations), seed=seed, verbose=False)
        log(f"{search}: {ev.n_objectives()} objetivos, población {algo.pop_size}")
    elif search != "random":
        raise ValueError(f"search desconocido: {search}")

    first = 0
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint, "rb") as f:
            st = pickle.load(f)
        ev.history, gen_rows, first = st["history"], st["gen_rows"], st["g"] + 1
        for k, v in st["counters"].items():
            setattr(ev, k, v)
        rng.set_state(st["rng"])
        if evo:
            algo = cloudpickle.loads(st["algo"])
        log(f"reanudando desde la generación {first} ({len(ev.history)} evaluaciones ya hechas)")
    if eval_budget and not reeval_parents and ev.n_evals >= eval_budget:
        first = n_generations  # la búsqueda ya había alcanzado su presupuesto

    for g in range(first, n_generations):
        start = len(ev.history)
        success = float("nan")
        if evo:
            off = algo.ask()
            if g == 0 or not reeval_parents:
                off.set("F", ev.evaluate_population(off.get("X"), g))
            else:
                parents = algo.pop
                X = np.vstack([parents.get("X"), off.get("X")])
                F = ev.evaluate_population(X, g)
                parents.set("F", F[: len(parents)])
                off.set("F", F[len(parents):])
            if g > 0:
                algo.pop.set("is_off", False)
            off.set("is_off", True)
            algo.tell(infills=off)
            success = float(np.mean([bool(i.get("is_off")) for i in algo.pop])) if g > 0 else float("nan")
            Xpop = algo.pop.get("X")
        else:
            n = pop_size if (g == 0 or not reeval_parents) else 2 * pop_size  # con reevaluación: mismo nº que el evolutivo (padres + hijos)
            Xpop = rng.rand(n, space.n_var)
            ev.evaluate_population(Xpop, g)
        recs = ev.history[start:]
        front = output_front(ev.history, multi=multi)
        row = {"generation": g, "n_evals": ev.n_evals, "n_local_trainings_search": ev.n_local_trainings,
               "front_size": len(front), "offspring_success": success,
               "diversity": float(np.mean(np.linalg.norm(Xpop[:, None] - Xpop[None], axis=2)) / np.sqrt(space.n_var)),
               **(_gen_stats(recs) if recs else {})}
        if multi and recs:  # C objetivos: sin HV; se reportan el mejor por media ponderada y por peor objetivo entre todo lo evaluado
            allr = ev.history
            row["best_acc_rep"] = float(max(r["acc_rep"] for r in allr))
            row["best_min_obj"] = float(max(min(json.loads(r["obj_acc"])[: ev.n_acc_obj]) for r in allr))
            row["mean_n_params_norm"] = float(np.mean([r["n_params_norm"] for r in recs]))
        elif not multi:
            row["hv_rep"] = _hv(np.array([[-r["acc_rep"], r["n_params_norm"]] for r in front]))
            if recs and "mon_acc_mean" in recs[0]:
                row["hv_monitor"] = _hv(np.array([[-r["mon_acc_mean"], r["n_params_norm"]] for r in front]))
        gen_rows.append(row)
        if checkpoint:  # escritura atómica: una interrupción nunca deja un checkpoint a medias
            st = {"g": g, "history": ev.history, "gen_rows": gen_rows, "rng": rng.get_state(), "algo": cloudpickle.dumps(algo) if evo else None,
                  "counters": {k: getattr(ev, k) for k in ("n_evals", "n_local_trainings", "n_monitor_node_evals", "n_objective_node_evals", "train_time", "n_cache_hits", "_cache")}}
            with open(checkpoint + ".tmp", "wb") as f:
                pickle.dump(st, f)
            os.replace(checkpoint + ".tmp", checkpoint)
        log(f"gen {g}: evals={row['n_evals']} front={row['front_size']}" + ("" if multi else f" hv_rep={row['hv_rep']:.4f}"))
        if eval_budget and not reeval_parents and ev.n_evals >= eval_budget:
            log(f"presupuesto alcanzado: {ev.n_evals} evaluaciones reales ({ev.n_cache_hits} aciertos de caché)")
            break
    return ev.history, gen_rows
