"""Refinamiento multifidelidad del frente de salida: reevaluar con más nodos solo a los candidatos prometedores."""
import json
from typing import Dict, List

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def candidate_set(records: List[dict], max_cand: int = 12) -> List[dict]:
    """Candidatos = capas no dominadas 1, 2, ... (por accuracy estimada media y parámetros) hasta max_cand arquitecturas.

    Se agrega por genoma (media de acc_rep entre todas sus evaluaciones): una arquitectura reevaluada varias veces cuenta una vez.
    """
    by: Dict[str, dict] = {}
    for r in records:
        g = by.setdefault(r["genome"], {"genome": r["genome"], "vector": json.loads(r["vector"]) if isinstance(r["vector"], str) else r["vector"],
                                        "n_params_norm": r["n_params_norm"], "accs": []})
        g["accs"].append(r["acc_rep"])
    cands = [dict(g, acc_rep_mean=float(np.mean(g["accs"])), n_evals=len(g["accs"])) for g in by.values()]
    F = np.array([[-c["acc_rep_mean"], c["n_params_norm"]] for c in cands])
    out: List[dict] = []
    for layer in NonDominatedSorting().do(F):
        out += sorted((cands[i] for i in layer), key=lambda c: -c["acc_rep_mean"])
        if len(out) >= max_cand:
            break
    return out[:max_cand]


def select_final(refined: List[dict], k: int = 5, acc_key: str = "acc_refined") -> List[dict]:
    """Mismo criterio que run_cs: frente no dominado por accuracy y parámetros, ordenado por accuracy, máximo k."""
    F = np.array([[-r[acc_key], r["n_params_norm"]] for r in refined])
    front = [refined[i] for i in NonDominatedSorting().do(F, only_non_dominated_front=True)]
    return sorted(front, key=lambda r: -r[acc_key])[:k]
