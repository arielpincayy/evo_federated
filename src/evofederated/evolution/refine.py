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


def select_hof(records: List[dict], K: int = 12) -> List[dict]:
    """Hall of fame en C objetivos: conjunto no dominado del archivo reducido a K representantes (regla fija y determinista).

    Se agrega por genoma (media de sus vectores `obj_acc` y de `acc_rep`). Candidatos = capas no dominadas 1, 2, ... hasta tener
    al menos K. Se incluyen (a) el mejor por la media ponderada `acc_rep`, (b) el mejor por su peor objetivo (max-min) y se completa
    hasta K por selección farthest-first en el espacio de objetivos normalizado (empates: menor índice).
    """
    by: Dict[str, dict] = {}
    for r in records:
        obj = json.loads(r["obj_acc"]) if isinstance(r["obj_acc"], str) else list(r["obj_acc"])
        g = by.setdefault(r["genome"], {"genome": r["genome"], "vector": json.loads(r["vector"]) if isinstance(r["vector"], str) else r["vector"],
                                        "n_params_norm": r["n_params_norm"], "objs": [], "accs": []})
        g["objs"].append(obj); g["accs"].append(r["acc_rep"])
    cands = [dict(g, obj=np.mean(g["objs"], axis=0), acc_rep_mean=float(np.mean(g["accs"])), n_evals=len(g["accs"])) for g in by.values()]
    O = np.array([c["obj"] for c in cands])
    pool: List[int] = []
    for layer in NonDominatedSorting().do(-O):
        pool += sorted(int(i) for i in layer)
        if len(pool) >= K:
            break
    Z = O[pool]; rng_ = np.where(np.ptp(Z, axis=0) > 0, np.ptp(Z, axis=0), 1.0); Z = (Z - Z.min(axis=0)) / rng_
    chosen = [int(np.argmax([cands[i]["acc_rep_mean"] for i in pool]))]
    b = int(np.argmax(O[pool].min(axis=1)))
    if b not in chosen:
        chosen.append(b)
    while len(chosen) < min(K, len(pool)):
        d = np.min(np.linalg.norm(Z[:, None] - Z[chosen][None], axis=2), axis=1)
        d[chosen] = -1.0
        chosen.append(int(np.argmax(d)))
    out = []
    for k, j in enumerate(chosen):
        c = cands[pool[j]]
        out.append({"genome": c["genome"], "vector": c["vector"], "n_params_norm": c["n_params_norm"], "acc_rep_mean": c["acc_rep_mean"],
                    "obj_mean": [float(x) for x in c["obj"]], "n_evals": c["n_evals"], "n_pool": len(pool),
                    "reason": "max_media" if k == 0 else ("max_min" if k == 1 and chosen[1] == b else "farthest_first")})
    return out
