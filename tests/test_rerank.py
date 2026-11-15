import json

from run_rerank import pick_candidates


def _rec(i):  # el individuo i domina a los anteriores (mismas columnas de tamaño, accuracies crecientes)
    return {"genome": f"g{i}", "vector": json.dumps([0.1]), "obj_acc": json.dumps([0.1 * i, 0.1 * i, -0.1]), "acc_rep": 0.1 * i, "n_params_norm": 0.1}


def test_pick_candidates_only_uses_the_prefix_of_the_archive():
    recs = [_rec(i) for i in range(8)]
    pre = {c["genome"] for c in pick_candidates(recs, 3, 2, prefix=4)}
    assert pre <= {"g0", "g1", "g2", "g3"} and len(pre) == 3
    full = {c["genome"] for c in pick_candidates(recs, 3, 2)}  # prefix = 0: todo el archivo
    assert "g7" in full and len(full) == 3
