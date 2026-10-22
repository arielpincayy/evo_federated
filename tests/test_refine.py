import json

from evofederated.evolution.refine import candidate_set, select_final


def _rec(g, acc, p):
    return {"genome": g, "vector": json.dumps([0.1]), "acc_rep": acc, "n_params_norm": p}


def test_candidates_aggregate_by_genome_and_follow_layers():
    recs = [_rec("a", 0.8, 0.5), _rec("a", 0.6, 0.5), _rec("b", 0.5, 0.1), _rec("c", 0.4, 0.6), _rec("d", 0.3, 0.9)]
    c = candidate_set(recs, max_cand=3)
    assert [x["genome"] for x in c][:2] == ["a", "b"]  # a (media 0.7) y b no dominados; c y d en capas posteriores
    assert next(x for x in c if x["genome"] == "a")["n_evals"] == 2 and len(c) == 3


def test_select_final_is_nondominated_sorted_and_capped():
    r = [{"genome": g, "acc_refined": a, "n_params_norm": p} for g, a, p in
         [("a", 0.9, 0.5), ("b", 0.7, 0.1), ("c", 0.6, 0.3), ("d", 0.8, 0.9)]]
    f = select_final(r, k=2)
    assert [x["genome"] for x in f] == ["a", "b"]  # c dominada por b, d dominada por a
