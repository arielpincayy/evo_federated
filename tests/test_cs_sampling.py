import dataclasses

import numpy as np

from evofederated.evolution.cs_evaluator import ClusterSampleEvaluator
from evofederated.evolution.cs_nas import run_search
from evofederated.models.cnn_genome import CNNGenome, CNNSpace, max_n_params

N_CLASSES = 6  # clases del conjunto de juguete (ver conftest.py)
SPACE = CNNSpace(max_blocks=2, channels=(4, 8), kernels=(3,), activations=("relu",))
SHAPE = (1, 8, 8)
TRAIN = {"lr": 1e-2, "optimizer": "adam"}
FED = {"rounds": 1, "local_epochs": 1, "weight_by_cluster": True}


def _ev(clients, labels=None, sampler="cluster", monitor=True, seed=3, n_reps=3, holdout=True, objectives="mean_params"):
    return ClusterSampleEvaluator(clients, SHAPE, N_CLASSES, SPACE, TRAIN, FED, max_n_params(SPACE, SHAPE, N_CLASSES),
                                  labels=labels, sampler=sampler, n_reps=n_reps, seed=seed, monitor=monitor,
                                  eval_holdout=holdout, objectives=objectives)


def test_one_rep_per_cluster_weights_sum_to_one_and_deterministic(toy):
    clients, groups = toy
    ev = _ev(clients, groups)
    reps, w = ev.representatives(0)
    assert len(reps) == 3 and sorted(groups[r] for r in reps) == [0, 1, 2]
    assert abs(sum(w) - 1) < 1e-12 and w == [1 / 3] * 3
    assert ev.representatives(0) == _ev(clients, groups).representatives(0)
    assert any(ev.representatives(g)[0] != reps for g in range(1, 8))  # cambian entre generaciones


def test_holdout_evaluator_is_another_member_of_the_same_cluster(toy):
    clients, groups = toy
    ev = _ev(clients, groups)
    for g in range(6):
        reps, evals, _ = ev.assignment(g)
        assert all(r != e and groups[r] == groups[e] for r, e in zip(reps, evals))
        assert reps == _ev(clients, groups, holdout=False).assignment(g)[0]  # los reps no cambian con holdout
    assert _ev(clients, groups, holdout=False).assignment(0)[1] == reps_of(clients, groups)
    r_ev = _ev(clients, None, "random", n_reps=3).assignment(0)
    assert not set(r_ev[0]) & set(r_ev[1])  # random: evaluadores disjuntos de los que entrenan


def reps_of(clients, groups):
    return _ev(clients, groups).assignment(0)[0]


def test_evaluator_nodes_do_not_train(toy):
    clients, groups = toy
    ev = _ev(clients, groups, monitor=False)
    ev.evaluate_population(np.random.RandomState(0).rand(2, SPACE.n_var), 0)
    assert ev.n_local_trainings == 2 * 3 * FED["rounds"]  # solo los 3 reps entrenan


def test_singleton_clusters_equal_full_participation(toy):
    clients, _ = toy
    X = np.random.RandomState(0).rand(2, SPACE.n_var)
    f_cluster = _ev(clients, np.arange(len(clients)), "cluster", monitor=False).evaluate_population(X, 0)
    f_full = _ev(clients, None, "full", monitor=False).evaluate_population(X, 0)
    np.testing.assert_allclose(f_cluster, f_full)


def test_monitor_never_changes_fitness_and_same_arch_is_independent(toy):
    clients, groups = toy
    X = np.random.RandomState(1).rand(3, SPACE.n_var)
    f_on = _ev(clients, groups, monitor=True).evaluate_population(X, 0)
    ev = _ev(clients, groups, monitor=False)
    f_off = ev.evaluate_population(X, 0)
    np.testing.assert_allclose(f_on, f_off)
    again = ev.evaluate_population(X[[0]], 0)  # no hay pesos compartidos entre arquitecturas evaluadas antes
    np.testing.assert_allclose(again[0], f_off[0])


def test_search_only_reads_val_of_representatives_and_test_only_in_final(toy):
    clients, groups = toy
    calls = []
    for cid, c in clients.items():
        orig = c.evaluate

        def spy(model, device="cpu", split="test", criterion=None, _o=orig, _cid=cid):
            calls.append((_cid, split))
            return _o(model, device=device, split=split, criterion=criterion)

        c.evaluate = spy
    ev = _ev(clients, groups, monitor=False)
    ev.evaluate_population(np.random.RandomState(2).rand(2, SPACE.n_var), 0)
    reps, evals, _ = ev.assignment(0)
    assert calls and all(s == "val" and cid in set(evals) for cid, s in calls)
    calls.clear()
    ev.evaluate_test(CNNGenome.from_vector(np.full(SPACE.n_var, 0.5), SPACE), init_seed=1000)
    assert {s for _, s in calls} == {"test"} and len(ev.history) == 2


def test_nsga2_reevaluates_parents_with_same_reps_and_counts_cost(toy):
    clients, groups = toy
    ev = _ev(clients, groups)
    hist, rows = run_search(ev, SPACE, pop_size=4, n_generations=3, search="nsga2", seed=0, log=lambda *_: None)
    per_gen = {g: [h for h in hist if h["generation"] == g] for g in range(3)}
    assert len(per_gen[0]) == 4 and all(4 < len(per_gen[g]) <= 8 for g in (1, 2))
    for g in range(3):
        assert len({h["reps"] for h in per_gen[g]}) == 1  # mismos representantes en toda la generación
    assert ev.n_local_trainings == 3 * FED["rounds"] * len(hist)
    assert {"hv_rep", "hv_monitor", "est_rmse", "offspring_success", "diversity"} <= set(rows[-1])


def test_random_search_has_same_budget_per_generation(toy):
    clients, groups = toy
    ev = _ev(clients, groups)
    hist, _ = run_search(ev, SPACE, pop_size=4, n_generations=3, search="random", seed=0, log=lambda *_: None)
    assert [sum(h["generation"] == g for h in hist) for g in range(3)] == [4, 8, 8]


def test_per_node_objectives_one_per_cluster_and_consistent_with_records(toy):
    import json
    clients, groups = toy
    ev = _ev(clients, groups, objectives="per_node")
    assert ev.n_objectives() == 3
    X = np.random.RandomState(4).rand(3, SPACE.n_var)
    F = ev.evaluate_population(X, 0)
    assert F.shape == (3, 3)
    for f, rec in zip(F, ev.history):
        obj = json.loads(rec["obj_acc"])
        np.testing.assert_allclose(f, -np.array(obj))
        assert abs(rec["acc_rep"] - np.mean(obj)) < 1e-9  # clusters del mismo tamaño: pesos iguales
    assert ev.n_local_trainings == 3 * 3 * FED["rounds"]


def test_mean_params_still_has_two_objectives_and_records_obj_acc(toy):
    import json
    clients, groups = toy
    ev = _ev(clients, groups)
    F = ev.evaluate_population(np.random.RandomState(5).rand(2, SPACE.n_var), 0)
    assert F.shape == (2, 2) and len(json.loads(ev.history[0]["obj_acc"])) == 3


def test_per_cluster_mean_uses_only_untrained_members_and_never_test(toy):
    import json
    clients, groups = toy
    calls = []
    for cid, c in clients.items():
        orig = c.evaluate

        def spy(model, device="cpu", split="test", criterion=None, _o=orig, _cid=cid):
            calls.append((_cid, split))
            return _o(model, device=device, split=split, criterion=criterion)

        c.evaluate = spy
    ev = _ev(clients, groups, monitor=False, objectives="per_cluster_mean")
    F = ev.evaluate_population(np.random.RandomState(6).rand(1, SPACE.n_var), 0)
    reps = set(ev.assignment(0)[0])
    assert F.shape == (1, 3) and all(s == "val" and cid not in reps for cid, s in calls)
    assert len({cid for cid, _ in calls}) == 9 and ev.n_objective_node_evals == 9  # 3 clusters x 3 miembros sin entrenar
    assert ev.n_local_trainings == 3 * FED["rounds"]  # evaluar a más nodos no añade entrenamientos


def test_nsga3_runs_with_c_objectives_and_reevaluates_parents(toy):
    import json
    from pymoo.util.ref_dirs import get_reference_directions
    clients, groups = toy
    ev = _ev(clients, groups, objectives="per_node")
    hist, rows = run_search(ev, SPACE, pop_size=6, n_generations=2, search="nsga3", seed=0, ref_partitions=2, log=lambda *_: None)
    n0 = [h for h in hist if h["generation"] == 0]; n1 = [h for h in hist if h["generation"] == 1]
    assert len(n0) == len(get_reference_directions("das-dennis", 3, n_partitions=2)) == 6 and 6 < len(n1) <= 12
    assert len({h["reps"] for h in n1}) == 1  # mismos nodos para padres e hijos
    assert ev.n_local_trainings == 3 * FED["rounds"] * len(hist) and "best_min_obj" in rows[-1]


def test_nsga3_reference_directions_for_ten_objectives():
    from pymoo.util.ref_dirs import get_reference_directions
    assert len(get_reference_directions("das-dennis", 10, n_partitions=2)) == 55


def test_search_checkpoint_resumes_and_matches_an_uninterrupted_run(toy, tmp_path):
    clients, groups = toy
    ck = str(tmp_path / "ck.pkl")
    kw = dict(pop_size=6, search="nsga3", seed=0, log=lambda *_: None)
    ev0 = _ev(clients, groups, objectives="per_node")
    hist0, _ = run_search(ev0, SPACE, n_generations=3, **kw)  # referencia sin interrupciones
    ev1 = _ev(clients, groups, objectives="per_node")
    hist1, _ = run_search(ev1, SPACE, n_generations=2, checkpoint=ck, **kw)
    ev2 = _ev(clients, groups, objectives="per_node")  # proceso nuevo: reanuda y completa la generación 2
    hist2, rows2 = run_search(ev2, SPACE, n_generations=3, checkpoint=ck, **kw)
    assert hist2[: len(hist1)] == hist1 and len(rows2) == 3
    assert [h["genome"] for h in hist2] == [h["genome"] for h in hist0]  # misma trayectoria que sin interrupción
    assert ev2.n_evals == len(hist2) and ev2.n_local_trainings == 3 * FED["rounds"] * len(hist2)
