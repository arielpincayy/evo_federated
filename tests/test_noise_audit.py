import numpy as np

from run_noise_audit import build_reps, e3_scores


def test_build_reps_keeps_cluster_weights_and_membership():
    labels = np.array([0] * 15 + [1] * 25)
    reps, rc, w = build_reps(labels, list(range(40)), seed=1, draw=3, r=3)
    assert len(reps) == 6 and len(set(reps)) == 6 and abs(sum(w) - 1) < 1e-12
    for ci, size in ((0, 15), (1, 25)):
        assert abs(sum(wi for wi, c in zip(w, rc) if c == ci) - size / 40) < 1e-12  # el peso total del cluster se conserva
        assert all(labels[r_] == ci for r_, c in zip(reps, rc) if c == ci)
    assert build_reps(labels, list(range(40)), 1, 3, 3) == (reps, rc, w) and build_reps(labels, list(range(40)), 1, 4, 3)[0] != reps


def test_e3_scores_exclude_every_trainer_of_the_cluster(tmp_path):
    labels = np.array([0] * 5 + [1] * 5)
    acc = np.zeros((2, 10)); acc[:, [0, 1, 5]] = 1.0  # los entrenadores (nodos 0, 1 del cluster 0 y 5 del 1) valen 1; el resto 0
    f = tmp_path / "z.npz"
    np.savez(f, acc=acc, reps=[0, 1, 5], rep_cluster=[0, 0, 1], labels=labels)
    assert np.allclose(e3_scores(np.load(f)), 0.0)  # r > 1: no cuenta a ninguno de los entrenadores
    np.savez(f, acc=acc, reps=[0, 5], labels=labels)  # formato r = 1 (sin rep_cluster)
    assert e3_scores(np.load(f))[0, 0] == 0.25  # cluster 0 sin el nodo 0: {1, 2, 3, 4} con un 1 (nodo 1) => 0.25
