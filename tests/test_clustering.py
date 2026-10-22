import numpy as np

from evofederated.federated.clustering import cluster_delta, cluster_labels_oracle, cluster_quality, cluster_random


def _deltas(seed=0):
    rng = np.random.RandomState(seed)
    dirs = rng.randn(3, 200)
    truth = np.repeat(np.arange(3), 10)
    return dirs[truth] + 0.1 * rng.randn(30, 200), truth


def test_delta_clustering_recovers_known_groups():
    from sklearn.metrics import adjusted_rand_score
    d, truth = _deltas()
    for method in ("kmeans", "average"):
        lab = cluster_delta(d, 3, method, seed=0)
        assert adjusted_rand_score(truth, lab) == 1.0


def test_cluster_quality_and_oracle():
    truth = np.repeat(np.arange(3), 10)
    dist = np.eye(3)[truth] * 100 + 1  # nodos de un mismo grupo comparten distribución de clases
    q = cluster_quality(truth, dist, cluster_labels_oracle(dist, 3))
    assert q["eta2_label_dist"] > 0.99 and q["ari_vs_label_clusters"] == 1.0
    assert cluster_quality(cluster_random(30, 3, seed=1), dist)["eta2_label_dist"] < 0.3


def test_random_clusters_are_nonempty_and_balanced():
    lab = cluster_random(23, 5, seed=0)
    counts = np.bincount(lab)
    assert len(counts) == 5 and counts.min() >= 4 and counts.max() <= 5
