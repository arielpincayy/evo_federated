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


def test_dbscan_assigns_noise_to_nearest_cluster_and_labels_are_contiguous():
    from sklearn.metrics import adjusted_rand_score
    from evofederated.federated.clustering import cluster_dbscan
    d, truth = _deltas()
    outliers = np.stack([d[truth == 0].mean(axis=0), d[truth == 2].mean(axis=0)]) + 3.0 * np.random.RandomState(1).randn(2, 200)
    lab, n_noise = cluster_dbscan(np.vstack([d, outliers]), q=0.9, min_samples=5)
    assert n_noise >= 1 and set(lab) == set(range(lab.max() + 1))
    assert adjusted_rand_score(truth, lab[:30]) == 1.0


def test_dbscan_on_random_data_still_assigns_every_node():
    from evofederated.federated.clustering import cluster_dbscan
    lab, n_noise = cluster_dbscan(np.random.RandomState(0).randn(30, 200), q=0.5, min_samples=5)
    assert len(lab) == 30 and lab.min() == 0 and set(lab) == set(range(lab.max() + 1)) and n_noise > 0
