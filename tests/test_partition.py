import numpy as np
from evofederated.data.datasets import dirichlet_partition, heterogeneity_metrics

def test_dirichlet_partition_shapes():
    labels = np.repeat(np.arange(10), 100)  # 1000 samples
    part = dirichlet_partition(labels, n_clients=5, alpha=0.5, seed=42)
    assert part.n_clients == 5
    assert part.class_distribution.shape == (5, 10)
    total = part.class_distribution.sum()
    assert total == len(labels)
    # splits sum to client_indices
    for cid in range(5):
        total_client = len(part.splits[cid]["train"]) + len(part.splits[cid]["val"]) + len(part.splits[cid]["test"])
        assert total_client == len(part.client_indices[cid])

def test_dirichlet_alpha_effect():
    labels = np.repeat(np.arange(5), 200)
    part_high = dirichlet_partition(labels, n_clients=4, alpha=10, seed=0)
    part_low = dirichlet_partition(labels, n_clients=4, alpha=0.1, seed=0)
    metrics_high = heterogeneity_metrics(part_high.class_distribution)
    metrics_low = heterogeneity_metrics(part_low.class_distribution)
    # low alpha should be more heterogeneous (higher pairwise TV)
    assert metrics_low["mean_pairwise_TV"] >= metrics_high["mean_pairwise_TV"]

def test_reproducibility():
    labels = np.random.randint(0, 3, size=300)
    p1 = dirichlet_partition(labels, n_clients=3, alpha=0.5, seed=123)
    p2 = dirichlet_partition(labels, n_clients=3, alpha=0.5, seed=123)
    np.testing.assert_array_equal(p1.class_distribution, p2.class_distribution)
    p3 = dirichlet_partition(labels, n_clients=3, alpha=0.5, seed=999)
    assert not np.array_equal(p1.class_distribution, p3.class_distribution)


def test_natural_partition_one_writer_per_client_and_disjoint():
    from evofederated.data.datasets import natural_partition
    rng = np.random.RandomState(0)
    writers = np.array([f"w{i}" for i in rng.randint(0, 30, 3000)])
    labels = rng.randint(0, 62, 3000)
    p = natural_partition(writers, labels, n_clients=10, seed=1, min_samples_per_client=50)
    seen = set()
    for cid, s in p.splits.items():
        idx = s["train"] + s["val"] + s["test"]
        assert len(set(idx)) == len(idx) and not seen & set(idx)
        seen |= set(idx)
        assert len(set(writers[idx])) == 1 and len(idx) >= 50
        assert len(s["val"]) > 0 and len(s["test"]) > 0
    assert p.class_distribution.shape == (10, 62)
    q = natural_partition(writers, labels, n_clients=10, seed=1, min_samples_per_client=50)
    assert p.splits == q.splits
