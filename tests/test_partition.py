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
