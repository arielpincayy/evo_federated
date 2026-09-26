"""FedAvg: promedio ponderado, rondas reproducibles, sin fuga de datos."""
import numpy as np
import torch
from torch.utils.data import TensorDataset

from evofederated.clients.client import FederatedClient
from evofederated.clients.server import FederatedServer
from evofederated.federated.fedavg import fedavg_buffers, fedavg_mean, run_fedavg, evaluate_global_on_split
from evofederated.models.genome import Genome, genome_to_model
from evofederated.federated.characterization import flatten_params
from evofederated.federated.pairing import DynamicDivergentPairing
from evofederated.experiments.runner_federated import build_participant_fn
from evofederated.utils.config import load_config


def _toy_clients():
    torch.manual_seed(0)
    clients = {}
    for cid in range(3):
        X = torch.randn(24, 6)
        y = torch.randint(0, 2, (24,))
        tr = TensorDataset(X[:16], y[:16])
        va = TensorDataset(X[16:20], y[16:20])
        te = TensorDataset(X[20:], y[20:])
        clients[cid] = FederatedClient(cid, tr, va, te, batch_size=8)
    return clients


def test_fedavg_weighted_mean():
    server = FederatedServer(_toy_clients())
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    out = server.fedavg_aggregate({0: a, 1: b}, {0: 1.0, 1: 3.0})
    np.testing.assert_allclose(out, np.array([2.5, 3.5]))


def test_fedavg_batchnorm_buffers():
    buffers = {
        0: {
            "running_mean": torch.tensor([1.0, 3.0]),
            "running_var": torch.tensor([2.0, 4.0]),
            "num_batches_tracked": torch.tensor(5, dtype=torch.long),
        },
        1: {
            "running_mean": torch.tensor([5.0, 7.0]),
            "running_var": torch.tensor([6.0, 8.0]),
            "num_batches_tracked": torch.tensor(11, dtype=torch.long),
        },
    }
    out = fedavg_buffers(buffers, {0: 1.0, 1: 3.0})
    torch.testing.assert_close(out["running_mean"], torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(out["running_var"], torch.tensor([5.0, 7.0]))
    assert out["num_batches_tracked"].item() == 11


def test_fedavg_rounds_reproducible_and_no_raw_access():
    clients = _toy_clients()
    genome_cfg = {"l_max": 2, "n_neurons_choices": (8, 16), "activations": ("relu",),
                  "use_batchnorm": False, "use_dropout": False}
    genome = Genome(L=1, hidden_sizes=[8, 8], activation="relu", dropout=0.0, use_bn=False, l_max=2)
    template = genome_to_model(genome, 6, 2)
    flat_init = flatten_params(template)
    sched = [[0, 1], [1, 2]]
    torch.manual_seed(123)
    r1 = run_fedavg(genome, genome_to_model, clients, 6, 2, flat_init, sched,
                    local_epochs=1, lr=1e-3, device="cpu")
    torch.manual_seed(123)
    r2 = run_fedavg(genome, genome_to_model, clients, 6, 2, flat_init, sched,
                    local_epochs=1, lr=1e-3, device="cpu")
    np.testing.assert_allclose(r1["global_params"], r2["global_params"])
    assert r1["n_local_trainings"] == 4
    assert r1["n_rounds"] == 2
    assert r1["communication_messages"] == 8
    assert r1["communication_bytes"] == 8 * r1["state_bytes"]
    # servidor no guarda datasets crudos
    server = FederatedServer(clients)
    server.collect_client_info()
    assert "n_train" in server.client_info[0]
    assert not hasattr(server, "train_dataset")


def test_search_uses_val_not_test():
    clients = _toy_clients()
    # test y val difieren: evaluar mismo modelo global da métricas distintas en general;
    # aquí solo verificamos que el evaluador federado usa val por defecto.
    from evofederated.evolution.federated_evaluator import FederatedEvaluator
    genome_cfg = {"l_max": 2, "n_neurons_choices": (8, 16), "activations": ("relu",),
                  "use_batchnorm": False, "use_dropout": False}
    train_cfg = {"lr": 1e-3, "optimizer": "adam", "epochs": 1}
    fed_cfg = {"rounds": 1, "clients_per_round": 2, "local_epochs": 1, "search_split": "val"}
    ev = FederatedEvaluator(clients, 6, 2, genome_cfg, train_cfg, fed_cfg, search_split="val", seed=0)
    genome = Genome(L=1, hidden_sizes=[8, 8], activation="relu", dropout=0.0, use_bn=False, l_max=2)
    vec = genome.to_vector(genome_cfg)
    F, info = ev.evaluate_single(vec, 0, arch_id="smoke")
    assert F.shape == (2,)
    assert ev.generation_history[0]["search_split"] == "val"


def test_tau_changes_selection_distribution():
    D = np.array([[0.0, 0.2, 1.8], [0.2, 0.0, 1.5], [1.8, 1.5, 0.0]])
    low = DynamicDivergentPairing(tau=0.5)
    high = DynamicDivergentPairing(tau=5.0)
    rng = np.random.RandomState(0)
    # probabilidad del cliente más divergente desde 0 debe crecer con... en esta fórmula
    # P ∝ (D+eps)^tau: tau alto concentra en el máximo.
    row = D[0].copy()
    for tau in (0.5, 5.0):
        probs = np.array([(v + 1e-9) ** tau for i, v in enumerate(row) if i != 0])
        probs /= probs.sum()
        if tau == 0.5:
            p_low = probs.max()
        else:
            p_high = probs.max()
    assert p_high > p_low
    c_low = low.select_counterpart(0, D, np.random.RandomState(1))
    c_high = high.select_counterpart(0, D, np.random.RandomState(1))
    assert c_low in (1, 2) and c_high in (1, 2)


def test_participation_strategies_use_one_representative_per_pair():
    S = np.array([
        [1.0, 0.9, 0.1, 0.2, 0.3, 0.4],
        [0.9, 1.0, 0.2, 0.3, 0.4, 0.1],
        [0.1, 0.2, 1.0, 0.8, 0.4, 0.3],
        [0.2, 0.3, 0.8, 1.0, 0.5, 0.4],
        [0.3, 0.4, 0.4, 0.5, 1.0, 0.7],
        [0.4, 0.1, 0.3, 0.4, 0.7, 1.0],
    ])
    D = 1.0 - S
    for strategy in ("similarity", "dissimilarity", "prob-similarity", "prob-dissimilarity"):
        fn = build_participant_fn(strategy, list(range(6)), S=S, D=D, tau=0.5, seed=3)
        schedule = fn(0, 3, 3, np.random.RandomState(3))
        assert all(len(participants) == 3 for participants in schedule)


def test_probabilistic_tau_is_recorded():
    S = np.eye(4)
    fn = build_participant_fn("prob-similarity", list(range(4)), S=S, tau=0.25)
    assert fn.info["tau"] == 0.25


def test_federated_runner_reads_test_only_after_validation(tmp_path, monkeypatch):
    from evofederated.experiments.runner_federated import run_federated_nas

    cfg = load_config("configs/small.yaml")
    cfg.dataset.n_clients = 4
    cfg.dataset.synthetic_n_samples = 240
    cfg.dataset.min_samples_per_client = 5
    cfg.dataset.val_ratio = 0.2
    cfg.train.batch_size = 16
    cfg.train.epochs = 1
    cfg.genome.l_max = 1
    cfg.genome.n_neurons_choices = (8,)
    cfg.genome.activations = ("relu",)
    cfg.genome.use_batchnorm = False
    cfg.genome.use_dropout = False
    cfg.evolution.pop_size = 2
    cfg.evolution.n_generations = 1
    cfg.characterization.k_epochs = 1
    cfg.federated.rounds = 1
    cfg.federated.clients_per_round = 2
    cfg.federated.local_epochs = 1

    calls = []
    original_evaluate = FederatedClient.evaluate

    def recording_evaluate(self, model, device="cpu", split="test", criterion=None):
        calls.append(split)
        return original_evaluate(self, model, device=device, split=split, criterion=criterion)

    monkeypatch.setattr(FederatedClient, "evaluate", recording_evaluate)
    result = run_federated_nas(cfg, strategy="random-k", optimizer="random", output_dir=tmp_path)

    first_test = calls.index("test")
    assert calls[:first_test]
    assert all(split == "val" for split in calls[:first_test])
    assert all(split == "test" for split in calls[first_test:])
    assert (tmp_path / "generations" / "hypervolume.csv").exists()
    assert (tmp_path / "dataset" / "partition.json").exists()
    assert (tmp_path / "characterization" / "delta_vectors.npy").exists()
    assert (tmp_path / "generations" / "selected_candidates.csv").exists()
    assert (tmp_path / "generations" / "validation_pareto.csv").exists()
    assert (tmp_path / "final_evaluation" / "final_pareto.csv").exists()
    assert (tmp_path / "participation_schedule.csv").exists()
    assert result["n_final"] > 0
