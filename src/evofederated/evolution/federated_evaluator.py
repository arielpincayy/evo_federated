"""Evaluador federado común: cada genoma -> modelo global FedAvg -> media/mínimo en validation.

Búsqueda usa split="val". Test queda reservado para evaluación final externa.
"""
import hashlib
import json
import numpy as np
import torch

from ..models.genome import Genome, genome_to_model, model_stats
from ..federated.characterization import flatten_params
from ..federated.fedavg import run_fedavg, evaluate_global_on_split


def seed_for_genome(genome: Genome) -> int:
    h = hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


class FederatedEvaluator:
    def __init__(
        self,
        clients: dict,
        input_dim: int,
        n_classes: int,
        genome_config: dict,
        train_config: dict,
        federated_config: dict,
        participant_fn=None,
        search_split: str = "val",
        device: str = "cpu",
        seed: int = 42,
    ):
        self.clients = clients
        self.client_ids_sorted = sorted(clients.keys())
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.genome_config = genome_config
        self.train_config = train_config
        self.fed = dict(federated_config)
        self.participant_fn = participant_fn
        self.search_split = search_split
        self.device = device
        self.rng = np.random.RandomState(seed)
        self.eval_count = 0
        self.n_local_trainings = 0
        self.n_rounds_total = 0
        self.communication_bytes_total = 0
        self.communication_messages_total = 0
        self.eval_time_total = 0.0
        self.generation = 0
        self.generation_history = []

    def set_generation(self, gen: int):
        self.generation = int(gen)

    def _participants(self) -> list:
        rounds = int(self.fed.get("rounds", 2))
        k = int(self.fed.get("clients_per_round", min(2, len(self.client_ids_sorted))))
        if self.participant_fn is not None:
            sched = self.participant_fn(self.generation, rounds, k, self.rng)
            # ponytail: solo valida rondas y no vacío; cada estrategia define su k
            assert len(sched) == rounds and all(len(p) > 0 for p in sched)
            return sched
        # defecto: todos los clientes cada ronda
        return [list(self.client_ids_sorted) for _ in range(rounds)]

    def evaluate_single(self, vec: np.ndarray, individual_idx: int, arch_id: str = None):
        genome = Genome.from_vector(vec, self.genome_config)
        stats = model_stats(genome_to_model(genome, self.input_dim, self.n_classes))
        torch.manual_seed(seed_for_genome(genome))
        template = genome_to_model(genome, self.input_dim, self.n_classes)
        flat_init = flatten_params(template)

        sched = self._participants()
        fed = run_fedavg(
            genome,
            genome_to_model,
            self.clients,
            self.input_dim,
            self.n_classes,
            flat_init,
            sched,
            local_epochs=int(self.fed.get("local_epochs", 1)),
            lr=float(self.train_config.get("lr", 1e-3)),
            optimizer_name=str(self.train_config.get("optimizer", "adam")),
            device=self.device,
        )
        per_client = evaluate_global_on_split(
            fed["model"], self.clients, split=self.search_split, device=self.device
        )
        accs = [per_client[c]["accuracy"] for c in self.client_ids_sorted]
        f1s = [per_client[c]["macro_f1"] for c in self.client_ids_sorted]
        mean_acc, min_acc = float(np.mean(accs)), float(np.min(accs))
        mean_f1, min_f1 = float(np.mean(f1s)), float(np.min(f1s))

        self.eval_count += 1
        self.n_local_trainings += int(fed["n_local_trainings"])
        self.n_rounds_total += int(fed["n_rounds"])
        self.communication_bytes_total += int(fed["communication_bytes"])
        self.communication_messages_total += int(fed["communication_messages"])
        self.eval_time_total += float(fed["train_time"])
        rec = {
            "generation": int(self.generation),
            "individual_idx": int(individual_idx),
            "arch_id": arch_id,
            "genome": json.dumps(genome.to_dict()),
            "mean_accuracy": mean_acc,
            "min_accuracy": min_acc,
            "mean_f1": mean_f1,
            "min_f1": min_f1,
            "n_params": stats["n_params"],
            "participants": str(sched),
            "n_local_trainings": int(fed["n_local_trainings"]),
            "n_rounds": int(fed["n_rounds"]),
            "search_split": self.search_split,
        }
        for cid, acc in zip(self.client_ids_sorted, accs):
            rec[f"acc_client_{cid}"] = float(acc)
        self.generation_history.append(rec)
        # minimización: (-media, -mínimo)
        return np.array([-mean_acc, -min_acc], dtype=float), {
            "genome": genome,
            "mean_accuracy": mean_acc,
            "min_accuracy": min_acc,
            "mean_f1": mean_f1,
            "min_f1": min_f1,
            "per_client": per_client,
            "schedule": sched,
        }

    def evaluate_batch(self, X: np.ndarray):
        F = np.zeros((X.shape[0], 2))
        for i in range(X.shape[0]):
            f, _ = self.evaluate_single(X[i], individual_idx=i, arch_id=f"g{self.generation}_i{i}")
            F[i] = f
        return F
