"""Many-objective evaluator: N silos = N objetivos, maximize accuracy_i => minimize 1-accuracy_i."""
import copy
import hashlib
import json
import time
from typing import List, Dict

import numpy as np
import torch

from ..models.genome import Genome, genome_to_model, model_stats
from ..federated.characterization import flatten_params, set_model_params
from pymoo.core.problem import Problem


def _seed_for_genome(genome: Genome) -> int:
    h = hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


class ManyObjectiveEvaluator:
    """
    Evaluador many-objective: cada silo es un objetivo explícito.
    No promedia antes del optimizador.
    Internamente minimiza 1 - accuracy_i (si accuracy ∈[0,1] => objetivo ∈[0,1], 0 ideal).
    Transformación explícita documentada.
    Evaluación FULL por defecto (todos los silos) para preservar N objetivos.
    """
    def __init__(
        self,
        clients: dict,
        input_dim: int,
        n_classes: int,
        genome_config: dict,
        train_config: dict,
        device: str = "cpu",
        seed: int = 42,
        generation: int = 0,
    ):
        self.clients = clients
        self.client_ids_sorted = sorted(clients.keys())
        self.n_clients = len(clients)
        self.n_obj = self.n_clients
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.genome_config = genome_config
        self.train_config = train_config
        self.device = device
        self.generation = generation
        self.rng = np.random.RandomState(seed)
        self.eval_count = 0
        self.eval_time_total = 0.0
        self.generation_history = []  # per individual
        self.per_generation_metrics = []

    def set_generation(self, gen: int):
        self.generation = int(gen)

    def _train_and_eval(self, genome: Genome, client_id: int, flat_init: np.ndarray):
        model = genome_to_model(genome, self.input_dim, self.n_classes)
        set_model_params(model, flat_init)
        client = self.clients[client_id]
        train_res = client.train(
            model,
            epochs=self.train_config.get("epochs", 5),
            lr=self.train_config.get("lr", 1e-3),
            optimizer_name=self.train_config.get("optimizer", "adam"),
            device=self.device,
        )
        # Búsqueda en validation; test reservado a evaluación final.
        eval_res = client.evaluate(model, device=self.device, split="val")
        return eval_res, train_res

    def evaluate_single(self, vec: np.ndarray, individual_idx: int, arch_id: str = None):
        genome = Genome.from_vector(vec, self.genome_config)
        stats = model_stats(genome_to_model(genome, self.input_dim, self.n_classes))
        seed = _seed_for_genome(genome)
        torch.manual_seed(seed)
        template = genome_to_model(genome, self.input_dim, self.n_classes)
        flat_init = flatten_params(template)

        # Full evaluation: train+evaluate on all silos independently with same init
        accuracies = []
        losses = []
        f1s = []
        per_client = {}
        total_time = 0.0
        for cid in self.client_ids_sorted:
            # need fresh copy with same init for each client
            eval_res, train_res = self._train_and_eval(genome, cid, flat_init)
            acc = float(eval_res["accuracy"])
            loss = float(eval_res["loss"])
            f1 = float(eval_res["macro_f1"])
            accuracies.append(acc)
            losses.append(loss)
            f1s.append(f1)
            per_client[int(cid)] = {"accuracy": acc, "loss": loss, "f1": f1, "n": int(eval_res["n"]), "train_time": float(train_res["time"])}
            self.eval_count += 1
            self.eval_time_total += float(train_res["time"])
            total_time += float(train_res["time"])

        # Objectives: minimize 1 - acc per silo (range 0..1)
        # Explicit transform: f_i = 1 - accuracy_i
        F = np.array([1.0 - acc for acc in accuracies], dtype=float)

        # aggregated metrics (for reporting not optimization)
        mean_acc = float(np.mean(accuracies))
        median_acc = float(np.median(accuracies))
        worst_acc = float(np.min(accuracies))
        best_acc = float(np.max(accuracies))
        std_acc = float(np.std(accuracies))
        gap = float(best_acc - worst_acc)

        info = {
            "genome": genome,
            "genome_dict": genome.to_dict(),
            "genome_hash": hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()[:12],
            "accuracies": accuracies,
            "losses": losses,
            "f1s": f1s,
            "per_client": per_client,
            "F": F,
            "mean_accuracy": mean_acc,
            "median_accuracy": median_acc,
            "worst_accuracy": worst_acc,
            "best_accuracy": best_acc,
            "std_accuracy": std_acc,
            "gap": gap,
            "stats": stats,
            "elapsed": float(total_time),
            "generation": int(self.generation),
        }

        rec = {
            "generation": int(self.generation),
            "arch_id": arch_id,
            "individual_idx": int(individual_idx),
            "genome": json.dumps(genome.to_dict()),
            "genome_hash": info["genome_hash"],
            "genome_L": genome.L,
            "genome_hidden": str(genome.hidden_sizes),
            "genome_act": genome.activation,
            "genome_dropout": genome.dropout,
            "genome_bn": genome.use_bn,
            "n_params": stats["n_params"],
            "mean_accuracy": mean_acc,
            "median_accuracy": median_acc,
            "worst_accuracy": worst_acc,
            "best_accuracy": best_acc,
            "std_accuracy": std_acc,
            "gap": gap,
        }
        for idx, cid in enumerate(self.client_ids_sorted):
            rec[f"acc_client_{cid}"] = float(accuracies[idx])
            rec[f"f1_client_{cid}"] = float(f1s[idx])
            rec[f"loss_client_{cid}"] = float(losses[idx])
            rec[f"obj_{cid}"] = float(F[idx])  # 1 - acc
        self.generation_history.append(rec)
        return F, info

    def evaluate_batch(self, X: np.ndarray):
        n = X.shape[0]
        F = np.zeros((n, self.n_obj))
        for i in range(n):
            vec = X[i]
            arch_id = f"g{self.generation}_i{i}"
            f, _ = self.evaluate_single(vec, individual_idx=i, arch_id=arch_id)
            F[i] = f
        return F


class ManyObjectiveProblem(Problem):
    def __init__(self, n_var: int, evaluator: ManyObjectiveEvaluator):
        super().__init__(n_var=n_var, n_obj=evaluator.n_obj, n_constr=0, xl=0.0, xu=1.0)
        self.evaluator = evaluator

    def _evaluate(self, X, out, *args, **kwargs):
        F = self.evaluator.evaluate_batch(X)
        out["F"] = F
