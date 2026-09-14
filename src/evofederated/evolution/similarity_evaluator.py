"""Single-objective evaluator for similarity pairing with alternating representatives.

Fitness(x) = mean accuracy over representatives of current generation.
Each individual evaluated on exactly one representative per group (pair/singleton).

Design:
- Keep initialization deterministic per genome via hash seed, so both representatives
  see same initial weights for same genome (fair comparison, no contamination).
- No weight reuse between consecutive client evaluations (fresh model each time).
"""
import copy
import hashlib
import json
import time
from typing import List, Tuple

import numpy as np
import torch

from ..models.genome import Genome, genome_to_model, model_stats
from ..federated.characterization import flatten_params, set_model_params


def _seed_for_genome(genome: Genome) -> int:
    h = hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


class SimilarityEvaluator:
    def __init__(
        self,
        clients: dict,
        input_dim: int,
        n_classes: int,
        genome_config: dict,
        train_config: dict,
        pairs: List[Tuple[int, int]],
        singletons: List[int],
        client_ids_sorted: List[int],
        similarity_matrix: np.ndarray = None,
        device: str = "cpu",
        seed: int = 42,
        generation: int = 0,
        rep_mode: str = "alternating",
        full_mode: bool = False,
    ):
        self.clients = clients
        self.client_ids_sorted = client_ids_sorted  # e.g., [0,1,2,3,4,5,6,7]
        self.n_clients = len(client_ids_sorted)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.genome_config = genome_config
        self.train_config = train_config
        self.pairs = pairs  # list of (pos indices 0..N-1)
        self.singletons = singletons
        self.S = similarity_matrix
        self.device = device
        self.generation = generation
        self.rep_mode = rep_mode  # alternating | random
        self.full_mode = full_mode
        self.rng = np.random.RandomState(seed)
        # tracking
        self.eval_count = 0  # client-architecture trainings
        self.eval_time_total = 0.0
        self.generation_history = []  # per individual records
        self.per_generation_metrics = []  # aggregated per generation

    def set_generation(self, gen: int):
        self.generation = int(gen)

    def _get_representatives(self, generation: int = None) -> List[int]:
        if generation is None:
            generation = self.generation
        # full mode: all clients are representatives
        if self.full_mode:
            reps_pos = list(range(self.n_clients))
            reps_cids = [self.client_ids_sorted[p] for p in reps_pos]
            return reps_cids, reps_pos
        reps_pos = []
        if self.rep_mode == "alternating":
            for a, b in self.pairs:
                rep_pos = a if (generation % 2 == 0) else b
                reps_pos.append(int(rep_pos))
        elif self.rep_mode == "random":
            tmp_rng = np.random.RandomState((self.rng.randint(0, 2**31-1) + generation) % (2**31-1))
            for idx, (a, b) in enumerate(self.pairs):
                rep_pos = a if tmp_rng.rand() < 0.5 else b
                reps_pos.append(int(rep_pos))
        elif self.rep_mode == "probabilistic":
            # probabilistic representative: sample with P proportional to similarity? For now uniform 0.5 like random but with different RNG seed offset to be distinguishable.
            # In future could weight by similarity, but keep deterministic per generation.
            tmp_rng = np.random.RandomState((self.rng.randint(0, 2**31-1) + generation*997) % (2**31-1))
            for idx, (a, b) in enumerate(self.pairs):
                # For similarity, we could bias toward higher similarity intra-pair? but both members equivalent similaritywise;
                # So we keep 0.5 but deterministic per generation using different seed.
                rep_pos = a if tmp_rng.rand() < 0.5 else b
                reps_pos.append(int(rep_pos))
        else:
            # fallback alternating
            for a, b in self.pairs:
                rep_pos = a if (generation % 2 == 0) else b
                reps_pos.append(int(rep_pos))
        for s in self.singletons:
            reps_pos.append(int(s))
        # translate pos indices to actual client_ids
        reps_cids = [self.client_ids_sorted[p] for p in reps_pos]
        # also need pos mapping for logging
        return reps_cids, reps_pos

    def _train_and_eval(self, genome: Genome, client_id: int, flat_init: np.ndarray):
        """Fresh model with same init, train locally, eval on test."""
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
        eval_res = client.evaluate(model, device=self.device, split="test")
        return eval_res, train_res

    def evaluate_single(self, vec: np.ndarray, individual_idx: int, arch_id: str = None) -> Tuple[float, dict]:
        genome = Genome.from_vector(vec, self.genome_config)
        stats = model_stats(genome_to_model(genome, self.input_dim, self.n_classes))

        # deterministic init for this genome (same for all reps)
        seed = _seed_for_genome(genome)
        torch.manual_seed(seed)
        template = genome_to_model(genome, self.input_dim, self.n_classes)
        flat_init = flatten_params(template)

        reps_cids, reps_pos = self._get_representatives(self.generation)

        # evaluate on each representative
        accs = []
        f1s = []
        losses = []
        per_rep = {}
        total_time = 0.0
        for rep_pos, rep_cid in zip(reps_pos, reps_cids):
            eval_res, train_res = self._train_and_eval(genome, rep_cid, flat_init)
            accs.append(float(eval_res["accuracy"]))
            f1s.append(float(eval_res["macro_f1"]))
            losses.append(float(eval_res["loss"]))
            per_rep[int(rep_cid)] = {
                "accuracy": float(eval_res["accuracy"]),
                "macro_f1": float(eval_res["macro_f1"]),
                "loss": float(eval_res["loss"]),
                "n": int(eval_res["n"]),
                "train_time": float(train_res["time"]),
            }
            self.eval_count += 1
            self.eval_time_total += float(train_res["time"])
            total_time += float(train_res["time"])

        # fitness single objective: mean accuracy over representatives
        fitness = float(np.mean(accs)) if accs else 0.0
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        # For minimization, pymoo will use -fitness or 1-fitness; caller handles
        info = {
            "genome": genome,
            "genome_dict": genome.to_dict(),
            "genome_hash": hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()[:12],
            "representatives": reps_cids,
            "representatives_pos": reps_pos,
            "generation": int(self.generation),
            "individual_idx": int(individual_idx),
            "arch_id": arch_id,
            "accs": accs,
            "f1s": f1s,
            "losses": losses,
            "mean_accuracy": float(fitness),
            "mean_f1": float(mean_f1),
            "std_accuracy": float(np.std(accs)) if accs else 0.0,
            "per_rep": per_rep,
            "stats": stats,
            "elapsed": float(total_time),
        }
        # record per individual
        rec = {
            "generation": int(self.generation),
            "arch_id": arch_id,
            "individual_idx": int(individual_idx),
            "mean_accuracy": float(fitness),
            "mean_f1": float(mean_f1),
            "std_accuracy": float(np.std(accs)) if accs else 0.0,
            "min_accuracy": float(np.min(accs)) if accs else 0.0,
            "max_accuracy": float(np.max(accs)) if accs else 0.0,
            "worst_accuracy": float(np.min(accs)) if accs else 0.0,
            "representatives": str(reps_cids),
            "representatives_pos": str(reps_pos),
            "genome": json.dumps(genome.to_dict()),
            "genome_L": genome.L,
            "genome_hidden": str(genome.hidden_sizes),
            "genome_act": genome.activation,
            "genome_dropout": genome.dropout,
            "genome_bn": genome.use_bn,
            "genome_hash": info["genome_hash"],
            "n_params": stats["n_params"],
        }
        # add per-rep accuracies as columns
        for cid, vals in per_rep.items():
            rec[f"acc_rep_{cid}"] = vals["accuracy"]
            rec[f"f1_rep_{cid}"] = vals["macro_f1"]
        self.generation_history.append(rec)
        return fitness, info

    def evaluate_batch(self, X: np.ndarray):
        """For pymoo: X shape (n, n_var) -> F shape (n,1) minimization."""
        n = X.shape[0]
        F = np.zeros((n, 1))
        for i in range(n):
            vec = X[i]
            arch_id = f"g{self.generation}_i{i}"
            fitness, _ = self.evaluate_single(vec, individual_idx=i, arch_id=arch_id)
            # convert to minimization: 1 - accuracy or -accuracy. Use 1 - accuracy in [0,1]
            # For stability use -fitness also works. Use 1 - fitness to keep in [0,1]
            F[i, 0] = 1.0 - float(fitness)
        return F
