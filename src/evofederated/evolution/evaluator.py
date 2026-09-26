"""Evaluator: decodifica genomas, entrena/evalúa en clientes, produce objetivos.
Maneja contaminación, pairing, contadores de evaluaciones."""
import copy
import time
import numpy as np
import torch
import torch.nn as nn

from ..models.genome import Genome, genome_to_model, model_stats
from ..federated.pairing import PairingStrategy


class Evaluator:
    def __init__(
        self,
        clients: dict,
        input_dim: int,
        n_classes: int,
        genome_config: dict,
        train_config: dict,
        pairing_strategy: PairingStrategy,
        divergence_matrix: np.ndarray = None,
        strategy_name: str = "dynamic",
        seed: int = 42,
        device: str = "cpu",
        generation: int = 0,
    ):
        self.clients = clients
        self.client_ids_sorted = sorted(clients.keys())
        self.n_clients = len(clients)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.genome_config = genome_config
        self.train_config = train_config
        self.pairing = pairing_strategy
        self.D = divergence_matrix
        self.strategy_name = strategy_name.lower()
        self.device = device
        self.generation = generation
        self.rng = np.random.RandomState(seed)
        # tracking
        self.eval_count = 0  # number of client-architecture trainings
        self.eval_time_total = 0.0
        self.pair_history = []  # list of dicts
        self.pop_history = []  # list of dicts
        # For balancing reference client assignment
        self._next_ref_idx = 0

    def set_generation(self, gen: int):
        self.generation = gen

    def _get_ref_client_idx(self, individual_idx: int) -> int:
        # balanced round-robin: i % n_clients
        # but we want reproducible and simple
        # Use individual_idx mapping plus generation offset?
        # Spec: "La asignación puede distribuirse de manera balanceada"
        # Use deterministic round robin
        return individual_idx % self.n_clients

    def evaluate_single(self, vec: np.ndarray, individual_idx: int, arch_id: str = None) -> tuple:
        """
        Evalúa genoma vec. Retorna (F minimization vector shape (2,), info dict).
        """
        genome = Genome.from_vector(vec, self.genome_config)

        # stats
        model_template = genome_to_model(genome, self.input_dim, self.n_classes)
        stats = model_stats(model_template)

        # select reference client
        ref_pos = self._get_ref_client_idx(individual_idx)
        ref_cid = self.client_ids_sorted[ref_pos]

        # select counterpart(s) based on strategy
        if self.strategy_name == "full":
            # evaluate on all clients
            # need to produce 2 objectives: mean and min F1
            f1s = []
            accs = []
            losses = []
            for cid in self.client_ids_sorted:
                f1, acc, loss, elapsed = self._train_and_eval(genome, cid)
                f1s.append(f1)
                accs.append(acc)
                losses.append(loss)
                self.eval_count += 1
                self.eval_time_total += elapsed
            # objectives: -mean_F1, -min_F1 (both minimization)
            f1s_arr = np.array(f1s)
            mean_f1 = float(f1s_arr.mean())
            min_f1 = float(f1s_arr.min())
            median_f1 = float(np.median(f1s_arr))
            std_f1 = float(f1s_arr.std())
            # For FULL, also track pair history? No pair; log aggregated
            info = {
                "genome": genome,
                "genome_dict": genome.to_dict(),
                "ref_client": ref_cid,
                "counterpart": None,
                "D_ij": None,
                "strategy": self.strategy_name,
                "f1s_all": f1s,
                "accs_all": accs,
                "mean_f1": mean_f1,
                "min_f1": min_f1,
                "median_f1": median_f1,
                "std_f1": std_f1,
                "f1_i": mean_f1,  # for compatibility, treat mean as f1_i
                "f1_j": min_f1,   # and min as f1_j
                "stats": stats,
                "elapsed": self.eval_time_total,
                "eval_count": self.eval_count,
            }
            # minimization objectives
            F = np.array([-mean_f1, -min_f1])
            # pair_history empty for full? Add entry with counterpart = all
            self.pair_history.append({
                "generation": self.generation,
                "arch_id": arch_id,
                "individual_idx": individual_idx,
                "ref": ref_cid,
                "counterpart": -1,
                "D_ij": 0.0,
                "strategy": self.strategy_name,
                "F": F.tolist(),
            })
            self.pop_history.append({
                "generation": self.generation,
                "arch_id": arch_id,
                "individual_idx": individual_idx,
                "F1_mean": mean_f1,
                "F1_min": min_f1,
                "F1_median": median_f1,
                "F": F.tolist(),
                "n_params": stats["n_params"],
                "genome": genome.to_dict(),
            })
            return F, info

        else:
            # 2-client evaluation
            # Select counterpart
            if self.D is not None:
                # D is indexed by positions 0..N-1 corresponding to sorted client_ids
                counterpart_pos = self.pairing.select_counterpart(ref_pos, self.D, self.rng)
            else:
                # fallback random without D
                from ..federated.pairing import RandomPairing
                counterpart_pos = RandomPairing().select_counterpart(ref_pos, np.zeros((self.n_clients, self.n_clients)), self.rng)

            counterpart_cid = self.client_ids_sorted[counterpart_pos]
            D_ij = float(self.D[ref_pos, counterpart_pos]) if self.D is not None else 0.0

            # Train/eval on ref
            f1_i, acc_i, loss_i, elapsed_i = self._train_and_eval(genome, ref_cid)
            self.eval_count += 1
            self.eval_time_total += elapsed_i
            # Train/eval on counterpart with fresh weights
            f1_j, acc_j, loss_j, elapsed_j = self._train_and_eval(genome, counterpart_cid)
            self.eval_count += 1
            self.eval_time_total += elapsed_j

            F = np.array([-f1_i, -f1_j])

            info = {
                "genome": genome,
                "genome_dict": genome.to_dict(),
                "ref_client": ref_cid,
                "counterpart": counterpart_cid,
                "D_ij": D_ij,
                "strategy": self.strategy_name,
                "f1_i": float(f1_i),
                "f1_j": float(f1_j),
                "acc_i": float(acc_i),
                "acc_j": float(acc_j),
                "loss_i": float(loss_i),
                "loss_j": float(loss_j),
                "stats": stats,
                "elapsed": elapsed_i + elapsed_j,
            }

            self.pair_history.append({
                "generation": self.generation,
                "arch_id": arch_id,
                "individual_idx": individual_idx,
                "ref": ref_cid,
                "counterpart": counterpart_cid,
                "D_ij": D_ij,
                "strategy": self.strategy_name,
                "F": F.tolist(),
            })
            self.pop_history.append({
                "generation": self.generation,
                "arch_id": arch_id,
                "individual_idx": individual_idx,
                "ref": ref_cid,
                "counterpart": counterpart_cid,
                "F1_i": float(f1_i),
                "F1_j": float(f1_j),
                "F": F.tolist(),
                "n_params": stats["n_params"],
                "genome": genome.to_dict(),
            })

            return F, info

    def _train_and_eval(self, genome: Genome, client_id: int):
        """
        Crea fresh model, entrena localmente cliente, evalúa en test.
        Returns F1, acc, loss, elapsed.
        Crucial: no reuse weights.
        """
        client = self.clients[client_id]
        model = genome_to_model(genome, self.input_dim, self.n_classes)
        # ensure fresh initialization controlled by global seed + arch-specific? 
        # We keep PyTorch default init but ensure reproducibility via seed sequence.
        # To avoid contamination, we set seed based on generation + client + genome hash?
        # Simpler: use current RNG but ensure model is freshly created each time.
        # Use same initial weights for both clients for same architecture?
        # Spec: "Ambas evaluaciones deben partir de condiciones iniciales controladas y comparables. 
        # No reutilices accidentalmente los pesos entrenados en C_i como inicialización para C_j."
        # Interpretation: for same architecture, both clients should start from same initialization.
        # So we need deterministic initialization per architecture, not per client.
        # Approach: set torch seed based on hash of genome dict + global seed, then recreate model.
        # But we already created model; so we can reseed and reinitialize deterministically.
        # Simpler: capture initial state after creation, then for second evaluation reuse same initial flat?
        # We'll implement caller-level handling by passing precomputed initial flat if needed.
        # For now, each call creates fresh model with same architecture but random init.
        # To guarantee comparability, we will use a deterministic seed per genome.
        # However this would require computing seed from genome; we do that here.
        
        # Deterministic init per genome: use hash of genome.to_dict() + global eval seed
        # To keep independence between two evaluations of same genome, they should start from same init.
        # We can achieve by seeding before model creation inside evaluate_single before calling _train_and_eval.
        # For simplicity, we'll seed inside evaluate_single and create two models from same seed.
        # But _train_and_eval is called sequentially; second call will have different RNG state.
        # So we need to handle outside.
        # Workaround: In evaluate_single we could generate initial flat once and set for both.
        # For now, we implement internal determinism via genome-aware seeding.
        # Let's compute a seed offset from genome
        import hashlib, json
        gdict = genome.to_dict()
        # stable hash
        h = hashlib.md5(json.dumps(gdict, sort_keys=True).encode()).hexdigest()
        seed_offset = int(h[:8], 16) % (2**31 - 1)
        # Use base seed 42 + offset, but keep generation independent? We'll use offset only
        torch.manual_seed(seed_offset % (2**31 - 1))
        # Re-create model with this seed
        model = genome_to_model(genome, self.input_dim, self.n_classes)

        # Now train
        train_res = client.train(
            model,
            epochs=self.train_config.get("epochs", 5),
            lr=self.train_config.get("lr", 1e-3),
            optimizer_name=self.train_config.get("optimizer", "adam"),
            device=self.device,
        )
        # Búsqueda en validation; test reservado a evaluación final.
        eval_res = client.evaluate(model, device=self.device, split="val")
        return eval_res["macro_f1"], eval_res["accuracy"], eval_res["loss"], train_res["time"]

    def evaluate_batch(self, X: np.ndarray):
        """For pymoo: evaluate matrix X -> F matrix."""
        n = X.shape[0]
        F = np.zeros((n, 2))
        # We need arch_id generation
        for i in range(n):
            vec = X[i]
            arch_id = f"g{self.generation}_i{i}"
            f, _ = self.evaluate_single(vec, individual_idx=i, arch_id=arch_id)
            F[i] = f
        return F

    def evaluate_single_deterministic_pair(self, vec: np.ndarray, ref_cid: int, counterpart_cid: int):
        """Helper for exhaustive validation: evaluate genome on two specific clients with same init."""
        genome = Genome.from_vector(vec, self.genome_config)
        # generate deterministic init
        import hashlib, json
        gdict = genome.to_dict()
        h = hashlib.md5(json.dumps(gdict, sort_keys=True).encode()).hexdigest()
        seed_offset = int(h[:8], 16) % (2**31 - 1)
        torch.manual_seed(seed_offset % (2**31 - 1))
        model_i = genome_to_model(genome, self.input_dim, self.n_classes)
        # capture initial flat
        from ..federated.characterization import flatten_params, set_model_params
        flat_init = flatten_params(model_i)
        # eval ref
        client_i = self.clients[ref_cid]
        # Use copy with same init
        model_copy_i = copy.deepcopy(model_i)
        train_res_i = client_i.train(model_copy_i, epochs=self.train_config.get("epochs",5), lr=self.train_config.get("lr",1e-3), optimizer_name=self.train_config.get("optimizer","adam"), device=self.device)
        eval_i = client_i.evaluate(model_copy_i, device=self.device, split="val")
        # eval counterpart with same init
        model_copy_j = genome_to_model(genome, self.input_dim, self.n_classes)
        set_model_params(model_copy_j, flat_init)
        client_j = self.clients[counterpart_cid]
        train_res_j = client_j.train(model_copy_j, epochs=self.train_config.get("epochs",5), lr=self.train_config.get("lr",1e-3), optimizer_name=self.train_config.get("optimizer","adam"), device=self.device)
        eval_j = client_j.evaluate(model_copy_j, device=self.device, split="val")
        return eval_i, eval_j
