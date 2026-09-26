"""Family A: Dissimilarity + NSGA-II per pair.

For each pair Pi=(Ai,Bi):
  f1(x)=accuracy_Ai(x) , f2(x)=accuracy_Bi(x)  maximize -> minimize [1-acc]
NSGA-II with exactly 2 objectives. No averaging.

Isolated evaluation: each client gets fresh model with same deterministic init per genome.
"""
import hashlib, json
import numpy as np
import torch
from pymoo.core.problem import Problem

from ..models.genome import Genome, genome_to_model, model_stats
from ..federated.characterization import flatten_params, set_model_params

def _seed_for_genome(genome: Genome) -> int:
    h = hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()
    return int(h[:8],16) % (2**31-1)

class DissimilarityPairProblem(Problem):
    def __init__(self, n_var: int, evaluator):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0.0, xu=1.0)
        self.evaluator = evaluator
    def _evaluate(self, X, out, *args, **kwargs):
        F = self.evaluator.evaluate_batch(X)
        out["F"] = F

class PairEvaluator:
    """Evaluator for a single pair (Ai,Bi)."""
    def __init__(self, clients: dict, client_a_id: int, client_b_id: int,
                 input_dim: int, n_classes: int,
                 genome_config: dict, train_config: dict,
                 pair_id: int = 0, pair_tuple=(0,1),
                 device="cpu", seed=42, family="dissimilarity", pairing_strategy="deterministic_dissimilarity"):
        self.clients = clients
        self.client_a = client_a_id
        self.client_b = client_b_id
        self.pair_id = pair_id
        self.pair_tuple = pair_tuple
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.genome_config = genome_config
        self.train_config = train_config
        self.device = device
        self.generation = 0
        self.seed = seed
        self.family = family
        self.pairing_strategy = pairing_strategy
        self.eval_count = 0
        self.eval_time_total = 0.0
        self.generation_history = []  # per individual per generation
        self.full_eval_history = []  # full evaluation periodic

    def set_generation(self, gen: int):
        self.generation = int(gen)

    def _train_and_eval(self, genome: Genome, client_id: int, flat_init: np.ndarray):
        # debug device type
        if not isinstance(self.device, str):
            print(f"[DEBUG] device is not str: {type(self.device)} {self.device!r} train_config device {self.train_config.get('device')}")
            # force to cpu string
            dev = "cpu"
        else:
            dev = self.device
        model = genome_to_model(genome, self.input_dim, self.n_classes)
        set_model_params(model, flat_init)
        client = self.clients[client_id]
        train_res = client.train(model, epochs=self.train_config.get("epochs",5),
                                 lr=self.train_config.get("lr",1e-3),
                                 optimizer_name=self.train_config.get("optimizer","adam"),
                                 device=dev)
        # Búsqueda en validation; test reservado a evaluación final.
        eval_res = client.evaluate(model, device=dev, split="val")
        return eval_res, train_res

    def evaluate_single(self, vec: np.ndarray, individual_idx: int, arch_id: str=None):
        genome = Genome.from_vector(vec, self.genome_config)
        stats = model_stats(genome_to_model(genome, self.input_dim, self.n_classes))
        seed = _seed_for_genome(genome)
        torch.manual_seed(seed)
        template = genome_to_model(genome, self.input_dim, self.n_classes)
        flat_init = flatten_params(template)
        # evaluate on both clients of the pair
        accs = []
        losses=[]
        f1s=[]
        per_node={}
        total_time=0.0
        for cid in [self.client_a, self.client_b]:
            eval_res, train_res = self._train_and_eval(genome, cid, flat_init)
            acc = float(eval_res["accuracy"])
            loss = float(eval_res["loss"])
            f1 = float(eval_res["macro_f1"])
            accs.append(acc); losses.append(loss); f1s.append(f1)
            per_node[int(cid)] = {"accuracy":acc,"loss":loss,"f1":f1,"n":int(eval_res["n"]),"train_time":float(train_res["time"])}
            self.eval_count += 1
            self.eval_time_total += float(train_res["time"])
            total_time+=float(train_res["time"])
        # objectives minimize 1-acc
        F = np.array([1.0 - accs[0], 1.0 - accs[1]], dtype=float)
        # derived metrics
        mean_acc = float(np.mean(accs))
        min_acc = float(np.min(accs))
        max_acc = float(np.max(accs))
        gap = float(max_acc - min_acc)
        info = {
            "genome": genome, "genome_dict": genome.to_dict(),
            "genome_hash": hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()[:12],
            "acc_a": accs[0], "acc_b": accs[1], "loss_a": losses[0], "loss_b": losses[1],
            "f1_a": f1s[0], "f1_b": f1s[1],
            "per_node": per_node, "F": F,
            "mean_accuracy": mean_acc, "min_accuracy": min_acc, "max_accuracy": max_acc, "gap": gap,
            "stats": stats, "elapsed": float(total_time), "generation": int(self.generation),
        }
        rec = {
            "generation": int(self.generation),
            "pair_id": int(self.pair_id),
            "pair": str(self.pair_tuple),
            "client_a": int(self.client_a),
            "client_b": int(self.client_b),
            "individual_idx": int(individual_idx),
            "arch_id": arch_id,
            "genome": json.dumps(genome.to_dict()),
            "genome_hash": info["genome_hash"],
            "genome_L": genome.L, "genome_hidden": str(genome.hidden_sizes),
            "genome_act": genome.activation, "genome_dropout": genome.dropout, "genome_bn": genome.use_bn,
            "n_params": stats["n_params"],
            "accuracy_a": float(accs[0]), "accuracy_b": float(accs[1]),
            "f1_a": float(f1s[0]), "f1_b": float(f1s[1]),
            "loss_a": float(losses[0]), "loss_b": float(losses[1]),
            "mean_accuracy": mean_acc, "min_accuracy": min_acc, "max_accuracy": max_acc, "gap": gap,
            "obj_a": float(F[0]), "obj_b": float(F[1]),
            "family": self.family, "optimizer": "nsga2", "pairing_strategy": self.pairing_strategy,
        }
        self.generation_history.append(rec)
        return F, info

    def evaluate_batch(self, X: np.ndarray):
        n = X.shape[0]
        F = np.zeros((n,2))
        for i in range(n):
            vec = X[i]
            arch_id = f"pair{self.pair_id}_g{self.generation}_i{i}"
            f,_ = self.evaluate_single(vec, individual_idx=i, arch_id=arch_id)
            F[i]=f
        return F

    def evaluate_full_on_all_clients(self, genome: Genome, all_client_ids):
        """Full evaluation of a specific genome on all silos (for final matrices)."""
        seed = _seed_for_genome(genome)
        torch.manual_seed(seed)
        template = genome_to_model(genome, self.input_dim, self.n_classes)
        flat_init = flatten_params(template)
        results=[]
        for cid in all_client_ids:
            eval_res, train_res = self._train_and_eval(genome, cid, flat_init)
            results.append((cid, eval_res, train_res))
        return results
