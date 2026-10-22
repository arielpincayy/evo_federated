"""Evaluador por muestreo estratificado: un representante al azar por cluster y por generación.

Fitness de búsqueda (minimización): (-acc_estimada, n_params / n_params_max).
  acc_estimada = sum_c W_c * acc_val(eval_c), W_c = |cluster c| / N  (estimador estratificado de la accuracy media por nodo).

Los representantes se sortean con un RNG derivado de (seed, generación): todos los individuos de una generación
(padres e hijos) se evalúan con los mismos representantes. La búsqueda solo toca `val` de representantes y evaluadores;
la accuracy se mide en un nodo evaluador del mismo cluster que no entrenó (`eval_holdout`);
`monitor` evalúa además el mismo modelo en todos los nodos solo para reporte y nunca influye en el fitness.
"""
import copy
import hashlib
import json
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from ..federated.characterization import flatten_params
from ..federated.fedavg import evaluate_global_on_split, run_fedavg
from ..models.cnn_genome import CNNGenome, CNNSpace, cnn_to_model, n_params


def seed_for_genome(genome: CNNGenome, init_seed: int) -> int:
    h = hashlib.md5((json.dumps(genome.to_dict(), sort_keys=True) + f"|{init_seed}").encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def _summ(per_client: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    a = np.array([v["accuracy"] for v in per_client.values()])
    f = np.array([v["macro_f1"] for v in per_client.values()])
    n = np.array([v["n"] for v in per_client.values()], dtype=float)
    return {"acc_mean": float(a.mean()), "acc_pooled": float((a * n).sum() / max(n.sum(), 1.0)),
            "acc_worst": float(a.min()), "f1_mean": float(f.mean())}


class ClusterSampleEvaluator:
    def __init__(
        self,
        clients: dict,
        input_shape,
        n_classes: int,
        space: CNNSpace,
        train_cfg: dict,
        fed_cfg: dict,
        param_ref: int,
        labels: Optional[np.ndarray] = None,
        sampler: str = "cluster",  # cluster | random | full
        n_reps: int = 10,  # solo para sampler=random
        seed: int = 42,
        init_seed: Optional[int] = None,
        device: str = "cpu",
        monitor: bool = True,
        eval_holdout: bool = True,
    ):
        self.clients = clients
        self.ids = sorted(clients)
        self.valid = [c for c in self.ids if clients[c].n_val > 0 and clients[c].n_train > 0]
        self.input_shape, self.n_classes, self.space = tuple(input_shape), int(n_classes), space
        self.train_cfg, self.fed = train_cfg, dict(fed_cfg)
        self.param_ref = int(param_ref)
        if sampler == "cluster":
            assert labels is not None and len(labels) == len(self.ids), "sampler=cluster requiere una etiqueta por nodo"
        self.labels = None if labels is None else np.asarray(labels)
        self.sampler, self.n_reps = sampler, int(n_reps)
        self.seed = int(seed)
        self.init_seed = self.seed if init_seed is None else int(init_seed)
        self.device, self.monitor, self.eval_holdout = device, bool(monitor), bool(eval_holdout)
        self.history: List[dict] = []
        self.n_evals = 0
        self.n_local_trainings = 0
        self.n_monitor_node_evals = 0
        self.train_time = 0.0

    def _model_fn(self, g, _input_dim, n_classes):
        return cnn_to_model(g, self.input_shape, n_classes)

    # --- representantes -------------------------------------------------
    def assignment(self, generation: int):
        """(reps, eval_nodes, pesos) de la generación; iguales para toda la población. Pesos suman 1.

        reps entrenan el modelo; eval_nodes lo evalúan (solo inferencia). Con `eval_holdout`, el nodo evaluador es
        otro miembro del mismo cluster que no entrenó (evita el sesgo optimista de evaluar en datos ya vistos);
        si el cluster tiene un solo nodo válido (o sampler=full), el evaluador es el propio representante.
        """
        rng = np.random.RandomState((self.seed * 1_000_003 + int(generation)) % (2**32))
        valid = set(self.valid)
        if self.sampler == "full":
            return list(self.valid), list(self.valid), [1.0 / len(self.valid)] * len(self.valid)
        if self.sampler == "random":
            reps = sorted(int(x) for x in rng.choice(self.valid, size=min(self.n_reps, len(self.valid)), replace=False))
            rest = [c for c in self.valid if c not in set(reps)]
            ev = list(reps)
            if self.eval_holdout and len(rest) >= len(reps):
                ev = sorted(int(x) for x in rng.choice(rest, size=len(reps), replace=False))
            return reps, ev, [1.0 / len(reps)] * len(reps)
        reps, evals, sizes = [], [], []
        pools = []
        for c in np.unique(self.labels):
            members = [self.ids[i] for i in np.where(self.labels == c)[0] if self.ids[i] in valid]
            if members:
                reps.append(int(rng.choice(members)))
                pools.append(members)
                sizes.append(len(members))
        for r, members in zip(reps, pools):  # 2ª pasada: los representantes no cambian respecto a sin holdout
            others = [m for m in members if m != r]
            evals.append(int(rng.choice(others)) if self.eval_holdout and others else r)
        w = np.array(sizes, dtype=float)
        return reps, evals, (w / w.sum()).tolist()

    def representatives(self, generation: int):
        reps, _, w = self.assignment(generation)
        return reps, w

    # --- evaluación -----------------------------------------------------
    def _train(self, genome: CNNGenome, reps: List[int], weights: List[float], init_seed: int):
        torch.manual_seed(seed_for_genome(genome, init_seed))
        flat_init = flatten_params(cnn_to_model(genome, self.input_shape, self.n_classes))
        fed = run_fedavg(
            genome, self._model_fn, self.clients, None, self.n_classes, flat_init,
            [list(reps)] * int(self.fed.get("rounds", 1)),
            local_epochs=int(self.fed.get("local_epochs", 1)),
            lr=float(self.train_cfg.get("lr", 1e-3)),
            optimizer_name=str(self.train_cfg.get("optimizer", "adam")),
            device=self.device,
            weight_override=dict(zip(reps, weights)) if self.fed.get("weight_by_cluster", True) else None,
        )
        return fed

    def evaluate_population(self, X: np.ndarray, generation: int) -> np.ndarray:
        reps, evals, w = self.assignment(generation)
        F = np.zeros((len(X), 2))
        for i, vec in enumerate(X):
            g = CNNGenome.from_vector(vec, self.space)
            t0 = time.time()
            fed = self._train(g, reps, w, self.init_seed)
            sub = {e: self.clients[e] for e in set(evals)}
            per_eval = evaluate_global_on_split(fed["model"], sub, split="val", device=self.device)
            acc = float(sum(wi * per_eval[e]["accuracy"] for e, wi in zip(evals, w)))
            npar = n_params(fed["model"])
            rec = {"generation": int(generation), "individual_idx": i, "genome": json.dumps(g.to_dict()),
                   "vector": json.dumps([float(x) for x in vec]),
                   "acc_rep": acc, "n_params": npar, "n_params_norm": npar / self.param_ref,
                   "reps": json.dumps(reps), "eval_nodes": json.dumps(evals),
                   "n_eval_fallback": int(sum(e == r for e, r in zip(evals, reps))), "n_local_trainings": int(fed["n_local_trainings"])}
            if self.monitor:  # solo reporte: mismo modelo, todos los nodos (inferencia), jamás entra al fitness
                allc = evaluate_global_on_split(fed["model"], self.clients, split="val", device=self.device)
                rec.update({f"mon_{k}": v for k, v in _summ(allc).items()})
                if self.labels is not None:
                    a = np.array([allc[c]["accuracy"] for c in self.ids])
                    rec["mon_acc_by_cluster"] = json.dumps([float(a[self.labels == c].mean()) for c in np.unique(self.labels)])
                self.n_monitor_node_evals += len(allc)
            self.n_evals += 1
            self.n_local_trainings += int(fed["n_local_trainings"])
            self.train_time += time.time() - t0
            self.history.append(rec)
            F[i] = (-acc, npar / self.param_ref)
        return F

    def evaluate_test(self, genome: CNNGenome, init_seed: int, participation: str = "full") -> Dict[str, float]:
        """Evaluación final fuera de la búsqueda: reentrena desde cero con otra semilla y mide en `test`."""
        if participation == "full":
            reps = list(self.valid)
            w = [1.0 / len(reps)] * len(reps)
        else:
            reps, w = self.representatives(10_000_000 + init_seed)
        fed = self._train(genome, reps, w, init_seed)
        res = _summ(evaluate_global_on_split(fed["model"], self.clients, split="test", device=self.device))
        res["n_params"] = n_params(fed["model"])
        res["n_local_trainings"] = int(fed["n_local_trainings"])
        return res
