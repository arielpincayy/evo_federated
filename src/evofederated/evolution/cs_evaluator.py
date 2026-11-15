"""Evaluador por muestreo estratificado: un representante al azar por cluster y por generación.

Fitness de búsqueda (minimización), según `objectives`:
  mean_params      (-acc_estimada, n_params / n_params_max); acc_estimada = sum_c W_c * acc_val(eval_c), W_c = |cluster c| / N.
  per_node         (-acc_val(eval_1), ..., -acc_val(eval_C)): un objetivo por cluster = accuracy del nodo evaluador de ese cluster.
  per_cluster_size (-acc_val(eval_1), ..., -acc_val(eval_C), n_params/param_ref): un objetivo por cluster y el tamaño como objetivo propio (C+1).
  per_cluster_mean_size (-m_1, ..., -m_C, n_params/param_ref): lo anterior con la media del cluster (todos los miembros que no entrenaron) y el tamaño como objetivo propio.
  per_cluster_mean (-m_1, ..., -m_C), m_c = media de la accuracy en TODOS los miembros del cluster c que no entrenaron
                   (solo inferencia; ablación de menor ruido que relaja la premisa de no evaluar a todos los nodos).
En todos los modos se registra `obj_acc` (accuracies por objetivo), `acc_rep` (su media ponderada) y el nº de parámetros.

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
        objectives: str = "mean_params",  # mean_params | per_node | per_cluster_size | per_cluster_mean_size | per_cluster_mean
        size_penalty: float = 0.0,  # solo modos por cluster: obj_c = acc_c - size_penalty * n_params/param_ref
        cache: bool = False,  # MOEA v2: una arquitectura decodificada ya evaluada reutiliza su fitness y no entrena
        reps_per_cluster: int = 1,  # MOEA v3: r representantes por cluster (solo sampler=cluster); con 1 el comportamiento es el original
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
        self.reps_per_cluster = int(reps_per_cluster)
        if self.labels is not None:  # nodo -> índice de su cluster
            ucl = list(np.unique(self.labels))
            self._cluster_of = {self.ids[i]: ucl.index(self.labels[i]) for i in range(len(self.ids))}
        self.sampler, self.n_reps = sampler, int(n_reps)
        self.seed = int(seed)
        self.init_seed = self.seed if init_seed is None else int(init_seed)
        self.device, self.monitor, self.eval_holdout = device, bool(monitor), bool(eval_holdout)
        if objectives not in ("mean_params", "per_node", "per_cluster_size", "per_cluster_mean_size", "per_cluster_mean"):
            raise ValueError(f"objectives desconocido: {objectives}")
        if objectives in ("per_cluster_mean", "per_cluster_mean_size"):
            assert sampler == "cluster", f"{objectives} requiere sampler=cluster"
        assert objectives not in ("mean_params", "per_cluster_size", "per_cluster_mean_size") or size_penalty == 0, "size_penalty solo aplica a per_node/per_cluster_mean (en mean_params y per_cluster_size el tamaño ya es un objetivo)"
        self.objectives, self.size_penalty = objectives, float(size_penalty)
        self.cache, self._cache, self.n_cache_hits = bool(cache), {}, 0
        self.n_objective_node_evals = 0
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
        if self.reps_per_cluster > 1:  # r representantes por cluster; peso |G_c|/(r·N) cada uno; un evaluador retenido por cluster entre los que no entrenaron
            reps, evals, rep_w, picks, pools = [], [], [], [], []
            for c in np.unique(self.labels):
                members = [self.ids[i] for i in np.where(self.labels == c)[0] if self.ids[i] in valid]
                if members:
                    picks.append([int(x) for x in rng.choice(members, size=min(self.reps_per_cluster, len(members)), replace=False)])
                    pools.append(members)
            total = float(sum(len(m) for m in pools))
            for pick, members in zip(picks, pools):
                reps += pick
                rep_w += [len(members) / total / len(pick)] * len(pick)
                others = [m for m in members if m not in set(pick)]
                evals.append(int(rng.choice(others)) if self.eval_holdout and others else pick[0])
            return reps, evals, rep_w
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

    def n_objectives(self) -> int:
        """Nº de objetivos del vector de fitness (2 en mean_params; C en los modos por cluster)."""
        if self.objectives == "mean_params":
            return 2
        return len(self.assignment(0)[1]) + (1 if self.objectives in ("per_cluster_size", "per_cluster_mean_size") else 0)

    @property
    def n_acc_obj(self):
        """Nº de objetivos de accuracy (los primeros de obj_acc); None en mean_params."""
        return None if self.objectives == "mean_params" else len(self.assignment(0)[1])  # nodos evaluadores = clusters

    def _cluster_mean_acc(self, model, reps: List[int]) -> List[float]:
        """Accuracy media en val de los miembros de cada cluster que no entrenaron (si no hay otro, el propio representante)."""
        valid, out, groups = set(self.valid), [], []
        for ci, c in enumerate(np.unique(self.labels)):
            trainers = {r for r in reps if self._cluster_of[r] == ci}  # todos los representantes del cluster (uno con r = 1)
            members = [self.ids[i] for i in np.where(self.labels == c)[0] if self.ids[i] in valid and self.ids[i] not in trainers]
            groups.append(members or sorted(trainers))
        nodes = sorted({m for g_ in groups for m in g_})
        res = evaluate_global_on_split(model, {m: self.clients[m] for m in nodes}, split="val", device=self.device)
        self.n_objective_node_evals += len(nodes)
        return [float(np.mean([res[m]["accuracy"] for m in g_])) for g_ in groups]

    def _cluster_w(self, reps, w):
        """Peso de cada cluster = suma de los pesos de sus representantes (|G_c|/N)."""
        out = [0.0] * len(np.unique(self.labels))
        for r, wi in zip(reps, w):
            out[self._cluster_of[r]] += wi
        return out

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
        F = np.zeros((len(X), self.n_objectives()))
        for i, vec in enumerate(X):
            g = CNNGenome.from_vector(vec, self.space)
            gj = json.dumps(g.to_dict())
            if self.cache and gj in self._cache:  # arquitectura ya evaluada: se reutiliza su fitness (sin entrenar ni registrar)
                F[i] = self._cache[gj]
                self.n_cache_hits += 1
                continue
            t0 = time.time()
            fed = self._train(g, reps, w, self.init_seed)
            sub = {e: self.clients[e] for e in set(evals)}
            per_eval = evaluate_global_on_split(fed["model"], sub, split="val", device=self.device)
            obj = [float(per_eval[e]["accuracy"]) for e in evals]
            if self.objectives in ("per_cluster_mean", "per_cluster_mean_size"):
                obj = self._cluster_mean_acc(fed["model"], reps)
            cw = self._cluster_w(reps, w) if (self.sampler == "cluster" and self.reps_per_cluster > 1) else w  # obj tiene un valor por cluster
            acc = float(sum(wi * oi for oi, wi in zip(obj, cw)))
            npar = n_params(fed["model"])
            obj_raw = list(obj)
            if self.size_penalty:  # obj_acc = vector que optimiza la búsqueda (penalizado); acc_rep y obj_raw quedan sin penalizar
                obj = [o - self.size_penalty * npar / self.param_ref for o in obj]
            if self.objectives in ("per_cluster_size", "per_cluster_mean_size"):  # el tamaño es un objetivo más; convención: todo se maximiza y F = -obj_acc
                obj = obj_raw + [-npar / self.param_ref]
            rec = {"generation": int(generation), "individual_idx": i, "genome": gj,
                   "vector": json.dumps([float(x) for x in vec]),
                   "acc_rep": acc, "obj_acc": json.dumps(obj), "obj_raw": json.dumps(obj_raw), "n_params": npar, "n_params_norm": npar / self.param_ref,
                   "reps": json.dumps(reps), "eval_nodes": json.dumps(evals),
                   "n_eval_fallback": int(sum(e in set(reps) for e in evals)), "n_local_trainings": int(fed["n_local_trainings"])}
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
            F[i] = (-acc, npar / self.param_ref) if self.objectives == "mean_params" else -np.array(obj)
            if self.cache:
                self._cache[gj] = F[i].copy()
        return F

    def evaluate_test(self, genome: CNNGenome, init_seed: int, participation: str = "full", return_nodes: bool = False):
        """Evaluación final fuera de la búsqueda: reentrena desde cero con otra semilla y mide en `test`.

        Con `return_nodes` devuelve además la lista (nodo, accuracy, n) por nodo."""
        if participation == "full":
            reps = list(self.valid)
            w = [1.0 / len(reps)] * len(reps)
        else:
            reps, w = self.representatives(10_000_000 + init_seed)
        fed = self._train(genome, reps, w, init_seed)
        per = evaluate_global_on_split(fed["model"], self.clients, split="test", device=self.device)
        res = _summ(per)
        res["n_params"] = n_params(fed["model"])
        res["n_local_trainings"] = int(fed["n_local_trainings"])
        if return_nodes:
            return res, [(int(c), v["accuracy"], v["n"]) for c, v in sorted(per.items())]
        return res
