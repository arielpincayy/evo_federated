"""Exhaustive validation, global Pareto, Hall of Fame — minimal ponytail impl."""
import hashlib
import json
import copy
import numpy as np
import pandas as pd
import torch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV

from ..models.genome import Genome, genome_to_model
from ..federated.characterization import flatten_params, set_model_params


def genome_hash(genome: Genome) -> str:
    return hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def exhaustive_evaluate_genome(genome: Genome, clients, input_dim, n_classes, train_cfg, device="cpu", split: str = "test"):
    """Evaluate one genome on all clients independently from same init. Returns dict metrics."""
    # deterministic init per genome
    h = hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()
    seed_offset = int(h[:8], 16) % (2**31 - 1)
    torch.manual_seed(seed_offset % (2**31 - 1))
    template = genome_to_model(genome, input_dim, n_classes)
    flat_init = flatten_params(template)
    f1s = []
    accs = []
    losses = []
    for cid in sorted(clients.keys()):
        model = genome_to_model(genome, input_dim, n_classes)
        set_model_params(model, flat_init)
        client = clients[cid]
        client.train(model, epochs=train_cfg.get("epochs", 5), lr=train_cfg.get("lr", 1e-3),
                     optimizer_name=train_cfg.get("optimizer", "adam"), device=device)
        res = client.evaluate(model, device=device, split=split)
        f1s.append(float(res["macro_f1"]))
        accs.append(float(res["accuracy"]))
        losses.append(float(res["loss"]))
    f1s_arr = np.array(f1s)
    accs_arr = np.array(accs)
    return {
        "f1s": f1s,
        "accs": accs,
        "losses": losses,
        "mean_f1": float(f1s_arr.mean()),
        "median_f1": float(np.median(f1s_arr)),
        "min_f1": float(f1s_arr.min()),
        "max_f1": float(f1s_arr.max()),
        "std_f1": float(f1s_arr.std()),
        "mean_acc": float(accs_arr.mean()),
        "min_acc": float(accs_arr.min()),
        "std_acc": float(accs_arr.std()),
    }


def exhaustive_evaluate_pareto(genomes, clients, input_dim, n_classes, train_cfg, device="cpu", strategy="unknown", generation=0, split: str = "test"):
    """Evaluate list of genomes exhaustively. Returns DataFrame rows."""
    rows = []
    for idx, genome in enumerate(genomes):
        ev = exhaustive_evaluate_genome(genome, clients, input_dim, n_classes, train_cfg, device, split=split)
        row = {
            "strategy": strategy,
            "generation": int(generation),
            "pareto_idx": int(idx),
            "genome": str(genome.to_dict()),
            "genome_L": genome.L,
            "genome_hidden": str(genome.hidden_sizes),
            "genome_act": genome.activation,
            "genome_dropout": genome.dropout,
            "genome_bn": genome.use_bn,
            "genome_hash": genome_hash(genome),
            "mean_f1": ev["mean_f1"],
            "median_f1": ev["median_f1"],
            "min_f1": ev["min_f1"],
            "max_f1": ev["max_f1"],
            "std_f1": ev["std_f1"],
            "mean_acc": ev["mean_acc"],
            "min_acc": ev["min_acc"],
            "std_acc": ev["std_acc"],
        }
        for cid_idx, cid in enumerate(sorted(clients.keys())):
            row[f"f1_client_{cid}"] = ev["f1s"][cid_idx]
            row[f"acc_client_{cid}"] = ev["accs"][cid_idx]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_global_pareto(df, ref_point=np.array([0.1, 0.1])):
    """Compute globally validated Pareto front over (mean_f1, min_f1) -> maximize both => minimize -mean,-min."""
    if df is None or df.empty:
        return df, 0.0, np.array([])
    F = np.array([[-r["mean_f1"], -r["min_f1"]] for _, r in df.iterrows()])
    nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
    is_pareto = np.zeros(len(df), dtype=bool)
    is_pareto[nds] = True
    # HV
    try:
        hv = HV(ref_point=np.array(ref_point, dtype=float))
        # only pareto front
        hv_val = hv.do(F[nds]) if len(nds) else 0.0
    except Exception:
        hv_val = 0.0
    return is_pareto, float(hv_val), F


def compute_hv_common(df, ref_point=np.array([0.1, 0.1])):
    if df is None or df.empty:
        return 0.0
    F = np.array([[-r["mean_f1"], -r["min_f1"]] for _, r in df.iterrows()])
    try:
        hv = HV(ref_point=np.array(ref_point, dtype=float))
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        pf = F[nds] if len(nds) else F
        return float(hv.do(pf))
    except Exception:
        return 0.0


class HallOfFame:
    """Simple Hall of Fame preserving best architectures without duplicates."""
    def __init__(self):
        self.entries = {}  # hash -> dict

    def add(self, genome: Genome, generation: int, strategy: str, seed: int,
            fitness: np.ndarray, exhaustive_metrics: dict, per_client_f1: dict):
        h = genome_hash(genome)
        # keep first occurrence (earliest generation) or best mean? ponytail: earliest, skip complex
        if h in self.entries:
            return False
        rec = {
            "genome_hash": h,
            "genome": str(genome.to_dict()),
            "genome_L": genome.L,
            "genome_hidden": str(genome.hidden_sizes),
            "genome_act": genome.activation,
            "genome_dropout": genome.dropout,
            "genome_bn": genome.use_bn,
            "generation": int(generation),
            "strategy": strategy,
            "seed": int(seed),
            "fitness_F0": float(fitness[0]) if fitness is not None else None,
            "fitness_F1": float(fitness[1]) if fitness is not None else None,
            "f1_i_est": -float(fitness[0]) if fitness is not None else None,
            "f1_j_est": -float(fitness[1]) if fitness is not None else None,
            "mean_f1": exhaustive_metrics.get("mean_f1"),
            "median_f1": exhaustive_metrics.get("median_f1"),
            "min_f1": exhaustive_metrics.get("min_f1"),
            "max_f1": exhaustive_metrics.get("max_f1"),
            "std_f1": exhaustive_metrics.get("std_f1"),
            "mean_acc": exhaustive_metrics.get("mean_acc"),
        }
        for cid, f1 in per_client_f1.items():
            rec[f"f1_client_{cid}"] = float(f1)
        self.entries[h] = rec
        return True

    def to_dataframe(self):
        if not self.entries:
            return pd.DataFrame()
        df = pd.DataFrame(list(self.entries.values()))
        # sort by mean_f1 desc, then min_f1 desc
        if "mean_f1" in df.columns:
            df = df.sort_values(["mean_f1", "min_f1"], ascending=False)
        return df

    def size(self):
        return len(self.entries)

    def best_mean(self):
        if not self.entries:
            return None
        return max(self.entries.values(), key=lambda x: x["mean_f1"])

    def best_worst(self):
        if not self.entries:
            return None
        return max(self.entries.values(), key=lambda x: x["min_f1"])
