"""FedAvg compartido para NAS: un modelo global por arquitectura candidata.

Solo viajan params/métricas; datos crudos quedan en cada cliente.

Los buffers flotantes de BatchNorm (`running_mean` y `running_var`) se
agregan con la misma ponderación por `n_train` que los parámetros. El contador
`num_batches_tracked` se agrega con el máximo, porque es un contador local y
no una estadística que deba promediarse.
"""
from typing import Dict, List, Any
import copy
import numpy as np
import torch

from ..federated.characterization import (
    flatten_params,
    get_model_buffers,
    set_model_buffers,
    set_model_params,
)


def model_state_nbytes(model) -> int:
    """Approximate serialized parameter and buffer state size in bytes."""
    tensors = list(model.parameters()) + list(model.buffers())
    return int(sum(t.numel() * t.element_size() for t in tensors))


def fedavg_mean(vectors: Dict[int, np.ndarray], weights: Dict[int, float]) -> np.ndarray:
    cids = sorted(vectors.keys())
    total = float(sum(weights[c] for c in cids))
    if total <= 0:
        raise ValueError("FedAvg needs positive total weight")
    agg = None
    for cid in cids:
        w = float(weights[cid]) / total
        agg = vectors[cid] * w if agg is None else agg + vectors[cid] * w
    return agg


def fedavg_buffers(
    buffers: Dict[int, Dict[str, torch.Tensor]], weights: Dict[int, float]
) -> Dict[str, torch.Tensor]:
    """Aggregate non-trainable state, preserving integer counters safely."""
    cids = sorted(buffers.keys())
    total = float(sum(weights[c] for c in cids))
    if total <= 0:
        raise ValueError("FedAvg needs positive total weight")
    names = set(buffers[cids[0]])
    if any(set(buffers[cid]) != names for cid in cids[1:]):
        raise ValueError("All clients must return the same model buffers")

    aggregated = {}
    for name in sorted(names):
        values = [buffers[cid][name] for cid in cids]
        if values[0].is_floating_point():
            value = torch.zeros_like(values[0])
            for cid, local in zip(cids, values):
                value.add_(local, alpha=float(weights[cid]) / total)
        else:
            value = values[0].clone()
            for local in values[1:]:
                value = torch.maximum(value, local)
        aggregated[name] = value
    return aggregated


def run_fedavg(
    genome,
    genome_to_model,
    clients: Dict[int, Any],
    input_dim: int,
    n_classes: int,
    flat_init: np.ndarray,
    participants_per_round: List[List[int]],
    local_epochs: int = 1,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Ejecuta rondas FedAvg para una arquitectura. Retorna params globales y coste."""
    model = genome_to_model(genome, input_dim, n_classes)
    set_model_params(model, flat_init)
    global_params = flat_init.copy()
    global_buffers = get_model_buffers(model)
    n_local_trainings = 0
    total_steps = 0
    total_time = 0.0
    communication_bytes = 0
    communication_messages = 0
    state_bytes = model_state_nbytes(model)

    for participants in participants_per_round:
        updates = {}
        buffer_updates = {}
        weights = {}
        for cid in participants:
            client = clients[cid]
            m = genome_to_model(genome, input_dim, n_classes)
            res = client.train_from_params(
                m,
                global_params,
                buffers=global_buffers,
                epochs=local_epochs,
                lr=lr,
                optimizer_name=optimizer_name,
                device=device,
            )
            updates[int(cid)] = res["params"]
            weights[int(cid)] = float(res["n_train"])
            buffer_updates[int(cid)] = res["buffers"]
            n_local_trainings += 1
            total_steps += int(res["train_steps"])
            total_time += float(res["train_time"])
            communication_bytes += 2 * state_bytes
            communication_messages += 2
        global_params = fedavg_mean(updates, weights)
        global_buffers = fedavg_buffers(buffer_updates, weights)
        set_model_params(model, global_params)
        set_model_buffers(model, global_buffers)

    return {
        "global_params": global_params,
        "model": model,
        "n_local_trainings": int(n_local_trainings),
        "total_steps": int(total_steps),
        "train_time": float(total_time),
        "n_rounds": int(len(participants_per_round)),
        "state_bytes": int(state_bytes),
        "communication_bytes": int(communication_bytes),
        "communication_messages": int(communication_messages),
    }


def evaluate_global_on_split(
    model, clients: Dict[int, Any], split: str = "val", device: str = "cpu"
) -> Dict[int, Dict[str, float]]:
    """Evalúa un modelo global ya entrenado en cada cliente. No entrena."""
    model.eval()
    out = {}
    for cid in sorted(clients.keys()):
        res = clients[cid].evaluate(copy.deepcopy(model), device=device, split=split)
        out[int(cid)] = {
            "accuracy": float(res["accuracy"]),
            "macro_f1": float(res["macro_f1"]),
            "loss": float(res["loss"]),
            "n": int(res["n"]),
        }
    return out
