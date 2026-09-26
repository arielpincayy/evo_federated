"""Caracterización federada: modelo sonda + delta vectors."""
from typing import Dict, Any, List
import copy
import torch
import torch.nn as nn
import numpy as np

from ..models.genome import Genome, genome_to_model, model_num_parameters


def flatten_params(model: nn.Module) -> np.ndarray:
    """Flatten all parameters in consistent order."""
    vecs = []
    for p in model.parameters():
        vecs.append(p.detach().cpu().numpy().ravel())
    if not vecs:
        return np.array([], dtype=np.float32)
    return np.concatenate(vecs)


def set_model_params(model: nn.Module, flat: np.ndarray):
    """Set model params from flat vector (order must match)."""
    pointer = 0
    for p in model.parameters():
        num = p.numel()
        arr = flat[pointer:pointer+num]
        pointer += num
        tensor = torch.from_numpy(arr.reshape(p.shape)).to(p.device).type(p.dtype)
        p.data.copy_(tensor)


def get_model_buffers(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a CPU copy of non-trainable model state, including BatchNorm."""
    return {
        name: buffer.detach().cpu().clone()
        for name, buffer in model.named_buffers()
    }


def set_model_buffers(model: nn.Module, buffers: Dict[str, torch.Tensor]):
    """Restore non-trainable model state without replacing module buffers."""
    model_buffers = dict(model.named_buffers())
    with torch.no_grad():
        for name, value in buffers.items():
            if name not in model_buffers:
                raise KeyError(f"Unknown model buffer: {name}")
            model_buffers[name].copy_(value.to(model_buffers[name].device))


def characterize_clients(
    clients: Dict[int, Any],
    decoder_input_dim: int,
    decoder_n_classes: int,
    genome_config: Dict[str, Any],
    train_config: Dict[str, Any],
    k_epochs: int = 5,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Fase de caracterización: entrenar mismo modelo sonda en cada cliente.
    Retorna dict con deltas, initial flat, etc.
    """
    rng = np.random.RandomState(seed)
    # generate probe genome deterministically
    probe_genome = Genome.random(genome_config, rng)

    # Create probe model
    probe_model = genome_to_model(probe_genome, decoder_input_dim, decoder_n_classes)
    # Ensure identical initialization: use seed for torch
    torch.manual_seed(seed)
    # Re-init probe model with same seed? Already after creation, need to re-create with seed control
    # We'll create via same rng path and set seed before model instantiation
    # For reproducibility, set seed then recreate
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    probe_model = genome_to_model(probe_genome, decoder_input_dim, decoder_n_classes)

    # Flatten initial
    theta0 = flatten_params(probe_model)

    deltas: Dict[int, np.ndarray] = {}
    metrics: Dict[int, Dict[str, Any]] = {}

    for cid, client in clients.items():
        # Fresh copy for each client starting from same theta0
        model_copy = copy.deepcopy(probe_model)
        # Ensure copy has theta0 exactly
        set_model_params(model_copy, theta0)

        # Train k epochs locally
        train_res = client.train(
            model_copy,
            epochs=k_epochs,
            lr=train_config.get("lr", 1e-3),
            optimizer_name=train_config.get("optimizer", "adam"),
            device=device,
        )
        theta_k = flatten_params(model_copy)
        delta = theta_k - theta0
        deltas[cid] = delta
        metrics[cid] = train_res

    return {
        "probe_genome": probe_genome,
        "theta0": theta0,
        "deltas": deltas,
        "metrics": metrics,
        "n_params": len(theta0),
    }


def compute_delta_vectors(deltas: Dict[int, np.ndarray]) -> np.ndarray:
    """Stack deltas into matrix N x P ordered by sorted client ids."""
    cids = sorted(deltas.keys())
    mat = np.stack([deltas[cid] for cid in cids], axis=0)
    return mat
