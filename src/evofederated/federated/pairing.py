"""Estrategias de pareamiento de clientes."""
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np


class PairingStrategy(ABC):
    @abstractmethod
    def select_counterpart(self, ref_client_idx: int, D: np.ndarray, rng: np.random.RandomState) -> int:
        pass


class RandomPairing(PairingStrategy):
    def select_counterpart(self, ref_client_idx: int, D: np.ndarray, rng: np.random.RandomState) -> int:
        n = D.shape[0]
        choices = [i for i in range(n) if i != ref_client_idx]
        return int(rng.choice(choices))


class FixedDivergentPairing(PairingStrategy):
    """Deterministic most divergent fixed."""
    def select_counterpart(self, ref_client_idx: int, D: np.ndarray, rng: np.random.RandomState) -> int:
        # argmax excluding self
        row = D[ref_client_idx].copy()
        row[ref_client_idx] = -1  # invalid
        return int(np.argmax(row))


class DynamicDivergentPairing(PairingStrategy):
    def __init__(self, tau: float = 5.0, eps: float = 1e-9):
        self.tau = tau
        self.eps = eps

    def select_counterpart(self, ref_client_idx: int, D: np.ndarray, rng: np.random.RandomState) -> int:
        row = D[ref_client_idx].copy()
        # exclude self by zero prob
        # compute probabilities for j != i
        n = D.shape[0]
        probs = np.zeros(n, dtype=float)
        for j in range(n):
            if j == ref_client_idx:
                probs[j] = 0.0
            else:
                probs[j] = (row[j] + self.eps) ** self.tau
        s = probs.sum()
        if s < 1e-12:
            # fallback uniform
            choices = [i for i in range(n) if i != ref_client_idx]
            return int(rng.choice(choices))
        probs /= s
        # categorical sampling
        return int(rng.choice(n, p=probs))


def get_pairing(name: str, tau: float = 5.0) -> PairingStrategy:
    name = name.lower()
    if name in ("random", "random-2", "random2"):
        return RandomPairing()
    elif name in ("fixed", "fixed-divergent", "fixed-divergent-2", "fixe"):
        return FixedDivergentPairing()
    elif name in ("dynamic", "dynamic-divergent", "dynamic-divergent-2", "proposed"):
        return DynamicDivergentPairing(tau=tau)
    elif name in ("full", "exhaustive"):
        # For FULL, pairing is not used but return a dummy to satisfy interface
        return RandomPairing()
    else:
        raise ValueError(f"Unknown pairing {name}")
