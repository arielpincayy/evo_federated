"""Servidor federado: solo recibe derivadas, no raw data."""
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
import numpy as np
import copy


class FederatedServer:
    """
    Mantiene model global, coordina caracterización, pareamiento, evolución.
    Nunca accede a datos crudos.
    """

    def __init__(self, clients: Dict[int, Any]):
        self.clients = clients
        self.n_clients = len(clients)
        # Allowed derived info storage
        self.client_info: Dict[int, Dict[str, Any]] = {}
        self.divergence_matrix: Optional[np.ndarray] = None

    def collect_client_info(self):
        for cid, client in self.clients.items():
            self.client_info[cid] = client.get_dataset_info()

    def aggregate_placeholder(self):
        """Placeholder for FedAvg etc., no usada directamente en NAS pero disponible."""
        pass

    def fedavg_aggregate(self, updates: Dict[int, np.ndarray], weights: Dict[int, float]) -> np.ndarray:
        """Weighted FedAvg over client updates.

        updates: client_id -> flat param vector (post local training).
        weights: client_id -> n_train samples. Only derived sizes, no raw data.
        Returns aggregated flat vector.
        """
        cids = sorted(updates.keys())
        total = float(sum(weights[c] for c in cids))
        if total <= 0:
            raise ValueError("FedAvg needs positive total weight")
        agg = None
        for cid in cids:
            w = float(weights[cid]) / total
            vec = updates[cid]
            agg = vec * w if agg is None else agg + vec * w
        return agg

    def set_divergence_matrix(self, D: np.ndarray):
        self.divergence_matrix = D

    def get_divergence_matrix(self) -> Optional[np.ndarray]:
        return self.divergence_matrix
