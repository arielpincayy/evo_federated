"""Cliente federado: posee dataset local, método train/eval. No expone raw data."""
from typing import Dict, Any, Callable, Optional
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def n_train(self):
        return len(self.train_dataset) if self.train_dataset is not None else 0

    @property
    def n_val(self):
        return len(self.val_dataset) if self.val_dataset is not None else 0

    @property
    def n_test(self):
        return len(self.test_dataset) if self.test_dataset is not None else 0

    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def get_val_loader(self) -> Optional[DataLoader]:
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_test_loader(self) -> Optional[DataLoader]:
        if self.test_dataset is None or len(self.test_dataset) == 0:
            return None
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def train(
        self,
        model: nn.Module,
        epochs: int,
        lr: float = 1e-3,
        optimizer_name: str = "adam",
        device: str = "cpu",
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        Entrena modelo localmente. Retorna métricas, NO datos.
        Model is trained in-place; caller should provide a fresh copy to avoid contamination.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        model = model.to(device)
        model.train()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer {optimizer_name}")

        loader = self.get_train_loader(shuffle=True)
        start = time.time()
        total_loss = 0.0
        steps = 0
        for _ in range(epochs):
            for xb, yb in loader:
                # ponytail: skip batch_size 1 to avoid BatchNorm failure (minimal fix, drop_last would also work)
                if xb.size(0) == 1:
                    continue
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

        elapsed = time.time() - start
        avg_loss = total_loss / max(1, steps)
        return {"loss": avg_loss, "time": elapsed, "steps": steps}

    def evaluate(
        self,
        model: nn.Module,
        device: str = "cpu",
        split: str = "test",
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """
        Evalúa modelo en split local. Retorna métricas.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if split == "test":
            loader = self.get_test_loader()
        elif split == "val":
            loader = self.get_val_loader()
        else:
            loader = self.get_train_loader(shuffle=False)

        if loader is None:
            return {"accuracy": 0.0, "loss": 0.0, "macro_f1": 0.0, "n": 0}

        model = model.to(device)
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                total_loss += loss.item() * yb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(yb.cpu().tolist())

        acc = correct / max(1, total)
        loss = total_loss / max(1, total)

        # macro F1
        try:
            from sklearn.metrics import f1_score

            macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        except Exception:
            macro_f1 = 0.0

        return {"accuracy": acc, "loss": loss, "macro_f1": macro_f1, "n": total}

    # Privacy enforcement helpers
    def get_model_update(self, delta: torch.Tensor):
        """Simulates sending delta (or model params) to server."""
        return delta

    def get_metrics(self, metrics: Dict[str, Any]):
        """Simulates sending metrics to server."""
        return metrics

    def get_dataset_info(self):
        """Only allowed info: size, class counts via metrics, not raw data."""
        return {"n_train": self.n_train, "n_val": self.n_val, "n_test": self.n_test, "client_id": self.client_id}
