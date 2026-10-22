"""Genoma CNN pequeño para NAS: bloques conv (+BN) + activación + maxpool, GAP y linear.

Vector real en [0,1] (compatible con SBX/PM de pymoo):
  [ n_blocks, ch_1..ch_Bmax, kernel, activation, dropout, use_bn ]
`Genome.from_vector` siempre decodifica una arquitectura válida.
"""
import dataclasses
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "elu": nn.ELU, "gelu": nn.GELU, "tanh": nn.Tanh}


@dataclasses.dataclass
class CNNSpace:
    """Define el espacio de búsqueda (viene de la configuración)."""
    max_blocks: int = 3
    channels: Tuple[int, ...] = (8, 16, 32, 64)
    kernels: Tuple[int, ...] = (3, 5)
    activations: Tuple[str, ...] = ("relu", "elu")
    max_dropout: float = 0.5
    use_batchnorm: bool = True

    @property
    def n_var(self) -> int:
        return self.max_blocks + 5


@dataclasses.dataclass
class CNNGenome:
    n_blocks: int
    channels: List[int]  # longitud n_blocks
    kernel: int
    activation: str
    dropout: float
    use_bn: bool

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def random_vector(space: CNNSpace, rng: np.random.RandomState) -> np.ndarray:
        return rng.rand(space.n_var)

    @staticmethod
    def from_vector(vec: np.ndarray, space: CNNSpace) -> "CNNGenome":
        v = np.clip(np.asarray(vec, dtype=float), 0.0, 1.0)

        def pick(x, n):
            return int(min(n - 1, max(0, round(x * (n - 1)))))

        n_blocks = 1 + pick(v[0], space.max_blocks)
        chans = [int(space.channels[pick(v[1 + i], len(space.channels))]) for i in range(n_blocks)]
        j = 1 + space.max_blocks
        kernel = int(space.kernels[pick(v[j], len(space.kernels))])
        act = space.activations[pick(v[j + 1], len(space.activations))]
        dropout = float(v[j + 2]) * space.max_dropout
        use_bn = bool(v[j + 3] > 0.5) if space.use_batchnorm else False
        return CNNGenome(n_blocks, chans, kernel, act, dropout, use_bn)


class CNN(nn.Module):
    def __init__(self, g: CNNGenome, input_shape: Tuple[int, int, int], n_classes: int):
        super().__init__()
        c, h, w = input_shape
        layers: List[nn.Module] = []
        for ch in g.channels:
            layers.append(nn.Conv2d(c, ch, g.kernel, padding=g.kernel // 2))
            if g.use_bn:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(ACTIVATIONS[g.activation]())
            if h >= 2 and w >= 2:
                layers.append(nn.MaxPool2d(2))
                h, w = h // 2, w // 2
            c = ch
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(g.dropout), nn.Linear(c, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def cnn_to_model(g: CNNGenome, input_shape, n_classes: int) -> nn.Module:
    return CNN(g, tuple(input_shape), n_classes)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def max_n_params(space: CNNSpace, input_shape, n_classes: int) -> int:
    """Parámetros de la arquitectura más grande del espacio (normaliza el 2º objetivo a (0,1])."""
    return n_params(cnn_to_model(
        CNNGenome(space.max_blocks, [max(space.channels)] * space.max_blocks, max(space.kernels),
                  space.activations[0], 0.0, space.use_batchnorm), input_shape, n_classes))
