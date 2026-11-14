"""Genoma CNN pequeño para NAS: bloques conv (+BN) + activación + maxpool, pooling, capas FC opcionales y linear de salida.

Vector real en [0,1] (compatible con SBX/PM de pymoo):
  [ conv_1..conv_B, fc_1..fc_F, kernel, activation, dropout, use_bn ]
conv_1 siempre activo; el resto de slots (conv_i, fc_j) se apagan si v < off_threshold y se saltan al decodificar.
`CNNGenome.from_vector` siempre decodifica una arquitectura válida. Ver docs/PROTOCOL.md (espacio v2).
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
    max_fc: int = 0  # slots de capas FC ocultas (0 = solo la Linear de salida)
    fc_widths: Tuple[int, ...] = (32, 64, 128, 256)
    pool: int = 1  # AdaptiveAvgPool2d(pool) antes de las FC: 1 = pooling global, 2 = entrada FC de 4*ch
    off_threshold: float = 0.25  # v < off_threshold apaga un slot opcional

    @property
    def n_var(self) -> int:
        return self.max_blocks + self.max_fc + 4


@dataclasses.dataclass
class CNNGenome:
    n_blocks: int
    channels: List[int]  # longitud n_blocks
    kernel: int
    activation: str
    dropout: float
    use_bn: bool
    fc: List[int] = dataclasses.field(default_factory=list)  # anchos de las FC ocultas
    pool: int = 1

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

        def slot(x, opts):  # None = slot apagado
            t = space.off_threshold
            return None if x < t else int(opts[pick((x - t) / (1 - t), len(opts))])

        convs = [int(space.channels[pick(v[0], len(space.channels))])]
        convs += [c for c in (slot(v[i], space.channels) for i in range(1, space.max_blocks)) if c]
        j = space.max_blocks
        fcs = [w for w in (slot(v[j + i], space.fc_widths) for i in range(space.max_fc)) if w]
        j += space.max_fc
        kernel = int(space.kernels[pick(v[j], len(space.kernels))])
        act = space.activations[pick(v[j + 1], len(space.activations))]
        dropout = float(v[j + 2]) * space.max_dropout
        use_bn = bool(v[j + 3] > 0.5) if space.use_batchnorm else False
        return CNNGenome(len(convs), convs, kernel, act, dropout, use_bn, fcs, space.pool)


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
        head: List[nn.Module] = [nn.AdaptiveAvgPool2d(g.pool), nn.Flatten()]
        d = c * g.pool * g.pool
        for w in g.fc:
            head += [nn.Dropout(g.dropout), nn.Linear(d, w), ACTIVATIONS[g.activation]()]
            d = w
        head += [nn.Dropout(g.dropout), nn.Linear(d, n_classes)]  # con fc=[] y pool=1 es la cabeza original
        self.head = nn.Sequential(*head)

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
                  space.activations[0], 0.0, space.use_batchnorm, [max(space.fc_widths)] * space.max_fc, space.pool),
        input_shape, n_classes))
