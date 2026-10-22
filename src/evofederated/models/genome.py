"""Genoma -> Decoder -> PyTorch Modelo.
Espacio MLP con longitud fija, L primeras capas activas.

Genoma vector (float/int) para pymoo:
  [ L_norm, n1_norm, n2_norm, ..., n_{Lmax}_norm, a_idx, dropout_norm, bn_flag ]
Simplificamos: usaremos representación entera/discreta internamente y codificación
real en [0,1] para pymoo continuous + repair. Pero también soportamos muestreo directo.

Para pymoo MixedInteger: usaremos variables:
  - L : int 1..Lmax
  - n_i : int index into choices (0..len(choices)-1)
  - a_idx : int index into activations
  - dropout : float 0..0.5
  - use_bn : int 0/1 (si genome_config.use_batchnorm True, else fijo 0)

Decoder garantiza arquitectura válida.
"""
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
import torch
import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
}

DEFAULT_CHOICES = (16, 32, 64, 128, 256)
DEFAULT_ACTS = ("relu", "tanh", "elu")


class Genome:
    """Contenedor de genoma decodificado."""

    def __init__(
        self,
        L: int,
        hidden_sizes: List[int],
        activation: str,
        dropout: float,
        use_bn: bool,
        l_max: int = 4,
    ):
        self.L = int(L)
        self.hidden_sizes = [int(x) for x in hidden_sizes[: self.L]]
        # pad hidden_sizes to l_max for vector consistency
        self.l_max = l_max
        self.activation = activation
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)

    def to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        """Codifica a vector float para pymoo (continuous 0-1 + categorical)."""
        choices = config.get("n_neurons_choices", DEFAULT_CHOICES)
        acts = config.get("activations", DEFAULT_ACTS)
        # L: 1..l_max -> norm 0..1
        l_max = config.get("l_max", self.l_max)
        vec = []
        vec.append((self.L - 1) / max(1, l_max - 1))
        for i in range(l_max):
            if i < self.L:
                idx = choices.index(self.hidden_sizes[i]) if self.hidden_sizes[i] in choices else 0
            else:
                idx = 0
            vec.append(idx / max(1, len(choices) - 1))
        # activation
        a_idx = acts.index(self.activation) if self.activation in acts else 0
        vec.append(a_idx / max(1, len(acts) - 1))
        # dropout 0..0.5 -> norm
        vec.append(self.dropout / 0.5)
        # bn
        vec.append(1.0 if self.use_bn else 0.0)
        return np.array(vec, dtype=float)

    @staticmethod
    def from_vector(vec: np.ndarray, config: Dict[str, Any]) -> "Genome":
        """Decodifica vector 0..1 a Genome."""
        choices = tuple(config.get("n_neurons_choices", DEFAULT_CHOICES))
        acts = tuple(config.get("activations", DEFAULT_ACTS))
        l_max = int(config.get("l_max", 4))
        # clip
        vec = np.clip(vec, 0, 1)
        # L
        L = int(round(vec[0] * (l_max - 1))) + 1
        L = max(1, min(L, l_max))
        hidden = []
        for i in range(l_max):
            idx = int(round(vec[1 + i] * (len(choices) - 1)))
            idx = max(0, min(idx, len(choices) - 1))
            hidden.append(int(choices[idx]))
        a_idx = int(round(vec[1 + l_max] * (len(acts) - 1)))
        a_idx = max(0, min(a_idx, len(acts) - 1))
        activation = acts[a_idx]
        dropout = float(vec[1 + l_max + 1] * 0.5)
        # clamp dropout
        dropout = max(0.0, min(dropout, 0.5))
        use_bn = bool(vec[1 + l_max + 2] > 0.5) if config.get("use_batchnorm", True) else False
        # if use_dropout False, force dropout 0
        if not config.get("use_dropout", True):
            dropout = 0.0
        return Genome(L=L, hidden_sizes=hidden, activation=activation, dropout=dropout, use_bn=use_bn, l_max=l_max)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "L": self.L,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "dropout": self.dropout,
            "use_bn": self.use_bn,
            "l_max": self.l_max,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Genome":
        return Genome(
            L=d["L"],
            hidden_sizes=d["hidden_sizes"],
            activation=d["activation"],
            dropout=d["dropout"],
            use_bn=d["use_bn"],
            l_max=d.get("l_max", 4),
        )

    @staticmethod
    def random(config: Dict[str, Any], rng: np.random.RandomState) -> "Genome":
        l_max = int(config.get("l_max", 4))
        choices = tuple(config.get("n_neurons_choices", DEFAULT_CHOICES))
        acts = tuple(config.get("activations", DEFAULT_ACTS))
        L = rng.randint(1, l_max + 1)
        hidden = [int(rng.choice(choices)) for _ in range(l_max)]
        # keep only first L active but generate all
        activation = str(rng.choice(acts))
        dropout = float(rng.uniform(0, 0.5)) if config.get("use_dropout", True) else 0.0
        use_bn = bool(rng.rand() > 0.5) if config.get("use_batchnorm", True) else False
        return Genome(L=L, hidden_sizes=hidden, activation=activation, dropout=dropout, use_bn=use_bn, l_max=l_max)

    def __repr__(self):
        return f"Genome(L={self.L}, hidden={self.hidden_sizes}, act={self.activation}, dropout={self.dropout:.2f}, bn={self.use_bn})"


class MLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, genome: Genome):
        super().__init__()
        self.genome = genome
        self.input_dim = input_dim
        self.n_classes = n_classes
        layers = []
        prev = input_dim
        act_cls = ACTIVATIONS.get(genome.activation, nn.ReLU)
        for i in range(genome.L):
            h = genome.hidden_sizes[i]
            layers.append(nn.Linear(prev, h))
            if genome.use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls())
            if genome.dropout > 1e-6:
                layers.append(nn.Dropout(genome.dropout))
            prev = h
        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, n_classes)

    def forward(self, x):
        # flatten if image
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.feature(x)
        return self.classifier(x)


class GenomeDecoder:
    def __init__(self, input_dim: int, n_classes: int, genome_config: Dict[str, Any]):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.config = genome_config

    def decode(self, genome: Genome) -> nn.Module:
        return MLP(self.input_dim, self.n_classes, genome)

    def decode_vector(self, vec: np.ndarray) -> nn.Module:
        g = Genome.from_vector(vec, self.config)
        return self.decode(g)


def genome_to_model(genome: Genome, input_dim: int, n_classes: int) -> nn.Module:
    return MLP(input_dim, n_classes, genome)


def model_num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_stats(model: nn.Module) -> Dict[str, Any]:
    n_params = model_num_parameters(model)
    # rough model size MB (float32)
    size_mb = n_params * 4 / (1024 * 1024)
    # depth: count Linear layers
    depth = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    return {"n_params": n_params, "size_mb": size_mb, "depth": depth}


MODEL_REGISTRY = {"mlp": MLP}
