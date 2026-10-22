"""Configuración: dataclasses anidadas + carga/guardado yaml/json."""
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


@dataclasses.dataclass
class DatasetConfig:
    name: str = "cifar10"  # cifar10 | fashion-mnist | synthetic
    data_root: str = "./data"
    n_clients: int = 100
    alpha: float = 0.5
    min_samples_per_client: int = 20
    val_ratio: float = 0.3  # val por nodo; test es el test oficial particionado aparte
    seed: int = 42
    synthetic_n_samples: int = 5000
    synthetic_n_features: int = 64
    synthetic_n_classes: int = 10
    synthetic_n_informative: int = 30


@dataclasses.dataclass
class TrainConfig:
    optimizer: str = "adam"
    lr: float = 1e-3
    batch_size: int = 32
    device: str = "cpu"
    num_workers: int = 0
    num_threads: int = 4  # hilos de torch por proceso (modelos chicos: más hilos no ayudan)


@dataclasses.dataclass
class SpaceConfig:
    max_blocks: int = 3
    channels: tuple = (8, 16, 32, 64)
    kernels: tuple = (3, 5)
    activations: tuple = ("relu", "elu")
    max_dropout: float = 0.5
    use_batchnorm: bool = True


@dataclasses.dataclass
class ProbeConfig:
    k_epochs: int = 1
    seed: int = 42


@dataclasses.dataclass
class SamplingConfig:
    sampler: str = "cluster"  # cluster | label_cluster | random_cluster | random | full
    n_clusters: int = 10  # C = nº de representantes por evaluación
    method: str = "kmeans"  # kmeans | average
    weight_by_cluster: bool = True  # FedAvg ponderado por tamaño de cluster
    eval_holdout: bool = True  # evaluar en otro nodo del cluster que no entrenó (inferencia)


@dataclasses.dataclass
class FederatedConfig:
    rounds: int = 1
    local_epochs: int = 1


@dataclasses.dataclass
class EvolutionConfig:
    search: str = "nsga2"  # nsga2 | random
    pop_size: int = 12
    n_generations: int = 10
    crossover_prob: float = 0.9
    crossover_eta: float = 15
    mutation_prob: float = 0.2
    mutation_eta: float = 20
    seed: int = 42


@dataclasses.dataclass
class FinalConfig:
    enabled: bool = True
    n_seeds: int = 3  # reentrenamientos independientes del frente final
    participation: str = "full"  # full | reps
    max_front: int = 10


@dataclasses.dataclass
class ExperimentConfig:
    name: str = "cs_nas"
    output_dir: str = "runs/cs_nas"
    seed: int = 42
    monitor: bool = True  # evaluar todos los nodos solo para reporte
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    space: SpaceConfig = dataclasses.field(default_factory=SpaceConfig)
    probe: ProbeConfig = dataclasses.field(default_factory=ProbeConfig)
    sampling: SamplingConfig = dataclasses.field(default_factory=SamplingConfig)
    federated: FederatedConfig = dataclasses.field(default_factory=FederatedConfig)
    evolution: EvolutionConfig = dataclasses.field(default_factory=EvolutionConfig)
    final: FinalConfig = dataclasses.field(default_factory=FinalConfig)


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _build(cls, data: Dict[str, Any]):
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        if isinstance(v, dict) and f.default_factory is not dataclasses.MISSING:
            v = _build(type(f.default_factory()), v)
        elif isinstance(v, list):
            v = tuple(v)
        kwargs[f.name] = v
    return cls(**kwargs)


def from_dict(data: Dict[str, Any]) -> ExperimentConfig:
    return _build(ExperimentConfig, _merge(dataclasses.asdict(ExperimentConfig()), data))


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("PyYAML no instalado")
            data = yaml.safe_load(f) or {}
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Formato no soportado: {path.suffix}")
    return from_dict(data)


def save_config(cfg: ExperimentConfig, path: str | Path):
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, ensure_ascii=False)
