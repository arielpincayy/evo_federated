"""Config handling: dataclass-based + yaml/json load."""
import json
import dataclasses
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


@dataclasses.dataclass
class DatasetConfig:
    name: str = "fashion-mnist"  # fashion-mnist | cifar10 | synthetic
    data_root: str = "./data"
    n_clients: int = 10
    alpha: float = 0.5
    min_samples_per_client: int = 10
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    seed: int = 42
    # synthetic only
    synthetic_n_samples: int = 5000
    synthetic_n_features: int = 64
    synthetic_n_classes: int = 10
    synthetic_n_informative: int = 30


@dataclasses.dataclass
class TrainConfig:
    optimizer: str = "adam"
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 5
    loss: str = "cross_entropy"
    device: str = "auto"  # auto | cpu | cuda
    num_workers: int = 0


@dataclasses.dataclass
class GenomeConfig:
    l_max: int = 4
    n_neurons_choices: tuple = (16, 32, 64, 128, 256)
    activations: tuple = ("relu", "tanh", "elu")
    dropout_range: tuple = (0.0, 0.5)
    use_batchnorm: bool = True
    use_dropout: bool = True


@dataclasses.dataclass
class CharacterizationConfig:
    k_epochs: int = 5
    seed: int = 42
    distance_metric: str = "cosine"  # cosine | l2_norm
    tau: float = 5.0


@dataclasses.dataclass
class FederatedConfig:
    rounds: int = 2
    clients_per_round: int = 2
    local_epochs: int = 1
    aggregation: str = "fedavg"  # solo fedavg ponderado por n_train
    search_split: str = "val"  # búsqueda en validation; test reservado al final
    final_split: str = "test"
    final_participation: str = "full"  # full | same: régimen usado en evaluación final


@dataclasses.dataclass
class EvolutionConfig:
    pop_size: int = 20
    n_generations: int = 10
    crossover_prob: float = 0.9
    crossover_eta: float = 15
    mutation_prob: float = 0.2
    mutation_eta: float = 20
    seed: int = 42
    # hypervolume
    hv_ref_point: tuple = (0.0, 0.0)  # in F1 space, will be transformed to minimization (1,1)? See docs
    # For minimization we use (-F1), so ref worse = (0,0) in max space => (0,0) transformed? Actually -F1 in [ -1,0], worse = 0
    # We'll store ref in minimization space explicitly
    hv_ref_min: tuple = (0.0, 0.0)  # reference for minimization: (0,0) worse than any -F1 (which are <=0)
    strategy: str = "dynamic"  # full | random | fixed | dynamic
    tau: float = 5.0
    control_every: int = 0  # 0 = only at end, else every T generations
    control_top_k: int = 5


@dataclasses.dataclass
class ExperimentConfig:
    name: str = "evo_federated"
    output_dir: str = "results"
    seed: int = 42
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    genome: GenomeConfig = dataclasses.field(default_factory=GenomeConfig)
    characterization: CharacterizationConfig = dataclasses.field(default_factory=CharacterizationConfig)
    evolution: EvolutionConfig = dataclasses.field(default_factory=EvolutionConfig)
    federated: FederatedConfig = dataclasses.field(default_factory=FederatedConfig)
    baselines: tuple = ("full", "random", "fixed", "dynamic")


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _merge_dict(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML not installed. pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    # Build nested configs
    base = dataclasses.asdict(ExperimentConfig())
    merged = _merge_dict(base, data)

    # reconstruct dataclasses
    cfg = ExperimentConfig(
        name=merged["name"],
        output_dir=merged["output_dir"],
        seed=merged["seed"],
        dataset=DatasetConfig(**merged["dataset"]),
        train=TrainConfig(**merged["train"]),
        genome=GenomeConfig(**merged["genome"]),
        characterization=CharacterizationConfig(**merged["characterization"]),
        evolution=EvolutionConfig(**merged["evolution"]),
        federated=FederatedConfig(**merged.get("federated", {})),
        baselines=tuple(merged.get("baselines", ("full", "random", "fixed", "dynamic"))),
    )
    return cfg


def save_config(cfg: ExperimentConfig, path: str | Path):
    path = Path(path)
    data = dataclasses.asdict(cfg)
    if path.suffix in (".yaml", ".yml"):
        if yaml is None:
            raise ImportError("PyYAML not installed")
        with open(path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
