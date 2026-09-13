"""Experimento piloto: verifica dificultad dataset y sensibilidad arquitectura.

Hace:
- genera dataset y partición
- crea 5-10 arquitecturas deliberadamente diferentes
- entrena todas en todos los clientes
- genera matriz arquitectura x cliente
- verifica variabilidad y cambios de ranking
"""
import time
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
import pandas as pd

from ..utils.seed import set_seed
from ..utils.config import ExperimentConfig
from ..data.datasets import get_dataset, make_synthetic_dataset, dirichlet_partition, get_input_dim_and_n_classes
from ..clients.client import FederatedClient
from ..models.genome import Genome, genome_to_model
from ..federated.characterization import flatten_params, set_model_params


def run_pilot(cfg: ExperimentConfig, output_dir: Path = None, n_archs: int = 8, verbose: bool = True):
    set_seed(cfg.seed)
    if output_dir is None:
        output_dir = Path(cfg.output_dir) / "pilot"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    device = cfg.train.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    if cfg.dataset.name == "synthetic":
        base_dataset = make_synthetic_dataset(
            n_samples=cfg.dataset.synthetic_n_samples,
            n_features=cfg.dataset.synthetic_n_features,
            n_informative=cfg.dataset.synthetic_n_informative,
            n_classes=cfg.dataset.synthetic_n_classes,
            seed=cfg.seed,
        )
        labels = base_dataset.tensors[1].numpy()
    else:
        train_set = get_dataset(cfg.dataset.name, root=cfg.dataset.data_root, train=True, download=True)
        if hasattr(train_set, "targets"):
            labels = np.array(train_set.targets) if not isinstance(train_set.targets, torch.Tensor) else train_set.targets.numpy()
        else:
            labels = np.array([train_set[i][1] for i in range(len(train_set))])
        base_dataset = train_set

    input_dim, n_classes = get_input_dim_and_n_classes(base_dataset)

    partition = dirichlet_partition(
        labels=labels,
        n_clients=cfg.dataset.n_clients,
        alpha=cfg.dataset.alpha,
        seed=cfg.dataset.seed,
        min_samples_per_client=cfg.dataset.min_samples_per_client,
        val_ratio=cfg.dataset.val_ratio,
        test_ratio=cfg.dataset.test_ratio,
    )

    # create clients
    clients = {}
    for cid in range(cfg.dataset.n_clients):
        from torch.utils.data import Subset
        train_idxs = partition.splits[cid]["train"]
        test_idxs = partition.splits[cid]["test"]
        train_ds = Subset(base_dataset, train_idxs)
        test_ds = Subset(base_dataset, test_idxs) if len(test_idxs)>0 else None
        clients[cid] = FederatedClient(cid, train_ds, None, test_ds, batch_size=cfg.train.batch_size)

    # Generate n_archs deliberately different
    rng = np.random.RandomState(cfg.seed)
    # Predefined diverse configs
    diverse_genomes = []
    # Create extremes
    # Small shallow, large deep, high dropout, no dropout, tanh vs relu, etc.
    presets = [
        {"L": 1, "hidden": [16], "act": "relu", "dropout": 0.0, "bn": False},
        {"L": 1, "hidden": [256], "act": "relu", "dropout": 0.0, "bn": False},
        {"L": 3, "hidden": [64, 32, 16], "act": "relu", "dropout": 0.0, "bn": False},
        {"L": 4, "hidden": [256, 128, 64, 32], "act": "tanh", "dropout": 0.5, "bn": True},
        {"L": 2, "hidden": [128, 64], "act": "elu", "dropout": 0.3, "bn": True},
        {"L": 4, "hidden": [64, 64, 64, 64], "act": "relu", "dropout": 0.2, "bn": True},
        {"L": 2, "hidden": [32, 32], "act": "tanh", "dropout": 0.0, "bn": False},
        {"L": 3, "hidden": [256, 256, 256], "act": "relu", "dropout": 0.1, "bn": True},
        {"L": 1, "hidden": [32], "act": "elu", "dropout": 0.5, "bn": True},
        {"L": 4, "hidden": [16, 32, 64, 128], "act": "tanh", "dropout": 0.0, "bn": False},
    ]
    for i in range(n_archs):
        if i < len(presets):
            p = presets[i]
            g = Genome(L=p["L"], hidden_sizes=p["hidden"] + [16]*(cfg.genome.l_max - len(p["hidden"])), activation=p["act"], dropout=p["dropout"], use_bn=p["bn"], l_max=cfg.genome.l_max)
            # Ensure hidden sizes length l_max
            if len(g.hidden_sizes) < cfg.genome.l_max:
                g.hidden_sizes += [16]*(cfg.genome.l_max - len(g.hidden_sizes))
            diverse_genomes.append(g)
        else:
            g = Genome.random(dict(L=cfg.genome.__dict__ if hasattr(cfg.genome, '__dict__') else {}), rng)
            # Use proper dict
            import dataclasses
            g = Genome.random(dataclasses.asdict(cfg.genome), rng)
            diverse_genomes.append(g)
    # If presets used, but random fallback above would have overwritten? Corrected: only else

    # Actually rebuild to ensure correct count
    if len(diverse_genomes) > n_archs:
        diverse_genomes = diverse_genomes[:n_archs]

    # Ensure we have n_archs via random if needed
    while len(diverse_genomes) < n_archs:
        import dataclasses
        g = Genome.random(dataclasses.asdict(cfg.genome), rng)
        diverse_genomes.append(g)

    # Train each arch on each client
    matrix_acc = np.zeros((n_archs, cfg.dataset.n_clients))
    matrix_f1 = np.zeros((n_archs, cfg.dataset.n_clients))

    records = []

    for arch_idx, genome in enumerate(diverse_genomes):
        # deterministic init per arch
        h = hashlib.md5(json.dumps(genome.to_dict(), sort_keys=True).encode()).hexdigest()
        seed_offset = int(h[:8], 16) % (2**31 - 1)
        torch.manual_seed(seed_offset % (2**31 - 1))
        template = genome_to_model(genome, input_dim, n_classes)
        flat_init = flatten_params(template)
        for cid in range(cfg.dataset.n_clients):
            model = genome_to_model(genome, input_dim, n_classes)
            set_model_params(model, flat_init)
            client = clients[cid]
            client.train(model, epochs=cfg.train.epochs, lr=cfg.train.lr, optimizer_name=cfg.train.optimizer, device=device)
            eval_res = client.evaluate(model, device=device, split="test")
            matrix_acc[arch_idx, cid] = eval_res["accuracy"]
            matrix_f1[arch_idx, cid] = eval_res["macro_f1"]
            records.append({
                "arch_idx": arch_idx,
                "genome": str(genome.to_dict()),
                "L": genome.L,
                "hidden": str(genome.hidden_sizes[:genome.L]),
                "act": genome.activation,
                "dropout": genome.dropout,
                "bn": genome.use_bn,
                "client": cid,
                "accuracy": eval_res["accuracy"],
                "macro_f1": eval_res["macro_f1"],
            })
            if verbose:
                print(f"Arch {arch_idx} ({genome}) on C{cid}: acc {eval_res['accuracy']:.3f} f1 {eval_res['macro_f1']:.3f}")

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "pilot_results.csv", index=False)

    # Matrices
    df_acc = pd.DataFrame(matrix_acc, columns=[f"C{i}" for i in range(cfg.dataset.n_clients)], index=[f"A{i}" for i in range(n_archs)])
    df_f1 = pd.DataFrame(matrix_f1, columns=[f"C{i}" for i in range(cfg.dataset.n_clients)], index=[f"A{i}" for i in range(n_archs)])
    df_acc.to_csv(output_dir / "matrix_accuracy.csv")
    df_f1.to_csv(output_dir / "matrix_f1.csv")

    # Analyze variability
    # Per arch std across clients, per client std across archs, overall
    overall_f1_std = float(matrix_f1.std())
    overall_acc_std = float(matrix_acc.std())
    per_arch_f1_range = (matrix_f1.max(axis=1) - matrix_f1.min(axis=1)).tolist()
    per_client_f1_range = (matrix_f1.max(axis=0) - matrix_f1.min(axis=0)).tolist()
    max_diff = float(matrix_f1.max() - matrix_f1.min())
    # ranking changes: count pairs arch i,j where ranking inverts between clients
    ranking_changes = 0
    total_pairs = 0
    for a1 in range(n_archs):
        for a2 in range(a1+1, n_archs):
            # find if exists clients where ordering flips
            diffs = matrix_f1[a1] - matrix_f1[a2]
            # if not all same sign
            if np.any(diffs > 0) and np.any(diffs < 0):
                ranking_changes += 1
            total_pairs += 1

    # Check triviality: if all accuracies >0.95 or stdev very small, warn
    is_trivial = False
    trivial_reason = ""
    if matrix_acc.mean() > 0.95 and matrix_acc.std() < 0.02:
        is_trivial = True
        trivial_reason = "High accuracy near ceiling with low variance"
    elif overall_f1_std < 0.02:
        is_trivial = True
        trivial_reason = f"Low F1 variance ({overall_f1_std:.3f}) suggests not sensitive to architecture"
    elif max_diff < 0.05:
        is_trivial = True
        trivial_reason = f"Max F1 spread {max_diff:.3f} too small"

    summary = {
        "n_archs": n_archs,
        "n_clients": cfg.dataset.n_clients,
        "overall_f1_mean": float(matrix_f1.mean()),
        "overall_f1_std": overall_f1_std,
        "overall_acc_mean": float(matrix_acc.mean()),
        "overall_acc_std": overall_acc_std,
        "per_arch_f1_range": per_arch_f1_range,
        "per_client_f1_range": per_client_f1_range,
        "max_diff_f1": max_diff,
        "ranking_changes": ranking_changes,
        "total_arch_pairs": total_pairs,
        "ranking_change_ratio": ranking_changes / max(1, total_pairs),
        "is_trivial": is_trivial,
        "trivial_reason": trivial_reason,
        "input_dim": input_dim,
        "n_classes": n_classes,
        "partition_alpha": cfg.dataset.alpha,
    }

    with open(output_dir / "pilot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Heatmap arch x client F1
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix_f1, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(label="F1")
    plt.xlabel("Cliente")
    plt.ylabel("Arquitectura")
    plt.title("Pilot: F1 arquitectura x cliente")
    plt.xticks(range(cfg.dataset.n_clients), [f"C{i}" for i in range(cfg.dataset.n_clients)])
    plt.yticks(range(n_archs), [f"A{i}" for i in range(n_archs)])
    for i in range(n_archs):
        for j in range(cfg.dataset.n_clients):
            plt.text(j, i, f"{matrix_f1[i,j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "pilot_heatmap_f1.png", dpi=150)
    plt.close()

    # Heatmap accuracy
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix_acc, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(label="Accuracy")
    plt.xlabel("Cliente")
    plt.ylabel("Arquitectura")
    plt.title("Pilot: Accuracy arquitectura x cliente")
    plt.xticks(range(cfg.dataset.n_clients), [f"C{i}" for i in range(cfg.dataset.n_clients)])
    plt.yticks(range(n_archs), [f"A{i}" for i in range(n_archs)])
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "pilot_heatmap_acc.png", dpi=150)
    plt.close()

    # Line plot per arch across clients
    plt.figure(figsize=(10, 5))
    for i in range(n_archs):
        plt.plot(range(cfg.dataset.n_clients), matrix_f1[i], marker="o", label=f"A{i} {diverse_genomes[i].hidden_sizes[:diverse_genomes[i].L]}")
    plt.xlabel("Cliente")
    plt.ylabel("F1")
    plt.title("F1 por cliente por arquitectura (ranking changes visible)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "pilot_f1_per_client.png", dpi=150)
    plt.close()

    # Bar per arch range
    plt.figure(figsize=(8, 4))
    plt.bar(range(n_archs), per_arch_f1_range)
    plt.xlabel("Arquitectura")
    plt.ylabel("F1 range (max-min) across clients")
    plt.title("Variabilidad F1 por arquitectura")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "pilot_per_arch_range.png", dpi=150)
    plt.close()

    print(f"\n[PILOT] Summary: {json.dumps(summary, indent=2)}")
    if is_trivial:
        print(f"[PILOT] WARNING: Dataset/proto may be trivial: {trivial_reason}")
    else:
        print("[PILOT] Dataset shows sufficient sensitivity to architecture.")

    return {
        "output_dir": output_dir,
        "summary": summary,
        "matrix_f1": matrix_f1,
        "matrix_acc": matrix_acc,
        "genomes": diverse_genomes,
        "df": df,
    }
