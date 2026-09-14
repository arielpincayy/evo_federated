"""Plotting for similarity-pairing variant (17 types, minimal but scientific)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _maybe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_fitness_vs_generation(output_dir: Path):
    path = output_dir / "generation_metrics.csv"
    if not path.exists():
        path = output_dir / "generations" / "generation_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    plt.figure(figsize=(8,5))
    plt.plot(df["generation"], df["best_fitness"], marker="o", label="best")
    plt.plot(df["generation"], df["mean_fitness"], marker="s", label="mean")
    plt.plot(df["generation"], df["median_fitness"], marker="^", label="median")
    plt.xlabel("Generación")
    plt.ylabel("Fitness (mean accuracy over representatives)")
    plt.title("Fitness vs generación")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "fitness_vs_generation.png", dpi=150)
    plt.close()

    # 3 Best/mean/worst
    plt.figure(figsize=(8,5))
    plt.plot(df["generation"], df["best_fitness"], marker="o", label="best")
    plt.plot(df["generation"], df["mean_fitness"], marker="s", label="mean")
    plt.plot(df["generation"], df["worst_fitness"], marker="x", label="worst")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Best / mean / worst fitness vs generación")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "fitness_best_mean_worst.png", dpi=150)
    plt.close()

    # median + std band
    plt.figure(figsize=(8,5))
    plt.plot(df["generation"], df["mean_fitness"], marker="o", label="mean", color="blue")
    plt.fill_between(df["generation"], df["mean_fitness"]-df["std_fitness"], df["mean_fitness"]+df["std_fitness"], color="blue", alpha=0.2, label="±std")
    plt.plot(df["generation"], df["median_fitness"], marker="^", label="median", color="green")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Mean ± std y median vs generación")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "fitness_mean_std_median.png", dpi=150)
    plt.close()


def plot_accuracy_vs_generation(output_dir: Path):
    # Accuracy media is same as fitness (since fitness = mean accuracy). But we also plot per-rep mean accuracy if available
    path = output_dir / "generation_metrics.csv"
    if not path.exists():
        path = output_dir / "generations" / "generation_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    # per-rep mean accuracy columns
    rep_cols = [c for c in df.columns if c.startswith("mean_acc_rep_")]
    if rep_cols:
        plt.figure(figsize=(8,5))
        for col in rep_cols:
            plt.plot(df["generation"], df[col], marker="o", label=col.replace("mean_acc_rep_","C"))
        plt.xlabel("Generación")
        plt.ylabel("Accuracy promedio del representante")
        plt.title("Accuracy individual por representante vs generación")
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / "accuracy_per_representative.png", dpi=150)
        plt.close()


def plot_fitness_distribution(output_dir: Path):
    path = output_dir / "population_history.csv"
    if not path.exists():
        path = output_dir / "generations" / "population_history.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    # Distribution per generation: boxplot
    gens = sorted(df["generation"].unique())
    if len(gens) == 0:
        return
    data = [df[df["generation"]==g]["mean_accuracy"].values for g in gens]
    plt.figure(figsize=(max(6, len(gens)*0.8), 5))
    plt.boxplot(data, labels=gens, showfliers=False)
    plt.xlabel("Generación")
    plt.ylabel("Fitness (mean accuracy) población")
    plt.title("Distribución de fitness por generación")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "fitness_distribution_per_generation.png", dpi=150)
    plt.close()

    # Violin alternative via hist overlay
    plt.figure(figsize=(8,5))
    for g in gens:
        vals = df[df["generation"]==g]["mean_accuracy"].values
        plt.hist(vals, bins=15, alpha=0.3, label=f"g{g}")
    plt.xlabel("Fitness")
    plt.ylabel("Frecuencia")
    plt.title("Histograma fitness por generación")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "fitness_histogram_per_generation.png", dpi=150)
    plt.close()


def plot_similarity_heatmap(output_dir: Path):
    sim_path = output_dir / "similarity" / "similarity_matrix.npy"
    if not sim_path.exists():
        sim_path = output_dir / "characterization" / "similarity_matrix.npy"
    if not sim_path.exists():
        # fallback divergence
        sim_path = output_dir / "characterization" / "divergence_matrix.npy"
        if not sim_path.exists():
            return
        # this is distance; convert to similarity approx?
    S = np.load(sim_path)
    # if it's distance (max 2, diag 0), we need to detect: distance will have diag 0, similarity diag 1
    if np.allclose(np.diag(S), 0):
        # it's distance, convert similarity = 1 - D for cosine where D in [0,2]
        S = 1.0 - S
        # clip
        S = np.clip(S, -1, 1)
        # diag should be 1
        np.fill_diagonal(S, 1.0)
    plt.figure(figsize=(6,5))
    im = plt.imshow(S, cmap="viridis", vmin=-1, vmax=1)
    plt.colorbar(im, label="Similitud coseno")
    plt.xlabel("Cliente")
    plt.ylabel("Cliente")
    plt.title("Matriz de similitud coseno entre silos (Δ)")
    plt.xticks(range(S.shape[0]), [str(i) for i in range(S.shape[0])])
    plt.yticks(range(S.shape[0]), [str(i) for i in range(S.shape[0])])
    # annotate values
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            plt.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center", color="white" if abs(S[i,j])<0.5 else "black", fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "similarity_heatmap.png", dpi=150)
    plt.close()

    # distance heatmap as well
    dist_path = output_dir / "similarity" / "distance_matrix.npy"
    if dist_path.exists():
        D = np.load(dist_path)
    else:
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
    plt.figure(figsize=(6,5))
    plt.imshow(D, cmap="magma", vmin=0, vmax=2)
    plt.colorbar(label="Distancia coseno")
    plt.title("Matriz de distancia (1 - sim)")
    plt.xlabel("Cliente")
    plt.ylabel("Cliente")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "distance_heatmap.png", dpi=150)
    plt.close()

    # distribution of similarities
    n = S.shape[0]
    triu = S[np.triu_indices(n, k=1)]
    plt.figure(figsize=(6,4))
    plt.hist(triu, bins=15, edgecolor="black")
    plt.xlabel("Similitud")
    plt.ylabel("Frecuencia")
    plt.title("Distribución similitudes entre silos (upper triangle)")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "similarity_distribution.png", dpi=150)
    plt.close()


def plot_pairs_visualization(output_dir: Path):
    sim_path = output_dir / "similarity" / "similarity_matrix.npy"
    pairs_path = output_dir / "similarity" / "pairs.json"
    if not sim_path.exists() or not pairs_path.exists():
        return
    S = np.load(sim_path)
    with open(pairs_path) as f:
        data = json.load(f)
    pairs = data.get("pairs_pos", [])
    singletons = data.get("singletons_pos", [])
    summary = data.get("summary", {})
    n = S.shape[0]
    # Create pair membership visualization: adjacency matrix with pairs highlighted
    mat = np.zeros((n, n))
    for a,b in pairs:
        mat[a,b] = mat[b,a] = 1
        # also similarity values for heatmap overlay maybe
    plt.figure(figsize=(6,5))
    plt.imshow(mat, cmap="Greys", vmin=0, vmax=1)
    plt.colorbar(label="Pareja (1=paired)")
    plt.title("Visualización de pares encontrados (matriz adyacencia)")
    plt.xlabel("Cliente")
    plt.ylabel("Cliente")
    for a,b in pairs:
        plt.text(b, a, f"{S[a,b]:.2f}", ha="center", va="center", color="red", fontsize=8, fontweight="bold")
        plt.text(a, b, f"{S[a,b]:.2f}", ha="center", va="center", color="red", fontsize=8, fontweight="bold")
    plt.xticks(range(n))
    plt.yticks(range(n))
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "pairs_visualization.png", dpi=150)
    plt.close()

    # Graph style pairs: simple line plot of client indices vs pair id
    if pairs:
        plt.figure(figsize=(8,3))
        for idx, (a,b) in enumerate(pairs):
            plt.plot([a, b], [idx, idx], marker="o", linewidth=3, label=f"pair {idx} sim {S[a,b]:.2f}")
        if singletons:
            for s in singletons:
                plt.scatter([s], [len(pairs)], color="red", marker="x", s=100, label=f"singleton {s}")
        plt.yticks(range(len(pairs)+(1 if singletons else 0)), [f"P{i}" for i in range(len(pairs))]+ (["singleton"] if singletons else []))
        plt.xlabel("Cliente id (pos)")
        plt.title("Parejas maximizando similitud (global matching)")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / "pairs_linear.png", dpi=150)
        plt.close()


def plot_intra_pair_similarity(output_dir: Path):
    sim_path = output_dir / "similarity" / "similarity_matrix.npy"
    pairs_path = output_dir / "similarity" / "pairs.json"
    if not sim_path.exists() or not pairs_path.exists():
        return
    S = np.load(sim_path)
    with open(pairs_path) as f:
        data = json.load(f)
    pairs = data.get("pairs_pos", [])
    if not pairs:
        return
    sims = [float(S[a,b]) for a,b in pairs]
    plt.figure(figsize=(8,4))
    plt.bar(range(len(sims)), sims, tick_label=[f"{a}-{b}" for a,b in pairs])
    plt.xlabel("Pareja")
    plt.ylabel("Similitud intra-pair")
    plt.title("Similitud intra-pair por pareja")
    plt.axhline(np.mean(sims), color="red", linestyle="--", label=f"mean {np.mean(sims):.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "intra_pair_similarity.png", dpi=150)
    plt.close()

    # also show compared to random baseline: overall mean similarity
    n = S.shape[0]
    all_sims = S[np.triu_indices(n,k=1)]
    plt.figure(figsize=(6,4))
    plt.hist(all_sims, bins=15, alpha=0.5, label="todas las parejas posibles", edgecolor="black")
    plt.hist(sims, bins=15, alpha=0.7, label="pares elegidos", edgecolor="black")
    plt.xlabel("Similitud")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.title("Distribución: elegidos vs todas")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "intra_pair_vs_all.png", dpi=150)
    plt.close()


def plot_final_per_silo(output_dir: Path):
    path = output_dir / "final_evaluation" / "per_client_metrics.csv"
    if not path.exists():
        path = output_dir / "final_evaluation.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    # take best (rank 0)
    row = df.iloc[0]
    client_cols = [c for c in df.columns if c.startswith("acc_client_")]
    if not client_cols:
        return
    accs = [float(row[c]) for c in client_cols]
    clients = [c.replace("acc_client_","C") for c in client_cols]
    plt.figure(figsize=(8,4))
    plt.bar(clients, accs)
    plt.xlabel("Silo")
    plt.ylabel("Accuracy")
    plt.title("Accuracy final por silo del mejor individuo (evaluación full)")
    plt.ylim(0,1)
    for i, v in enumerate(accs):
        plt.text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "final_accuracy_per_silo.png", dpi=150)
    plt.close()

    # loss per silo
    loss_cols = [c for c in df.columns if c.startswith("loss_client_")]
    if loss_cols:
        losses = [float(row[c]) for c in loss_cols]
        plt.figure(figsize=(8,4))
        plt.bar(clients, losses)
        plt.xlabel("Silo")
        plt.ylabel("Loss")
        plt.title("Loss final por silo")
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / "final_loss_per_silo.png", dpi=150)
        plt.close()

    # distribution final accuracy between silos (hist/box)
    plt.figure(figsize=(6,4))
    plt.boxplot(accs, vert=True)
    plt.ylabel("Accuracy")
    plt.title("Distribución accuracy final entre silos (boxplot)")
    plt.xticks([1], ["best model"])
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "final_accuracy_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(accs, bins=10, edgecolor="black")
    plt.xlabel("Accuracy")
    plt.ylabel("Frecuencia")
    plt.title("Histograma accuracy final entre silos")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "final_accuracy_histogram.png", dpi=150)
    plt.close()

    # F1 per silo if available
    f1_cols = [c for c in df.columns if c.startswith("f1_client_")]
    if f1_cols:
        f1s = [float(row[c]) for c in f1_cols]
        plt.figure(figsize=(8,4))
        plt.bar(clients, f1s, color="green")
        plt.xlabel("Silo")
        plt.ylabel("F1 macro")
        plt.title("F1 final por silo")
        plt.ylim(0,1)
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / "final_f1_per_silo.png", dpi=150)
        plt.close()


def plot_worst_client_vs_generation(output_dir: Path):
    # Use generation_metrics worst_fitness as proxy for worst rep accuracy
    path = output_dir / "generation_metrics.csv"
    if not path.exists():
        path = output_dir / "generations" / "generation_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or "worst_fitness" not in df.columns:
        return
    plt.figure(figsize=(8,4))
    plt.plot(df["generation"], df["worst_fitness"], marker="o", color="red", label="worst (rep)")
    plt.plot(df["generation"], df["best_fitness"], marker="o", color="green", label="best")
    plt.xlabel("Generación")
    plt.ylabel("Accuracy (worst / best)")
    plt.title("Worst-client accuracy (proxy rep) vs generación")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "worst_vs_generation.png", dpi=150)
    plt.close()


def plot_cost(output_dir: Path):
    cost_path = output_dir / "cost.json"
    gen_path = output_dir / "generation_metrics.csv"
    if not gen_path.exists():
        gen_path = output_dir / "generations" / "generation_metrics.csv"
    # cumulative evals vs generation
    if gen_path.exists():
        df = pd.read_csv(gen_path)
        if not df.empty and "eval_count" in df.columns:
            plt.figure(figsize=(8,4))
            plt.plot(df["generation"], df["eval_count"], marker="o", label="realizadas (representatives)")
            if "theoretical_full_cumulative" in df.columns:
                plt.plot(df["generation"], df["theoretical_full_cumulative"], marker="s", label="full (teórico)")
            else:
                # compute theoretical: generation * pop * N
                try:
                    import json
                    with open(cost_path) as f:
                        cost = json.load(f)
                    theoretical = df["generation"] * cost["pop_size"] * cost["n_clients"]
                    plt.plot(df["generation"], theoretical, marker="s", label="full (teórico)")
                except Exception:
                    pass
            plt.xlabel("Generación")
            plt.ylabel("Evaluaciones acumuladas")
            plt.title("Coste acumulado de evaluaciones vs generación")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "plots" / "cost_cumulative_evals.png", dpi=150)
            plt.close()

            # comparison bar realized vs full
            try:
                with open(cost_path) as f:
                    cost = json.load(f)
                plt.figure(figsize=(6,4))
                cats = ["realizadas", "full teórico"]
                vals = [cost["total_search_evaluations"], cost["theoretical_full_evaluations"]]
                plt.bar(cats, vals)
                plt.ylabel("Evaluaciones")
                plt.title(f"Evaluaciones totales (ahorro {cost['reduction_percent_vs_full']:.1f}%)")
                for i, v in enumerate(vals):
                    plt.text(i, v, str(v), ha="center", va="bottom")
                plt.tight_layout()
                plt.savefig(output_dir / "plots" / "evaluations_realized_vs_full.png", dpi=150)
                plt.close()
            except Exception:
                pass

            # reduction percent vs gen
            if "reduction_percent" in df.columns:
                plt.figure(figsize=(8,4))
                plt.plot(df["generation"], df["reduction_percent"], marker="o", color="purple")
                plt.xlabel("Generación")
                plt.ylabel("Reducción % vs full")
                plt.title("Reducción evaluaciones vs generación")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "plots" / "reduction_vs_generation.png", dpi=150)
                plt.close()

    # time accumulated
    if gen_path.exists():
        df = pd.read_csv(gen_path)
        if not df.empty and "elapsed" in df.columns:
            plt.figure(figsize=(8,4))
            plt.plot(df["generation"], df["elapsed"], marker="o", color="orange")
            plt.xlabel("Generación")
            plt.ylabel("Tiempo acumulado (s)")
            plt.title("Tiempo acumulado vs generación")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "plots" / "time_cumulative.png", dpi=150)
            plt.close()

    # cost breakdown pie
    if cost_path.exists():
        try:
            with open(cost_path) as f:
                cost = json.load(f)
            tb = cost.get("time_breakdown", {})
            if tb:
                labels = list(tb.keys())
                vals = list(tb.values())
                plt.figure(figsize=(6,6))
                plt.pie(vals, labels=labels, autopct="%1.1f%%")
                plt.title("Desglose tiempo")
                plt.tight_layout()
                plt.savefig(output_dir / "plots" / "time_breakdown.png", dpi=150)
                plt.close()
        except Exception:
            pass


def plot_class_distribution(output_dir: Path):
    path = output_dir / "dataset" / "class_distribution.csv"
    if not path.exists():
        return
    import pandas as pd
    df = pd.read_csv(path, index_col=0)
    plt.figure(figsize=(8,4))
    sizes = df.sum(axis=1)
    plt.bar(sizes.index.astype(str), sizes.values)
    plt.xlabel("Cliente")
    plt.ylabel("N muestras")
    plt.title("Muestras por cliente")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "samples_per_client.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10,5))
    df_norm = df.div(df.sum(axis=1), axis=0)
    df_norm.plot(kind="bar", stacked=True, figsize=(10,5), colormap="tab20")
    plt.xlabel("Cliente")
    plt.ylabel("Proporción")
    plt.title("Distribución de clases por cliente (normalizada)")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "class_dist_per_client.png", dpi=150)
    plt.close()


def generate_all_plots_similarity(output_dir: Path, cfg=None):
    output_dir = Path(output_dir)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    for fn in [
        plot_class_distribution,
        plot_similarity_heatmap,
        plot_pairs_visualization,
        plot_intra_pair_similarity,
        plot_fitness_vs_generation,
        plot_accuracy_vs_generation,
        plot_fitness_distribution,
        plot_final_per_silo,
        plot_worst_client_vs_generation,
        plot_cost,
    ]:
        try:
            fn(output_dir)
        except Exception as e:
            print(f"[WARN] plot {fn.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    print("Plots similarity generated (best effort).")


def generate_aggregate_plots(aggregate_dir: Path):
    """Across seeds: 14-17."""
    aggregate_dir = Path(aggregate_dir)
    # Expect seeds subdirs with generation_metrics.csv and summary.json
    # Also final_evaluation per seed
    # We will discover seeds
    import glob as glob_mod
    seed_dirs = [p for p in aggregate_dir.glob("seed_*") if p.is_dir()]
    if not seed_dirs:
        # may be inside phase1_validation/aggregate?
        return
    # 14 convergence per seed
    plt.figure(figsize=(8,5))
    for sd in sorted(seed_dirs):
        gm = sd / "generation_metrics.csv"
        if not gm.exists():
            gm = sd / "generations" / "generation_metrics.csv"
        if not gm.exists():
            continue
        df = pd.read_csv(gm)
        if df.empty:
            continue
        plt.plot(df["generation"], df["best_fitness"], marker="o", alpha=0.7, label=sd.name)
    plt.xlabel("Generación")
    plt.ylabel("Best fitness")
    plt.title("Curva de convergencia distintas seeds")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(aggregate_dir / "convergence_per_seed.png", dpi=150)
    plt.close()

    # 15 media y banda dispersión entre seeds
    # collect best per generation across seeds
    all_best = {}
    for sd in seed_dirs:
        gm = sd / "generation_metrics.csv"
        if not gm.exists():
            gm = sd / "generations" / "generation_metrics.csv"
        if not gm.exists():
            continue
        df = pd.read_csv(gm)
        for _, row in df.iterrows():
            g = int(row["generation"])
            all_best.setdefault(g, []).append(float(row["best_fitness"]))
    if all_best:
        gens = sorted(all_best.keys())
        means = [np.mean(all_best[g]) for g in gens]
        stds = [np.std(all_best[g]) for g in gens]
        medians = [np.median(all_best[g]) for g in gens]
        plt.figure(figsize=(8,5))
        plt.plot(gens, means, marker="o", label="mean")
        plt.fill_between(gens, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.2, label="±std")
        plt.plot(gens, medians, marker="s", label="median")
        plt.xlabel("Generación")
        plt.ylabel("Best fitness")
        plt.title("Media y banda dispersión entre seeds (best fitness)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(aggregate_dir / "mean_band_across_seeds.png", dpi=150)
        plt.close()

    # 16 boxplot accuracy final entre seeds
    accs = []
    worsts = []
    labels = []
    for sd in sorted(seed_dirs):
        fp = sd / "final_evaluation" / "per_client_metrics.csv"
        if not fp.exists():
            fp = sd / "final_evaluation.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if df.empty:
            continue
        row = df.iloc[0]
        # mean accuracy
        accs.append(float(row["mean_accuracy"]))
        worsts.append(float(row["worst_client_accuracy"]))
        labels.append(sd.name)
    if accs:
        plt.figure(figsize=(8,4))
        plt.boxplot(accs, vert=True)
        plt.ylabel("Mean accuracy (full)")
        plt.title("Boxplot accuracy final entre seeds")
        plt.xticks([1], ["seeds"])
        # also show points
        plt.scatter([1]*len(accs), accs, color="red", alpha=0.7)
        plt.tight_layout()
        plt.savefig(aggregate_dir / "boxplot_accuracy_across_seeds.png", dpi=150)
        plt.close()

        # also per seed bar
        plt.figure(figsize=(max(6, len(accs)*0.8),4))
        plt.bar(labels, accs)
        plt.xlabel("Seed")
        plt.ylabel("Mean accuracy")
        plt.title("Accuracy final por seed")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(aggregate_dir / "accuracy_per_seed_bar.png", dpi=150)
        plt.close()

    if worsts:
        plt.figure(figsize=(8,4))
        plt.boxplot(worsts, vert=True)
        plt.ylabel("Worst-client accuracy")
        plt.title("Boxplot worst-client accuracy entre seeds")
        plt.xticks([1], ["seeds"])
        plt.scatter([1]*len(worsts), worsts, color="red", alpha=0.7)
        plt.tight_layout()
        plt.savefig(aggregate_dir / "boxplot_worst_across_seeds.png", dpi=150)
        plt.close()

        plt.figure(figsize=(max(6, len(worsts)*0.8),4))
        plt.bar(labels, worsts)
        plt.xlabel("Seed")
        plt.ylabel("Worst accuracy")
        plt.title("Worst-client accuracy por seed")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(aggregate_dir / "worst_per_seed_bar.png", dpi=150)
        plt.close()

    # also std between clients across seeds
    # gather per seed std
    stds_final = []
    for sd in sorted(seed_dirs):
        fp = sd / "final_evaluation" / "per_client_metrics.csv"
        if not fp.exists():
            fp = sd / "final_evaluation.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if df.empty:
            continue
        row = df.iloc[0]
        stds_final.append(float(row["std_accuracy"]))
    if stds_final:
        plt.figure(figsize=(8,4))
        plt.hist(stds_final, bins=10, edgecolor="black")
        plt.xlabel("Std accuracy across clients")
        plt.ylabel("Frecuencia seeds")
        plt.title("Distribución std entre clientes across seeds")
        plt.tight_layout()
        plt.savefig(aggregate_dir / "std_across_seeds_hist.png", dpi=150)
        plt.close()
