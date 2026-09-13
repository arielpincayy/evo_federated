"""Plotting utilities: generate required plots (20 requested)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json


def _maybe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(output_dir: Path):
    path = output_dir / "dataset" / "class_distribution.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, index_col=0)
    # 1. cantidad muestras por cliente
    plt.figure(figsize=(8, 4))
    sizes = df.sum(axis=1)
    plt.bar(sizes.index.astype(str), sizes.values)
    plt.xlabel("Cliente")
    plt.ylabel("N muestras")
    plt.title("Muestras por cliente")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "samples_per_client.png", dpi=150)
    plt.savefig(output_dir / "plots" / "samples_per_client.pdf")
    plt.close()

    # 2. distribución clases por cliente (stacked)
    plt.figure(figsize=(10, 5))
    df_norm = df.div(df.sum(axis=1), axis=0)
    df_norm.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="tab20")
    plt.xlabel("Cliente")
    plt.ylabel("Proporción")
    plt.title("Distribución de clases por cliente (normalizada)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "class_dist_per_client.png", dpi=150)
    plt.savefig(output_dir / "plots" / "class_dist_per_client.pdf")
    plt.close()

    # 3. heatmap cliente x clase
    plt.figure(figsize=(8, 6))
    plt.imshow(df.values, aspect="auto", cmap="viridis")
    plt.colorbar(label="N muestras")
    plt.xlabel("Clase")
    plt.ylabel("Cliente")
    plt.title("Heatmap cliente x clase")
    plt.xticks(range(df.shape[1]), df.columns, rotation=45)
    plt.yticks(range(df.shape[0]), df.index)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "heatmap_client_class.png", dpi=150)
    plt.savefig(output_dir / "plots" / "heatmap_client_class.pdf")
    plt.close()


def plot_divergence(output_dir: Path):
    d_path = output_dir / "characterization" / "divergence_matrix.npy"
    if not d_path.exists():
        return
    D = np.load(d_path)
    plt.figure(figsize=(6, 5))
    plt.imshow(D, cmap="magma", vmin=0, vmax=D.max())
    plt.colorbar(label="Distancia coseno")
    plt.xlabel("Cliente")
    plt.ylabel("Cliente")
    plt.title("Matriz de divergencia (learning-update)")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "divergence_matrix.png", dpi=150)
    plt.savefig(output_dir / "plots" / "divergence_matrix.pdf")
    plt.close()

    # distribución distancias
    triu = D[np.triu_indices(D.shape[0], k=1)]
    plt.figure(figsize=(6, 4))
    plt.hist(triu, bins=15, edgecolor="black")
    plt.xlabel("Distancia")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de distancias entre clientes")
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "distance_distribution.png", dpi=150)
    plt.savefig(output_dir / "plots" / "distance_distribution.pdf")
    plt.close()

    # true TV if exists
    true_path = output_dir / "characterization" / "true_TV_matrix.npy"
    if true_path.exists():
        T = np.load(true_path)
        plt.figure(figsize=(6, 5))
        plt.imshow(T, cmap="magma")
        plt.colorbar(label="TV distancia")
        plt.title("True TV (distribución clases)")
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / "true_TV_matrix.png", dpi=150)
        plt.close()
        # scatter correlation
        plt.figure(figsize=(5, 5))
        n = D.shape[0]
        idx = np.triu_indices(n, k=1)
        plt.scatter(T[idx], D[idx])
        plt.xlabel("True TV")
        plt.ylabel("Delta cosine distance")
        plt.title("Proxy vs True divergence")
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / "proxy_vs_true.png", dpi=150)
        plt.close()


def plot_hv(output_dir: Path):
    hv_path = output_dir / "hypervolume_history.csv"
    if not hv_path.exists():
        return
    df = pd.read_csv(hv_path)
    # 16 HV por generación per strategy
    plt.figure(figsize=(8, 5))
    for strat, g in df.groupby("strategy"):
        g_sorted = g.sort_values("generation")
        plt.plot(g_sorted["generation"], g_sorted["hv"], marker="o", label=strat)
    plt.xlabel("Generación")
    plt.ylabel("Hypervolume")
    plt.title("HV por generación (interno)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "hv_per_generation.png", dpi=150)
    plt.savefig(output_dir / "plots" / "hv_per_generation.pdf")
    plt.close()

    # 18 HV vs evaluaciones acumuladas
    plt.figure(figsize=(8, 5))
    for strat, g in df.groupby("strategy"):
        g_sorted = g.sort_values("eval_count")
        plt.plot(g_sorted["eval_count"], g_sorted["hv"], marker="o", label=strat)
    plt.xlabel("Evaluaciones acumuladas (cliente-arquitectura)")
    plt.ylabel("HV")
    plt.title("HV vs evaluaciones acumuladas")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "hv_vs_evals.png", dpi=150)
    plt.savefig(output_dir / "plots" / "hv_vs_evals.pdf")
    plt.close()

    # 19 HV vs tiempo
    if "elapsed" in df.columns:
        plt.figure(figsize=(8, 5))
        for strat, g in df.groupby("strategy"):
            g_sorted = g.sort_values("elapsed")
            plt.plot(g_sorted["elapsed"], g_sorted["hv"], marker="o", label=strat)
        plt.xlabel("Tiempo acumulado (s)")
        plt.ylabel("HV")
        plt.title("HV vs tiempo")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / "hv_vs_time.png", dpi=150)
        plt.savefig(output_dir / "plots" / "hv_vs_time.pdf")
        plt.close()

    # per strategy also
    for strat, g in df.groupby("strategy"):
        plt.figure(figsize=(6, 4))
        g_sorted = g.sort_values("generation")
        plt.plot(g_sorted["generation"], g_sorted["hv"], marker="o")
        plt.xlabel("Generación")
        plt.ylabel("HV")
        plt.title(f"HV por generación - {strat}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / f"hv_{strat}.png", dpi=150)
        plt.close()


def plot_pareto(output_dir: Path):
    import glob as g
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    # Pareto fronts
    for strat_path in (output_dir).glob("strategy_*/final_pareto.csv"):
        strat = strat_path.parent.name.replace("strategy_", "")
        df = pd.read_csv(strat_path)
        if df.empty:
            continue
        plt.figure(figsize=(6, 5))
        # objectives are F0, F1 (minimization). Convert to F1_i_est, F1_j_est for interpret
        plt.scatter(df["F1_i_est"], df["F1_j_est"], c="blue", label="Pareto")
        plt.xlabel("F1 cliente i (o mean)")
        plt.ylabel("F1 cliente j (o min)")
        plt.title(f"Frente de Pareto final - {strat}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / f"pareto_{strat}.png", dpi=150)
        plt.savefig(output_dir / "plots" / f"pareto_{strat}.pdf")
        plt.close()

    # Combined pareto? Using exhaustive validation for common view
    exh_path = output_dir / "exhaustive_validation.csv"
    if exh_path.exists():
        df = pd.read_csv(exh_path)
        if not df.empty:
            plt.figure(figsize=(8, 5))
            for strat, g in df.groupby("strategy"):
                plt.scatter(g["mean_f1"], g["min_f1"], label=strat, alpha=0.7)
            plt.xlabel("mean F1 (exhaustivo)")
            plt.ylabel("min F1 (exhaustivo)")
            plt.title("Validación exhaustiva: mean vs min F1 por estrategia")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "plots" / "exhaustive_mean_vs_min.png", dpi=150)
            plt.savefig(output_dir / "plots" / "exhaustive_mean_vs_min.pdf")
            plt.close()

            # rendimiento por cliente de finalistas (heatmap)
            # For each strategy, plot per client F1
            for strat, g in df.groupby("strategy"):
                client_cols = [c for c in df.columns if c.startswith("f1_client_")]
                if not client_cols:
                    continue
                # Take best by mean_f1? Show all as heatmap
                mat = g[client_cols].values
                plt.figure(figsize=(10, 4))
                plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
                plt.colorbar(label="F1")
                plt.xlabel("Cliente")
                plt.ylabel("Arquitectura pareto")
                plt.title(f"Rendimiento por cliente - {strat}")
                plt.xticks(range(len(client_cols)), [c.replace("f1_client_","C") for c in client_cols])
                plt.tight_layout()
                plt.savefig(output_dir / "plots" / f"per_client_{strat}.png", dpi=150)
                plt.close()


def plot_savings(output_dir: Path):
    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        return
    df = pd.read_csv(summary_path)
    if df.empty:
        return
    # ahorro vs rendimiento
    plt.figure(figsize=(8, 5))
    # x = eval_count, y = hv_common
    for _, row in df.iterrows():
        plt.scatter(row["eval_count"], row["hv_common_validation"], label=row["strategy"], s=100)
        plt.text(row["eval_count"], row["hv_common_validation"], row["strategy"], fontsize=9, ha="right")
    plt.xlabel("Evaluaciones totales")
    plt.ylabel("HV común validación")
    plt.title("Ahorro computacional vs HV común")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "saving_vs_hv.png", dpi=150)
    plt.savefig(output_dir / "plots" / "saving_vs_hv.pdf")
    plt.close()

    # bar chart saving
    if "saving_percent" in df.columns:
        plt.figure(figsize=(8, 4))
        plt.bar(df["strategy"], df["saving_percent"])
        plt.ylabel("Ahorro vs FULL (%)")
        plt.title("Ahorro computacional por estrategia")
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / "saving_bar.png", dpi=150)
        plt.close()


def plot_pairing(output_dir: Path):
    # pairs selected por generación per strategy
    for hist_path in (output_dir / "generations").glob("pair_history_*.csv"):
        strat = hist_path.stem.replace("pair_history_", "")
        df = pd.read_csv(hist_path)
        if df.empty:
            continue
        # count participation per client
        plt.figure(figsize=(8, 4))
        counts = pd.concat([df["ref"], df["counterpart"]]).value_counts().sort_index()
        plt.bar(counts.index.astype(str), counts.values)
        plt.xlabel("Cliente")
        plt.ylabel("Participaciones")
        plt.title(f"Participación clientes - {strat}")
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / f"pair_participation_{strat}.png", dpi=150)
        plt.close()

        # distance covered per generation?
        if "D_ij" in df.columns:
            plt.figure(figsize=(8, 4))
            # avg D per generation
            gen_avg = df.groupby("generation")["D_ij"].mean()
            plt.plot(gen_avg.index, gen_avg.values, marker="o")
            plt.xlabel("Generación")
            plt.ylabel("D_ij promedio")
            plt.title(f"Distancia promedio por generación - {strat}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "plots" / f"pair_distance_{strat}.png", dpi=150)
            plt.close()


def plot_fitness_approx_vs_exhaustive(output_dir: Path):
    # correlation between partial fitness and exhaustive mean
    # Need to join? For each pareto arch, we have partial F1_i_est vs exhaustive mean
    # We'll plot for each strategy where available
    exh_path = output_dir / "exhaustive_validation.csv"
    if not exh_path.exists():
        return
    df_exh = pd.read_csv(exh_path)
    # For each strategy, load its pareto
    for strat in df_exh["strategy"].unique():
        pareto_path = output_dir / f"strategy_{strat}" / "final_pareto.csv"
        if not pareto_path.exists():
            continue
        df_pareto = pd.read_csv(pareto_path)
        # Need matching pareto_idx correspondence: df_exh has pareto_idx aligning with df_pareto order (same index)
        # Use order
        if len(df_pareto) != len(df_exh[df_exh["strategy"]==strat]):
            continue
        sub = df_exh[df_exh["strategy"]==strat].sort_values("pareto_idx")
        df_pareto = df_pareto.sort_values("idx")
        # Approx fitness: mean of F1_i_est, F1_j_est? For full mean is separate
        approx = (df_pareto["F1_i_est"].values + df_pareto["F1_j_est"].values)/2
        exhaustive = sub["mean_f1"].values
        plt.figure(figsize=(5,5))
        plt.scatter(approx, exhaustive)
        plt.xlabel("Fitness parcial promedio (F1_i+F1_j)/2")
        plt.ylabel("Fitness exhaustivo mean F1")
        plt.title(f"Aprox vs Exhaustivo - {strat}")
        # add correlation
        try:
            from scipy.stats import spearmanr
            corr,_ = spearmanr(approx, exhaustive)
            plt.text(0.05, 0.95, f"Spearman={corr:.2f}", transform=plt.gca().transAxes)
        except Exception:
            pass
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / f"approx_vs_exhaustive_{strat}.png", dpi=150)
        plt.close()


def _plot_evolution_and_diversity(output_dir: Path):
    # 7,8: evolución mejor/mediano y diversidad
    # Use population_history aggregated if exists
    pop_path = output_dir / "population_history.csv"
    if not pop_path.exists():
        pop_path = output_dir / "generations" / "population_history.csv"
        if not pop_path.exists():
            return
    try:
        df = pd.read_csv(pop_path)
        # Determine best per generation per strategy via max of F1_i or mean?
        # Population history contains F1_i, F1_j or F1_mean etc. Try to harmonize
        # Plot best median across generations if possible
        # Also diversity: std of genome hidden sizes or variation in F
        if "F1_i" in df.columns:
            # For 2-client strategies, use mean of F1_i and F1_j as proxy
            df["mean_F"] = (df["F1_i"].fillna(0) + df["F1_j"].fillna(0)) / 2
        elif "F1_mean" in df.columns:
            df["mean_F"] = df["F1_mean"]
        else:
            return
        for strat, g in df.groupby("strategy"):
            gen_best = g.groupby("generation")["mean_F"].max()
            gen_median = g.groupby("generation")["mean_F"].median()
            plt.figure(figsize=(8,4))
            plt.plot(gen_best.index, gen_best.values, marker="o", label="mejor")
            plt.plot(gen_median.index, gen_median.values, marker="s", label="mediana")
            plt.xlabel("Generación")
            plt.ylabel("F1 promedio (proxy)")
            plt.title(f"Evolución mejor/mediana - {strat}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "plots" / f"evolution_best_median_{strat}.png", dpi=150)
            plt.close()
            # diversity: std of F
            gen_std = g.groupby("generation")["mean_F"].std()
            plt.figure(figsize=(8,4))
            plt.plot(gen_std.index, gen_std.values, marker="o")
            plt.xlabel("Generación")
            plt.ylabel("std F1")
            plt.title(f"Diversidad población - {strat}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "plots" / f"diversity_{strat}.png", dpi=150)
            plt.close()
        # cumulative evaluations & time
        hv_path = output_dir / "hypervolume_history.csv"
        if hv_path.exists():
            dfhv = pd.read_csv(hv_path)
            # 9 cumulative evals per generation
            plt.figure(figsize=(8,4))
            for strat, g in dfhv.groupby("strategy"):
                plt.plot(g["generation"], g["eval_count"], marker="o", label=strat)
            plt.xlabel("Generación")
            plt.ylabel("Evaluaciones acumuladas")
            plt.title("Evaluaciones acumuladas por generación")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "plots" / "cumulative_evals.png", dpi=150)
            plt.close()
            # 10 tiempo acumulado
            plt.figure(figsize=(8,4))
            for strat, g in dfhv.groupby("strategy"):
                plt.plot(g["generation"], g["elapsed"], marker="o", label=strat)
            plt.xlabel("Generación")
            plt.ylabel("Tiempo acumulado (s)")
            plt.title("Tiempo acumulado por generación")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "plots" / "cumulative_time.png", dpi=150)
            plt.close()
        # 20 hv distribution placeholder (single run -> box of final hv per strategy)
        if hv_path.exists():
            dfhv = pd.read_csv(hv_path)
            finals = dfhv.groupby("strategy").tail(1)
            plt.figure(figsize=(8,4))
            plt.bar(finals["strategy"], finals["hv"])
            plt.ylabel("HV final")
            plt.title("HV final por estrategia (distribución con 1 seed)")
            plt.tight_layout()
            plt.savefig(output_dir / "plots" / "hv_final_distribution.png", dpi=150)
            plt.close()
    except Exception as e:
        print(f"_plot_evolution_and_diversity failed: {e}")


def plot_heatmap_arch_client(output_dir: Path):
    import glob
    exh_path = output_dir / "exhaustive_validation.csv"
    if exh_path.exists():
        df = pd.read_csv(exh_path)
        if not df.empty:
            client_cols = [c for c in df.columns if c.startswith("f1_client_")]
            if client_cols:
                # heatmap aggregated per strategy mean per client? Or matrix arch x client for each strategy
                # Create combined heatmap: rows strategies*archs, cols clients
                # Normalize strategy grouping visually
                # We'll produce separate heatmaps per strategy as before, but also combined
                try:
                    # Combined matrix
                    plt.figure(figsize=(10, 6))
                    mat = df[client_cols].values
                    plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
                    plt.colorbar(label="F1")
                    plt.xlabel("Client")
                    plt.ylabel("Architecture (grouped by strategy)")
                    plt.title("Architecture x Client F1 (Pareto, exhaustive)")
                    plt.xticks(range(len(client_cols)), [c.replace("f1_client_", "C") for c in client_cols])
                    # y ticks: show strategy groups
                    yticks = []
                    ylabels = []
                    pos = 0
                    for strat, g in df.groupby("strategy"):
                        mid = pos + len(g)//2
                        yticks.append(mid)
                        ylabels.append(strat)
                        pos += len(g)
                    plt.yticks(yticks, ylabels)
                    plt.tight_layout()
                    plt.savefig(output_dir / "plots" / "heatmap_arch_client.png", dpi=150)
                    plt.savefig(output_dir / "plots" / "heatmap_arch_client.pdf")
                    plt.close()
                except Exception as e:
                    print(f"heatmap_arch_client failed: {e}")
    # Hall of fame heatmap
    hof_path = output_dir / "hall_of_fame.csv"
    if hof_path.exists():
        try:
            df = pd.read_csv(hof_path)
            if not df.empty:
                client_cols = [c for c in df.columns if c.startswith("f1_client_")]
                if client_cols:
                    plt.figure(figsize=(12, max(4, len(df)*0.35)))
                    mat = df[client_cols].values
                    plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
                    plt.colorbar(label="F1")
                    plt.xlabel("Client")
                    plt.ylabel("Hall of Fame architecture")
                    plt.title("Hall of Fame x Client F1 (exhaustive)")
                    plt.xticks(range(len(client_cols)), [c.replace("f1_client_", "C") for c in client_cols])
                    plt.yticks(range(len(df)), df["genome_hash"].astype(str))
                    plt.tight_layout()
                    plt.savefig(output_dir / "plots" / "hall_of_fame_heatmap.png", dpi=150)
                    plt.close()
        except Exception as e:
            print(f"hof heatmap failed: {e}")
    # Checkpoint progression: mean f1 vs generation
    ckpt_path = output_dir / "checkpoints" / "checkpoints_summary.csv"
    if ckpt_path.exists():
        try:
            df = pd.read_csv(ckpt_path)
            if not df.empty:
                plt.figure(figsize=(8,5))
                for strat, g in df.groupby("strategy"):
                    g_sorted = g.sort_values("checkpoint_generation")
                    plt.plot(g_sorted["checkpoint_generation"], g_sorted["mean_of_mean_f1"], marker="o", label=f"{strat} mean")
                    plt.plot(g_sorted["checkpoint_generation"], g_sorted["best_mean_f1"], marker="s", linestyle="--", label=f"{strat} best")
                plt.xlabel("Generation (checkpoint)")
                plt.ylabel("Mean F1 (exhaustive)")
                plt.title("Global mean F1 vs generation (checkpoints)")
                plt.legend(fontsize=8)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "plots" / "checkpoint_mean_f1_vs_gen.png", dpi=150)
                plt.close()
                # worst
                plt.figure(figsize=(8,5))
                for strat, g in df.groupby("strategy"):
                    g_sorted = g.sort_values("checkpoint_generation")
                    # best_min_f1 per checkpoint
                    if "best_min_f1" in g.columns:
                        plt.plot(g_sorted["checkpoint_generation"], g_sorted["best_min_f1"], marker="o", label=strat)
                plt.xlabel("Generation")
                plt.ylabel("Best worst-client F1")
                plt.title("Worst-client F1 vs generation")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "plots" / "checkpoint_worst_f1_vs_gen.png", dpi=150)
                plt.close()
                # HV
                hv_path = output_dir / "checkpoints" / "hv_vs_generation_checkpoints.csv"
                if hv_path.exists():
                    dfhv = pd.read_csv(hv_path)
                    plt.figure(figsize=(8,5))
                    for strat, g in dfhv.groupby("strategy"):
                        g_sorted = g.sort_values("generation")
                        plt.plot(g_sorted["generation"], g_sorted["hv_common"], marker="o", label=strat)
                    plt.xlabel("Generation")
                    plt.ylabel("Common HV (exhaustive)")
                    plt.title("Global HV vs generation (checkpoints)")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_dir / "plots" / "checkpoint_hv_vs_gen.png", dpi=150)
                    plt.close()
        except Exception as e:
            print(f"checkpoint plots failed: {e}")
    # Global Pareto front per strategy
    for strat_dir in output_dir.glob("strategy_*"):
        gp = strat_dir / "global_pareto.csv"
        exh = strat_dir / "exhaustive_validation.csv"
        if exh.exists():
            try:
                df = pd.read_csv(exh)
                if not df.empty and "is_global_pareto" in df.columns:
                    plt.figure(figsize=(6,5))
                    plt.scatter(df["mean_f1"], df["min_f1"], c="gray", alpha=0.5, label="Pareto")
                    df_g = df[df["is_global_pareto"]]
                    plt.scatter(df_g["mean_f1"], df_g["min_f1"], c="red", label="Global Pareto")
                    plt.xlabel("Mean F1")
                    plt.ylabel("Worst-client F1")
                    plt.title(f"Global Pareto (mean vs worst) - {strat_dir.name}")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(output_dir / "plots" / f"global_pareto_{strat_dir.name}.png", dpi=150)
                    plt.close()
            except Exception as e:
                print(f"global pareto plot failed {strat_dir}: {e}")
    # Exhaustive per strategy heatmaps already done in plot_pareto, but ensure also pareto_all_clients_f1 heatmaps
    for strat_dir in output_dir.glob("strategy_*"):
        f1_path = strat_dir / "pareto_all_clients_f1.csv"
        if f1_path.exists():
            try:
                df = pd.read_csv(f1_path)
                client_cols = [c for c in df.columns if c.startswith("f1_client_")]
                if not df.empty and client_cols:
                    plt.figure(figsize=(8, max(3, len(df)*0.4+2)))
                    mat = df[client_cols].values
                    plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
                    plt.colorbar(label="F1")
                    plt.xlabel("Client")
                    plt.ylabel("Pareto idx")
                    plt.title(f"Pareto x Client F1 - {strat_dir.name}")
                    plt.xticks(range(len(client_cols)), [c.replace("f1_client_","C") for c in client_cols])
                    plt.yticks(range(len(df)), df["pareto_idx"].astype(str))
                    plt.tight_layout()
                    plt.savefig(output_dir / "plots" / f"heatmap_{strat_dir.name}_f1.png", dpi=150)
                    plt.close()
            except Exception as e:
                print(f"pareto f1 heatmap failed: {e}")


def generate_all_plots(output_dir: Path, cfg):
    output_dir = Path(output_dir)
    _maybe_mkdir(output_dir / "plots")
    try:
        plot_class_distribution(output_dir)
    except Exception as e:
        print(f"plot_class_distribution failed: {e}")
    try:
        plot_divergence(output_dir)
    except Exception as e:
        print(f"plot_divergence failed: {e}")
    try:
        plot_hv(output_dir)
    except Exception as e:
        print(f"plot_hv failed: {e}")
    try:
        plot_pareto(output_dir)
    except Exception as e:
        print(f"plot_pareto failed: {e}")
    try:
        plot_savings(output_dir)
    except Exception as e:
        print(f"plot_savings failed: {e}")
    try:
        plot_pairing(output_dir)
    except Exception as e:
        print(f"plot_pairing failed: {e}")
    try:
        plot_fitness_approx_vs_exhaustive(output_dir)
    except Exception as e:
        print(f"plot_fitness failed: {e}")
    try:
        _plot_evolution_and_diversity(output_dir)
    except Exception as e:
        print(f"evolution/diversity plot failed: {e}")
    try:
        plot_heatmap_arch_client(output_dir)
    except Exception as e:
        print(f"heatmap_arch_client failed: {e}")
    print("All plots generated (best effort).")
