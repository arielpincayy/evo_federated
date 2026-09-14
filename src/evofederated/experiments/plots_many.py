"""Graficas many-objective (no PCA visual, sino objetivo space)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import json


def plot_hypervolume(hv_csv: Path, out: Path):
    try:
        df = pd.read_csv(hv_csv)
        plt.figure(figsize=(7,4))
        plt.plot(df["generation"], df["hv"], marker="o")
        plt.xlabel("Generation")
        plt.ylabel("Hypervolume")
        plt.title("Hypervolume vs Generation")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
    except Exception as e:
        print(f"plot_hv failed {e}")

def plot_igd(igd_csv: Path, out: Path):
    try:
        df = pd.read_csv(igd_csv)
        if df.empty: return
        plt.figure(figsize=(7,4))
        plt.plot(df["generation"], df["igd"], marker="o", color="orange")
        plt.xlabel("Generation")
        plt.ylabel("IGD")
        plt.title("IGD vs Generation")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
    except Exception as e:
        print(f"plot igd {e}")

def plot_nondominated(hv_csv: Path, out: Path):
    try:
        df = pd.read_csv(hv_csv)
        plt.figure(figsize=(7,4))
        plt.plot(df["generation"], df["n_nd"], marker="s", color="green")
        plt.xlabel("Generation")
        plt.ylabel("# Nondominated")
        plt.title("Nondominated solutions vs Generation")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
    except Exception as e:
        print(f"plot nd {e}")

def plot_mean_worst(hv_csv: Path, out: Path):
    try:
        df = pd.read_csv(hv_csv)
        plt.figure(figsize=(7,4))
        plt.plot(df["generation"], df["mean_accuracy"], label="mean", marker="o")
        plt.plot(df["generation"], df["worst_accuracy"], label="worst", marker="x")
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.title("Mean / Worst-client Accuracy vs Generation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
    except Exception as e:
        print(f"plot mean worst {e}")

def plot_parallel_coordinates(pareto_csv: Path, out: Path):
    try:
        df = pd.read_csv(pareto_csv)
        if df.empty: return
        obj_cols = [c for c in df.columns if c.startswith("acc_client_")]
        if not obj_cols: return
        # limit to 30 solutions for readability
        sub = df.head(30)
        plt.figure(figsize=(max(8, len(obj_cols)*0.8), 5))
        for idx, row in sub.iterrows():
            vals = [row[c] for c in obj_cols]
            plt.plot(range(len(obj_cols)), vals, alpha=0.5, linewidth=1)
        plt.xticks(range(len(obj_cols)), [c.replace("acc_client_", "S") for c in obj_cols], rotation=45)
        plt.ylabel("Accuracy")
        plt.title("Parallel Coordinates Pareto Front (per-silo accuracy)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
    except Exception as e:
        print(f"parallel {e}")

def plot_heatmap(pareto_csv: Path, out: Path):
    try:
        df = pd.read_csv(pareto_csv)
        if df.empty: return
        obj_cols = [c for c in df.columns if c.startswith("acc_client_")]
        if not obj_cols: return
        sub = df.head(20)[obj_cols].values
        plt.figure(figsize=(max(8, len(obj_cols)*0.6), max(4, len(sub)*0.3)))
        im = plt.imshow(sub, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im, label="Accuracy")
        plt.yticks(range(len(sub)), [f"sol {i}" for i in range(len(sub))])
        plt.xticks(range(len(obj_cols)), [c.replace("acc_client_", "S") for c in obj_cols], rotation=45)
        plt.title("Objective Heatmap Pareto Solutions")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
    except Exception as e:
        print(f"heatmap {e}")

def plot_similarity_heatmap(S: np.ndarray, out: Path, title="Cosine Similarity"):
    try:
        plt.figure(figsize=(5,4))
        im = plt.imshow(S, vmin=-1, vmax=1, cmap="RdBu_r")
        plt.colorbar(im, label="sim")
        plt.title(title)
        plt.xlabel("Client")
        plt.ylabel("Client")
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
    except Exception as e:
        print(f"sim heatmap {e}")

def plot_pca_explained(pca_csv: Path, out: Path):
    try:
        df = pd.read_csv(pca_csv)
        plt.figure(figsize=(6,4))
        plt.plot(df["component"], df["cumulative"], marker="o")
        plt.bar(df["component"], df["explained_variance_ratio"], alpha=0.4)
        plt.xlabel("Component")
        plt.ylabel("Explained variance")
        plt.title("PCA Explained Variance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
    except Exception as e:
        print(f"pca expl {e}")

def plot_scatter_raw_vs_pca(S_raw: np.ndarray, S_pca: np.ndarray, out: Path):
    try:
        n = S_raw.shape[0]
        idx = np.triu_indices(n, k=1)
        a = S_raw[idx].ravel()
        b = S_pca[idx].ravel()
        plt.figure(figsize=(5,5))
        plt.scatter(a, b, alpha=0.7)
        lims = [min(a.min(), b.min())-0.05, max(a.max(), b.max())+0.05]
        plt.plot(lims, lims, "r--", alpha=0.5)
        plt.xlabel("S_raw")
        plt.ylabel("S_pca")
        plt.title("Raw vs PCA similarity scatter")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
    except Exception as e:
        print(f"scatter {e}")


def generate_all_plots_many(run_dir: Path):
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    hv = run_dir / "hypervolume.csv"
    igd = run_dir / "igd.csv"
    pareto = run_dir / "pareto_front.csv"
    pca_csv = run_dir / "similarity_pca" / "pca_explained_variance.csv"
    # core
    if hv.exists():
        plot_hypervolume(hv, plots_dir / "hypervolume_vs_generation.png")
        plot_nondominated(hv, plots_dir / "nondominated_vs_generation.png")
        plot_mean_worst(hv, plots_dir / "mean_worst_vs_generation.png")
    if igd.exists():
        plot_igd(igd, plots_dir / "igd_vs_generation.png")
    if pareto.exists():
        plot_parallel_coordinates(pareto, plots_dir / "parallel_coordinates_pareto.png")
        plot_heatmap(pareto, plots_dir / "heatmap_pareto.png")
        # accuracy final per silo box
        try:
            df = pd.read_csv(pareto)
            obj_cols = [c for c in df.columns if c.startswith("acc_client_")]
            if obj_cols:
                plt.figure(figsize=(8,4))
                vals = [df[c].values for c in obj_cols]
                plt.boxplot(vals, labels=[c.replace("acc_client_", "S") for c in obj_cols])
                plt.ylabel("Accuracy")
                plt.title("Final accuracy per silo (Pareto)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plots_dir / "accuracy_per_silo_box.png", dpi=150)
                plt.close()
        except Exception:
            pass
    # similarity
    for raw_path, pca_path in [(run_dir / "similarity_raw" / "similarity_matrix.npy", run_dir / "similarity_pca" / "similarity_matrix.npy")]:
        if raw_path.exists():
            S_raw = np.load(raw_path)
            plot_similarity_heatmap(S_raw, plots_dir / "similarity_raw_heatmap.png", "Raw cosine similarity")
        if pca_path.exists():
            S_pca = np.load(pca_path)
            plot_similarity_heatmap(S_pca, plots_dir / "similarity_pca_heatmap.png", "PCA cosine similarity")
        if raw_path.exists() and pca_path.exists():
            try:
                S_raw = np.load(raw_path)
                S_pca = np.load(pca_path)
                plot_scatter_raw_vs_pca(S_raw, S_pca, plots_dir / "raw_vs_pca_scatter.png")
            except Exception:
                pass
    if pca_csv.exists():
        plot_pca_explained(pca_csv, plots_dir / "pca_explained_variance.png")
