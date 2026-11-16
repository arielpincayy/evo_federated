#!/usr/bin/env python
"""Figuras del hipervolumen y de las fronteras de Pareto para el informe (report/informe.tex). Lee runs/ (no versionado) y escribe report/figures/*.pdf y *.png.

  python make_report_figures.py
Figuras: frentes_pareto (v2, v3, v4), hv_diferencias (forest plot), hv_convergencia, hv_eficiencia (Paso D) y frentes_semilla_v2 / frentes_semilla_v4 (anexo)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_moea import hv2
from evofederated.evolution.cs_nas import _nd

OUT = Path("report/figures")
COL = {"nsga3": "#0072B2", "rs": "#D55E00"}  # Okabe-Ito (apta para daltónicos)
LAB = {"nsga3": "nsga3 (MOEA)", "rs": "rs (aleatoria)"}
DESIGNS = [("v2", "runs/moea2", "v2: r=1, 1 988 eval."), ("v3", "runs/moea3", "v3: r=2, ≈1 000 eval."), ("v4", "runs/moea4", "v4: espacio ampliado, 504 eval.")]
SEEDS = range(10)
W = 6.3  # ancho de página en pulgadas


# ---------------------------------------------------------------- funciones puras (con tests)
def pareto_front(acc, size):
    """Índices de los puntos no dominados (maximizar accuracy, minimizar tamaño)."""
    return np.asarray(_nd(np.stack([-np.asarray(acc, float), np.asarray(size, float)], 1)))


def attainment(acc, size, grid):
    """f(s) = mejor accuracy entre los modelos con tamaño <= s (0 si ninguno alcanza ese tamaño): el frente de Pareto como función escalón."""
    acc, size = np.asarray(acc, float), np.asarray(size, float)
    return np.array([acc[size <= s].max() if (size <= s).any() else 0.0 for s in grid])


def common_grid(curves, n=200):
    """Interpola curvas (x, y), una por semilla, a una rejilla común dentro del rango compartido (sin extrapolar). -> (x, matriz semillas x n)."""
    lo, hi = max(c[0][0] for c in curves), min(c[0][-1] for c in curves)
    x = np.linspace(lo, hi, n)
    return x, np.stack([np.interp(x, cx, cy) for cx, cy in curves])


def boot_ci(x, n=5000, seed=0):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, float)
    return tuple(np.percentile([rng.choice(x, len(x)).mean() for _ in range(n)], [2.5, 97.5]))


# ---------------------------------------------------------------- datos
def hof_points(root, arm, seed):
    f = pd.read_csv(Path(root) / arm / f"seed{seed}" / "final_front.csv")
    f = f[f.roles.str.split("|").map(lambda r: "hof" in r)]
    return f.test_acc_mean.values, f.n_params_norm.values


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.dpi": 100, "savefig.bbox": "tight"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figuras
def fig_fronts():
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.6), sharey=False)
    for ax, (tag, root, title) in zip(axes, DESIGNS):
        pts = {a: [hof_points(root, a, s) for s in SEEDS] for a in COL}
        allsz = np.concatenate([p[1] for a in pts for p in pts[a]])
        grid = np.geomspace(allsz.min(), allsz.max(), 300)
        for a in COL:
            ax.scatter(np.concatenate([p[1] for p in pts[a]]), np.concatenate([p[0] for p in pts[a]]), s=6, color=COL[a], alpha=0.18, linewidths=0)
            att = np.stack([attainment(acc, sz, grid) for acc, sz in pts[a]])
            med, q1, q3 = np.median(att, 0), np.percentile(att, 25, 0), np.percentile(att, 75, 0)
            ok = med > 0
            ax.fill_between(grid[ok], q1[ok], q3[ok], color=COL[a], alpha=0.18, linewidth=0)
            ax.step(grid[ok], med[ok], where="post", color=COL[a], lw=1.8, label=LAB[a])
        ax.set_xscale("log")
        ax.set_title(title)
    fig.supxlabel("tamaño normalizado (parámetros / máximo del espacio, escala logarítmica)", fontsize=8, y=0.005)
    axes[0].set_ylabel("accuracy en test (100 nodos)")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "frentes_pareto")


def fig_forest():
    rows = [("v1 A (5 sem.)", "runs/moea/per_run.csv"), ("v1 B (5 sem.)", "runs/moea_b/per_run.csv"), ("v2 (10 sem.)", "runs/moea2/per_run_all.csv"),
            ("v3 (10 sem.)", "runs/moea3/per_run_all.csv"), ("v4 (10 sem.)", "runs/moea4/per_run_all.csv")]
    diffs = {}
    for name, f in rows:
        d = pd.read_csv(f).pivot(index="seed", columns="arm", values="hv_retrained")
        diffs[name] = (d["nsga3"] - d["rs"]).dropna()
    pooled = pd.concat([diffs[n] for n in ("v2 (10 sem.)", "v3 (10 sem.)", "v4 (10 sem.)")], axis=1).mean(axis=1)  # media por federación de los tres diseños
    diffs["media por federación\nv2–v4 (10 fed.)"] = pooled
    fig, ax = plt.subplots(figsize=(W, 2.9))
    rng = np.random.RandomState(1)
    for i, (name, d) in enumerate(diffs.items()):
        y = len(diffs) - 1 - i
        ax.scatter(d.values, y + rng.uniform(-0.18, 0.18, len(d)), s=14, color="0.55", alpha=0.7, linewidths=0, zorder=2)
        lo, hi = boot_ci(d.values)
        c = "#009E73" if i == len(diffs) - 1 else "black"
        ax.plot([lo, hi], [y, y], color=c, lw=2.2, zorder=3)
        ax.scatter([d.mean()], [y], marker="D", s=34, color=c, zorder=4)
        ax.text(0.092, y, f"{d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(diffs)))
    ax.set_yticklabels(list(diffs)[::-1])
    ax.set_xlim(-0.085, 0.17)
    ax.set_xlabel("diferencia en hipervolumen reentrenado (nsga3 − rs)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "hv_diferencias")
    return {k: round(v.mean(), 4) for k, v in diffs.items()}


def _new_hv_by_generation(root, arm, seed):
    d = Path(root) / arm / f"seed{seed}"
    pop = pd.read_csv(d / "population.csv")
    g = pop.groupby("generation")
    return np.array(g.size().index, float), np.array([hv2(x.acc_rep, x.n_params_norm) for _, x in g])


def fig_convergence():
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.2))
    for j, (tag, root, title) in enumerate(DESIGNS):
        arch = pd.read_csv(Path(root) / "archive_curves_all.csv")
        for a in COL:
            cur = [(g.evals.values.astype(float), g.hv.values) for _, g in arch[arch.arm == a].groupby("seed")]
            x, M = common_grid(cur)
            m, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
            axes[0, j].plot(x, m, color=COL[a], lw=1.6, label=LAB[a])
            axes[0, j].fill_between(x, m - se, m + se, color=COL[a], alpha=0.2, linewidth=0)
            gen = [_new_hv_by_generation(root, a, s) for s in SEEDS]
            x, M = common_grid(gen)
            m, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
            axes[1, j].plot(x, m, color=COL[a], lw=1.6)
            axes[1, j].fill_between(x, m - se, m + se, color=COL[a], alpha=0.2, linewidth=0)
        axes[0, j].set_title(title)
        axes[1, j].set_xlabel("generación")
        axes[0, j].set_xlabel("evaluaciones")
    axes[0, 0].set_ylabel("hipervolumen del archivo\n(accuracy estimada)")
    axes[1, 0].set_ylabel("hipervolumen de los\nindividuos nuevos")
    axes[0, 0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "hv_convergencia")


def fig_efficiency():
    d = pd.read_csv("runs/rerank_prefix/per_run.csv")
    d["b"] = d.budget.where(d.budget <= 1000, 1988)
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.7))
    out = {}
    for a in COL:
        g = d[d.arm == a].pivot(index="seed", columns="b", values="hv")
        bs = list(g.columns)
        m = g.mean().values
        ci = np.array([boot_ci(g[b].values) for b in bs])
        ax.plot(bs, m, "-o", color=COL[a], lw=1.6, ms=4, label=LAB[a])
        ax.fill_between(bs, ci[:, 0], ci[:, 1], color=COL[a], alpha=0.18, linewidth=0)
        for s in g.index:
            ax.plot(bs, g.loc[s].values, color=COL[a], lw=0.4, alpha=0.18)
        out[a] = [round(x, 4) for x in m]
    ax.set_xscale("log")
    ax.set_xticks([250, 500, 1000, 1988])
    ax.set_xticklabels(["250", "500", "1 000", "1 988"])
    ax.minorticks_off()
    ax.set_xlabel("evaluaciones usadas para elegir los 12 candidatos (B)")
    ax.set_ylabel("hipervolumen reentrenado")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, "hv_eficiencia")
    return out


def fig_seed_fronts(tag, root):
    fig, axes = plt.subplots(2, 5, figsize=(W, 3.9), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), SEEDS):
        hv = {}
        for a in COL:
            acc, sz = hof_points(root, a, s)
            ax.scatter(sz, acc, s=9, color=COL[a], alpha=0.55, linewidths=0, zorder=2)
            k = pareto_front(acc, sz)
            k = k[np.argsort(sz[k])]
            xs, ys = np.append(sz[k], 1e9), np.append(acc[k], acc[k][-1])
            ax.step(xs, ys, where="post", color=COL[a], lw=1.3, zorder=3)
            hv[a] = hv2(acc, sz)
        ax.set_xscale("log")
        ax.set_title(f"semilla {s}\nHV {hv['nsga3']:.3f} / {hv['rs']:.3f}", fontsize=7)
    allsz = np.concatenate([hof_points(root, a, s)[1] for a in COL for s in SEEDS])
    axes[0, 0].set_xlim(allsz.min() * 0.8, allsz.max() * 1.25)
    for ax in axes[1]:
        ax.set_xlabel("tamaño normalizado", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy en test", fontsize=7)
    fig.legend([plt.Line2D([], [], color=COL[a], lw=1.5) for a in COL], [LAB[a] for a in COL], loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, f"frentes_semilla_{tag}")


def main():
    style()
    fig_fronts()
    print("diferencias medias (nsga3 − rs):", fig_forest())
    fig_convergence()
    print("curva de eficiencia (media del hipervolumen reentrenado, B = 250, 500, 1000, 1988):", fig_efficiency())
    fig_seed_fronts("v2", "runs/moea2")
    fig_seed_fronts("v4", "runs/moea4")
    for tag, root, _ in DESIGNS:
        print(tag, "hipervolumen reentrenado medio:", {a: round(float(np.mean([hv2(*hof_points(root, a, s)) for s in SEEDS])), 4) for a in COL})
    print("figuras en", OUT)


if __name__ == "__main__":
    main()
