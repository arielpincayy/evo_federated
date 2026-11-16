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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registra la proyección 3d)
from pymoo.indicators.hv import HV

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


# ---------------------------------------------------------------- frentes en los 3 objetivos (accuracy cluster 1, accuracy cluster 2, tamaño)
def nondominated3(m1, m2, size):
    """Índices no dominados en los 3 objetivos (maximizar las accuracies de los dos clusters, minimizar el tamaño)."""
    return np.asarray(_nd(np.stack([-np.asarray(m1, float), -np.asarray(m2, float), np.asarray(size, float)], 1)))


def hv3(m1, m2, size):
    """Hipervolumen en los 3 objetivos (minimización de [-m1, -m2, tamaño]) con punto de referencia (0, 0, 1.05)."""
    F = np.stack([-np.asarray(m1, float), -np.asarray(m2, float), np.asarray(size, float)], 1)
    return float(HV(ref_point=np.array([0.0, 0.0, 1.05])).do(F)) if len(F) else 0.0


def cluster_acc(nodes, labels):
    """Accuracy media por nodo dentro de cada cluster, para cada modelo (genoma). -> DataFrame genoma x cluster."""
    return nodes.assign(cl=np.asarray(labels)[nodes.node.values]).groupby(["genome", "cl"]).acc.mean().unstack()


def search_points(root, arm, seed):
    """Objetivos estimados de TODAS las arquitecturas evaluadas en la búsqueda: (m1, m2, tamaño)."""
    p = pd.read_csv(Path(root) / arm / f"seed{seed}" / "population.csv")
    raw = np.stack(p.obj_raw.map(json.loads).values)
    return raw[:, 0], raw[:, 1], p.n_params_norm.values


def retrained_points(root, arm, seed):
    """Objetivos reales (accuracy en test por cluster, tamaño) de los 12 modelos reentrenados con los 100 nodos."""
    d = Path(root) / arm / f"seed{seed}"
    labels = json.load(open(d / "clusters.json"))["labels"]
    nodes = pd.read_csv(d / "final_nodes.csv")
    nodes = nodes[nodes.roles.str.split("|").map(lambda r: "hof" in r)]
    ca = cluster_acc(nodes, labels)
    f = pd.read_csv(d / "final_front.csv").drop_duplicates("genome").set_index("genome")
    return ca[0].values, ca[1].values, f.loc[ca.index, "n_params_norm"].values


def front2d(x, y, minimize_x):
    """Frente no dominado de una proyección 2D (y se maximiza), ordenado por x."""
    k = np.asarray(_nd(np.stack([x if minimize_x else -x, -y], 1)))
    o = np.argsort(x[k])
    return x[k][o], y[k][o]


def fig_front3d(tag, root, seed, source):
    """Frente de Pareto en 3D (arriba a la izquierda) y sus tres cortes 2D, con ambos brazos."""
    get = search_points if source == "busqueda" else retrained_points
    pts = {a: get(root, a, seed) for a in COL}
    fig = plt.figure(figsize=(W, 6.0))
    ax3 = fig.add_subplot(2, 2, 1, projection="3d")
    axs = [fig.add_subplot(2, 2, i) for i in (2, 3, 4)]
    proj = [(0, 1, False, "accuracy cluster 1", "accuracy cluster 2"), (2, 0, True, "tamaño normalizado", "accuracy cluster 1"), (2, 1, True, "tamaño normalizado", "accuracy cluster 2")]
    handles = []
    for a in COL:
        m = np.stack(pts[a], 1)
        nd = nondominated3(*pts[a])
        mask = np.zeros(len(m), bool)
        mask[nd] = True
        h = hv3(*[m[nd, i] for i in range(3)])
        handles.append(plt.Line2D([], [], color=COL[a], marker="o", ls="", ms=4, label=f"{LAB[a]}: {len(nd)} no dominados, HV 3D {h:.3f}"))
        ax3.scatter(m[mask, 0], m[mask, 1], np.log10(m[mask, 2]), color=COL[a], s=10, alpha=0.9, depthshade=False)
        for ax, (i, j, logx, xl, yl) in zip(axs, proj):
            if source == "busqueda":
                ax.scatter(m[~mask, i], m[~mask, j], s=3, color=COL[a], alpha=0.10, linewidths=0)
                ax.scatter(m[mask, i], m[mask, j], s=9, color=COL[a], alpha=0.85, linewidths=0)
            else:  # 12 modelos: rellenos los no dominados en 3D, huecos los dominados
                ax.scatter(m[~mask, i], m[~mask, j], s=16, facecolors="none", edgecolors=COL[a], linewidths=0.8)
                ax.scatter(m[mask, i], m[mask, j], s=16, color=COL[a], linewidths=0)
            fx, fy = front2d(m[:, i], m[:, j], minimize_x=logx)
            ax.plot(fx, fy, color=COL[a], lw=1.0, alpha=0.9)
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
    ax3.set_xlabel("acc. cluster 1", labelpad=-4, fontsize=7)
    ax3.set_ylabel("acc. cluster 2", labelpad=-4, fontsize=7)
    ax3.set_zlabel("log10 tamaño", labelpad=-4, fontsize=7)
    ax3.tick_params(labelsize=6, pad=-2)
    ax3.view_init(elev=22, azim=-58)
    ax3.set_title("frente en 3D (puntos no dominados)", fontsize=8)
    for ax, t in zip(axs, ("corte: cluster 1 frente a cluster 2", "corte: tamaño frente a cluster 1", "corte: tamaño frente a cluster 2")):
        ax.set_title(t, fontsize=8)
    what = "accuracy estimada en la búsqueda" if source == "busqueda" else "accuracy en test de los 12 modelos reentrenados"
    fig.suptitle(f"{tag}, semilla {seed}: frente en los 3 objetivos ({what})", fontsize=9, y=0.995)
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, fontsize=7)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    save(fig, f"frente3d_{source}_{tag}_s{seed}")
    return {a: (int(len(nondominated3(*pts[a]))), round(hv3(*[np.stack(pts[a], 1)[nondominated3(*pts[a]), i] for i in range(3)]), 4)) for a in COL}


def summary_3d():
    """Hipervolumen 3D del archivo de la búsqueda (estimado) por diseño: media de las semillas y semillas en las que gana nsga3."""
    for tag, root, _ in DESIGNS:
        h = {a: [] for a in COL}
        for s in SEEDS:
            for a in COL:
                m = np.stack(search_points(root, a, s), 1)
                nd = nondominated3(*search_points(root, a, s))
                h[a].append(hv3(*[m[nd, i] for i in range(3)]))
        d = np.array(h["nsga3"]) - np.array(h["rs"])
        print(f"{tag}: HV 3D del archivo (estimado) nsga3 {np.mean(h['nsga3']):.4f} / rs {np.mean(h['rs']):.4f}; diferencia media {d.mean():+.4f}, nsga3 gana en {(d > 0).sum()}/10 semillas")


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
    for tag, root, _ in DESIGNS:
        for source in ("busqueda", "reentrenado"):
            print(f"frente 3D {tag} semilla 0 ({source}): (no dominados, HV 3D)", fig_front3d(tag, root, 0, source))
    summary_3d()
    print("figuras en", OUT)


if __name__ == "__main__":
    main()
