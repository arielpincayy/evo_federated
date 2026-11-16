#!/usr/bin/env python
"""Confirmatoria del artículo y ablación del objetivo por cluster (docs/MOEA_DESIGN.md, última sección).

H1 (modelos livianos): mejor accuracy en test entre los 12 modelos reentrenados con tamaño <= s (v2: 0.02, v4: 0.005; 0 si ninguno), nsga3 - rs.
H2 (eficiencia): hipervolumen del archivo (accuracy estimada, tamaño) de nsga3 con el 50 % de sus evaluaciones menos el de rs con el 100 %.
Holm sobre los 4 tests (H1, H2 x v2, v4), Wilcoxon unilateral pareado por semilla; se cumple con Holm-p < 0.05 y >= 7/10 semillas a favor.
H3 (ablación): nsga3 - nsga3g en H1 y en el hipervolumen reentrenado, Wilcoxon bilateral, Holm sobre 4 tests.
Uso: python analyze_confirm.py [--seeds 10..19] [--h3-seeds 0..19]   (con --seeds 0..9 reproduce las cifras exploratorias)"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analyze_moea import hv2

DESIGNS = {"v2": ("runs/moea2", 0.02), "v4": ("runs/moea4", 0.005)}


def best_at_size(acc, size, s):
    """Mejor accuracy entre los modelos con tamaño <= s (0 si ninguno)."""
    acc, size = np.asarray(acc, float), np.asarray(size, float)
    return float(acc[size <= s].max()) if (size <= s).any() else 0.0


def archive_hv(pop, frac=1.0):
    """Hipervolumen del archivo tras las primeras frac*n evaluaciones (population.csv está en orden de evaluación)."""
    p = pop.iloc[: int(np.ceil(frac * len(pop)))]
    return hv2(p.acc_rep, p.n_params_norm)


def holm(p):
    """Corrección de Holm (p ajustados, en el orden de entrada)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def run_row(root, arm, seed, s):
    d = Path(root) / arm / f"seed{seed}"
    ff, pop = pd.read_csv(d / "final_front.csv"), pd.read_csv(d / "population.csv")
    hof = ff[ff.roles.str.split("|").map(lambda r: "hof" in r)]
    return dict(h1=best_at_size(hof.test_acc_mean, hof.n_params_norm, s), hv_retrained=hv2(hof.test_acc_mean, hof.n_params_norm),
                hv_archive=archive_hv(pop), hv_archive_half=archive_hv(pop, 0.5), hv_archive_initial=hv2(pop[pop.generation == 0].acc_rep, pop[pop.generation == 0].n_params_norm), n_evals=len(pop))


def collect(arms_seeds):
    rows = []
    for design, (root, s) in DESIGNS.items():
        for arm, seeds in arms_seeds.items():
            for seed in seeds:
                if (Path(root) / arm / f"seed{seed}" / "cost.json").exists():
                    rows.append(dict(design=design, arm=arm, seed=seed, **run_row(root, arm, seed, s)))
    return pd.DataFrame(rows)


def paired(df, design, a, b, col_a, col_b=None):
    """Diferencias pareadas por semilla a[col_a] - b[col_b]."""
    col_b = col_b or col_a
    x = df[(df.design == design) & (df.arm == a)].set_index("seed")[col_a]
    y = df[(df.design == design) & (df.arm == b)].set_index("seed")[col_b]
    return (x - y.reindex(x.index)).dropna()


def test(d, alternative):
    p = wilcoxon(d, alternative=alternative).pvalue if (d != 0).any() else 1.0
    rng = np.random.RandomState(0)
    lo, hi = np.percentile([rng.choice(d.values, len(d)).mean() for _ in range(5000)], [2.5, 97.5])
    return dict(mean=d.mean(), ci_lo=lo, ci_hi=hi, n=len(d), n_pos=int((d > 0).sum()), p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    ap.add_argument("--h3-seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--out", default="runs/confirm")
    a = ap.parse_args()
    df = collect({"nsga3": sorted(set(a.seeds) | set(a.h3_seeds)), "rs": a.seeds, "nsga3g": a.h3_seeds})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{a.out}/per_run.csv", index=False)
    conf = []
    for design in DESIGNS:
        sub = df[df.seed.isin(a.seeds)]
        conf.append(dict(hyp="H1 livianos", design=design, **test(paired(sub, design, "nsga3", "rs", "h1"), "greater")))
        conf.append(dict(hyp="H2 eficiencia", design=design, **test(paired(sub, design, "nsga3", "rs", "hv_archive_half", "hv_archive"), "greater")))
    t1 = pd.DataFrame(conf)
    t1["p_holm"] = holm(t1.p)
    t1["cumple"] = (t1.p_holm < 0.05) & (t1.n_pos >= 7)
    abl = []
    for design in DESIGNS:
        sub = df[df.seed.isin(a.h3_seeds)]
        for m, name in (("h1", "H3 ablación: livianos"), ("hv_retrained", "H3 ablación: hv reentrenado")):
            abl.append(dict(hyp=name, design=design, **test(paired(sub, design, "nsga3", "nsga3g", m), "two-sided")))
    t3 = pd.DataFrame(abl)
    t3["p_holm"] = holm(t3.p)
    t3["cumple"] = t3.p_holm < 0.05
    pd.set_option("display.width", 200)
    out = pd.concat([t1, t3], ignore_index=True)
    out.to_csv(f"{a.out}/tests.csv", index=False)
    print(df.groupby(["design", "arm"])[["h1", "hv_retrained", "hv_archive", "hv_archive_half", "hv_archive_initial"]].mean().round(4).to_string())
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
