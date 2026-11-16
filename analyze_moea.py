#!/usr/bin/env python
"""Análisis del estudio MOEA (docs/MOEA_DESIGN.md): nsga3 vs rs (aleatoria) al mismo presupuesto, pareado por semilla.
Lee runs/moea/{brazo}/seed{s}/{final_front.csv, population.csv, cost.json}. Uso: python analyze_moea.py [--root runs/moea]"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon

from evofederated.evolution.cs_nas import _hv, _nd

ARMS = ("nsga3", "rs")
METRICS = ("hv_retrained", "auc_archive", "slope_hv_gen", "gain_hv_gen")


def hv2(acc, size):
    return _hv(np.stack([-np.asarray(acc), np.asarray(size)], 1))


def run_metrics(d):
    cost = json.load(open(d / "cost.json"))
    pop, ff = pd.read_csv(d / "population.csv"), pd.read_csv(d / "final_front.csv")
    # convergencia: hipervolumen del archivo acumulado (accuracy estimada, tamaño) frente al nº de evaluaciones
    gens = sorted(pop.generation.unique())
    curve = np.array([(int((pop.generation <= g).sum()), hv2(pop[pop.generation <= g].acc_rep, pop[pop.generation <= g].n_params_norm)) for g in gens])
    auc = float(np.trapz(curve[:, 1], curve[:, 0]) / (curve[-1, 0] - curve[0, 0]))
    late_gain = float(curve[-1, 1] - curve[min(np.searchsorted(curve[:, 0], 0.5 * curve[-1, 0]), len(curve) - 1), 1])  # ganancia del archivo entre el 50 % y el 100 % de las evaluaciones (saturación)
    # población: hipervolumen de los individuos nuevos por generación (g0 completa; después solo hijos, los padres se reevalúan)
    new = pop if cost.get("reeval_parents", True) is False else pop[(pop.generation == 0) | (pop.individual_idx >= cost["pop_size"])]  # v2: solo se evalúan hijos
    hvg = np.array([hv2(g.acc_rep, g.n_params_norm) for _, g in new.groupby("generation")])
    slope = float(np.polyfit(np.arange(len(hvg)), hvg, 1)[0])
    # final reentrenado
    hof = ff[ff.roles.str.split("|").map(lambda r: "hof" in r)]
    pts = np.stack([-hof.test_acc_mean.values, hof.n_params_norm.values], 1)
    front = hof.iloc[_nd(pts)]
    raw = pop.assign(raw=pop.obj_raw.map(json.loads)).groupby("genome").raw.apply(lambda r: np.mean(np.stack(r.values), axis=0))
    gap = float(np.mean([raw[g].max() - raw[g].min() for g in hof.genome if g in raw.index]))
    row = dict(k=cost["k"], pop_size=cost["pop_size"], n_gen=cost["n_generations"], search_evals=cost["search_evals"], search_local_trainings=cost["search_local_trainings"],
               elapsed_h=cost["elapsed_sec"] / 3600, n_distinct=cost.get("n_distinct_genomes", pop.genome.nunique()), cache_hits=cost.get("n_cache_hits", 0), hv_retrained=hv2(hof.test_acc_mean, hof.n_params_norm), auc_archive=auc, late_gain_archive=late_gain, hv_archive_final=float(curve[-1, 1]),
               slope_hv_gen=slope, gain_hv_gen=float(hvg[-5:].mean() - hvg[0]), hv_gen0=float(hvg[0]), hv_gen_last=float(hvg[-1]),
               front_size=len(front), size_range=float(front.n_params_norm.max() - front.n_params_norm.min()), best_acc=float(hof.test_acc_mean.max()), cluster_gap_est=gap,
               tau_est_test=float(kendalltau(hof.acc_rep_mean, hof.test_acc_mean)[0]))  # fidelidad práctica del estimador en el hall of fame
    return row, curve


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/moea")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--suffix", default="", help="sufijo de los CSV de salida (p. ej. _rep para la réplica)")
    ap.add_argument("--compare", help="raíz de otra versión (p. ej. runs/moea) para comparar el hipervolumen reentrenado por brazo (descriptivo)")
    a = ap.parse_args()
    rows, curves, missing = [], [], []
    for arm in ARMS:
        for s in a.seeds:
            d = Path(a.root) / arm / f"seed{s}"
            if not (d / "cost.json").exists():
                missing.append(f"{arm}/seed{s}")
                continue
            r, c = run_metrics(d)
            rows.append(dict(arm=arm, seed=s, **r))
            curves += [dict(arm=arm, seed=s, evals=int(e), hv=h) for e, h in c]
    if missing:
        print("sin terminar:", ", ".join(missing))
    if not rows:
        raise SystemExit
    d = pd.DataFrame(rows)
    d.to_csv(Path(a.root) / f"per_run{a.suffix}.csv", index=False)
    pd.DataFrame(curves).to_csv(Path(a.root) / f"archive_curves{a.suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(d.round(4).to_string(index=False))
    print("\nmedia por brazo:\n", d.groupby("arm").mean(numeric_only=True).round(4).T.to_string())
    rng, out = np.random.RandomState(0), []
    w = d[d.arm == "nsga3"].set_index("seed")[list(METRICS)].join(d[d.arm == "rs"].set_index("seed")[list(METRICS)], rsuffix="_rs", how="inner")
    for m in METRICS:
        diff = (w[m] - w[m + "_rs"]).values
        if len(diff) < 2:
            continue
        ci = np.percentile([rng.choice(diff, len(diff)).mean() for _ in range(5000)], [2.5, 97.5])
        p = wilcoxon(diff, alternative="greater").pvalue if np.any(diff != 0) else 1.0
        out.append(dict(metrica=m, n=len(diff), media=diff.mean(), ic_lo=ci[0], ic_hi=ci[1], p_unilateral=p, a_favor=int((diff > 0).sum()), por_semilla=" ".join(f"{x:+.4f}" for x in diff)))
    c = pd.DataFrame(out)
    c.to_csv(Path(a.root) / f"paired_contrasts{a.suffix}.csv", index=False)
    print("\ncontrastes pareados nsga3 - rs (Wilcoxon unilateral e IC bootstrap descriptivo):\n", c.round(4).to_string(index=False))
    if a.compare:  # comparación entre versiones a igual coste de entrenamiento (descriptiva; bilateral): se recalculan las métricas del otro directorio con las mismas funciones
        o = pd.DataFrame([dict(arm=arm, seed=s, **run_metrics(Path(a.compare) / arm / f"seed{s}")[0]) for arm in ARMS for s in a.seeds if (Path(a.compare) / arm / f"seed{s}" / "cost.json").exists()])
        for arm in ARMS:
            for m in ("hv_retrained", "tau_est_test"):
                x, y = d[d.arm == arm].set_index("seed")[m], o[o.arm == arm].set_index("seed")[m]
                diff = (x - y).dropna()
                if len(diff) < 2:
                    continue
                ci = np.percentile([rng.choice(diff.values, len(diff)).mean() for _ in range(5000)], [2.5, 97.5])
                p2 = wilcoxon(diff.values).pvalue if np.any(diff.values != 0) else 1.0
                print(f"\n{arm} {m}: {a.root} − {a.compare}: media {diff.mean():+.4f} IC95 [{ci[0]:+.4f}, {ci[1]:+.4f}], p bilateral {p2:.4f}, {int((diff > 0).sum())}/{len(diff)} semillas a favor (medias {x.mean():.4f} frente a {y.mean():.4f})")
    if len(c):
        ok = lambda m: float(c[c.metrica == m].p_unilateral.iloc[0]) < 0.05
        if ok("hv_retrained") and ok("auc_archive"):
            print("\nDECISIÓN (preregistrada): el MOEA optimiza mejor que la búsqueda aleatoria (primaria y convergencia p < 0.05).")
        elif ok("slope_hv_gen") or ok("gain_hv_gen"):
            print("\nDECISIÓN (preregistrada): mejora la población sin ventaja demostrada en el frente final (primaria/convergencia no significativas).")
        else:
            print("\nDECISIÓN (preregistrada): no se detecta ventaja del MOEA sobre la búsqueda aleatoria.")
