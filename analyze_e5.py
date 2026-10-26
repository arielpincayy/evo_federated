#!/usr/bin/env python
"""E5: métricas preregistradas (docs/E5_DESIGN.md) por brazo y semilla, y contrastes pareados cuando hay varias semillas.

Por corrida (runs/e5/<brazo>/seed<s>/final_nodes.csv): población inicial vs hall of fame por nodo (media y mejor del conjunto); del mejor
modelo del hall of fame por media ponderada estimada (razón 'max_media', elegido sin mirar test): accuracy media, del peor cluster
(clusters de Δ de la semilla, comunes a todos los brazos) y del peor decil de nodos; y nº de parámetros del hall of fame.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="runs/e5")
a = ap.parse_args()
EPS = 1e-9
ARMS = ["n3_cs", "n3_rand", "n3_cs_lownoise", "rs_cs", "n2_cs", "n2_rand"]
rows = []
for d in sorted(glob.glob(f"{a.root}/*/seed*")):
    arm, seed = d.split("/")[-2], int(d.split("seed")[-1])
    if not os.path.exists(f"{d}/cost.json"):
        continue
    n = pd.read_csv(f"{d}/final_nodes.csv"); f = pd.read_csv(f"{d}/final_front.csv"); cost = json.load(open(f"{d}/cost.json"))
    ref = f"{a.root}/n3_cs/seed{seed}/clusters.json"
    lab = np.array(json.load(open(ref if os.path.exists(ref) else f"{d}/clusters.json"))["labels"]) if os.path.exists(ref) or os.path.exists(f"{d}/clusters.json") else None
    P = n.pivot_table(index="node", columns="genome", values="acc").sort_index()
    roles = f.set_index("genome").roles
    ini = [g for g in P.columns if "initial" in roles[g]]; hof = [g for g in P.columns if "hof" in roles[g].split("|")]
    r = {"arm": arm, "seed": seed, "n_ini": len(ini), "n_hof": len(hof)}
    for how, fn in (("media", lambda x: x.mean(axis=1)), ("mejor", lambda x: x.max(axis=1))):
        A, B = fn(P[ini]), fn(P[hof]); dlt = B - A
        r.update({f"{how}_mejoró": float((dlt > EPS).mean()), f"{how}_igual": float((dlt.abs() <= EPS).mean()), f"{how}_empeoró": float((dlt < -EPS).mean()), f"{how}_cambio_pp": float(100 * dlt.mean())})
    best = f[f.reason == "max_media"].genome.iloc[0]; acc = P[best]
    r.update({"best_acc_media": float(acc.mean()), "best_peor_decil": float(np.quantile(acc, 0.1)), "best_peor_nodo": float(acc.min()),
              "best_params_norm": float(f[f.genome == best].n_params_norm.iloc[0]), "hof_params_norm_medio": float(f[f.roles.str.contains("hof")].n_params_norm.mean()),
              "ini_acc_media_mejor": float(P[ini].mean().max()), "hof_acc_media_mejor": float(P[hof].mean().max()), "horas": cost["elapsed_sec"] / 3600})
    if lab is not None:
        r["best_peor_cluster"] = float(min(acc.values[lab == c].mean() for c in np.unique(lab)))
        r["ini_mejor_peor_cluster"] = float(max(min(P[g].values[lab == c].mean() for c in np.unique(lab)) for g in ini))
    gm = pd.read_csv(f"{d}/generation_metrics.csv"); r["front_size_final"] = int(gm.front_size.iloc[-1])
    rows.append(r)
R = pd.DataFrame(rows)
if R.empty:
    raise SystemExit("aún no hay corridas terminadas")
R.to_csv(f"{a.root}/per_run.csv", index=False)
pd.set_option("display.width", 220, "display.max_columns", 40)
cols = ["media_mejoró", "media_igual", "media_empeoró", "media_cambio_pp", "mejor_mejoró", "mejor_igual", "mejor_empeoró", "best_acc_media", "best_peor_cluster", "best_peor_decil", "best_params_norm", "hof_params_norm_medio", "front_size_final"]
S = R.groupby("arm")[[c for c in cols if c in R]].mean().reindex([x for x in ARMS if x in set(R.arm)]).round(3)
S.insert(0, "n_semillas", R.groupby("arm").seed.nunique())
print(S.T.to_string()); S.to_csv(f"{a.root}/summary_by_arm.csv")
if R.seed.nunique() >= 2:
    rng = np.random.RandomState(0); out = []
    for x, y in (("n3_cs", "n3_rand"), ("n3_cs", "rs_cs"), ("n3_cs", "n2_cs"), ("n3_cs", "n2_rand"), ("n3_cs_lownoise", "n3_cs")):
        for m in ("media_mejoró", "best_acc_media", "best_peor_cluster", "best_peor_decil", "best_params_norm"):
            pv = R.pivot(index="seed", columns="arm", values=m)
            if x in pv and y in pv:
                d_ = (pv[x] - pv[y]).dropna().values
                if len(d_):
                    b = [rng.choice(d_, len(d_)).mean() for _ in range(5000)]
                    out.append({"contraste": f"{x} - {y}", "métrica": m, "diferencia": d_.mean(), "ic_bajo": np.percentile(b, 2.5), "ic_alto": np.percentile(b, 97.5), "a_favor": int((d_ > 0).sum()), "n": len(d_)})
    C = pd.DataFrame(out); C.to_csv(f"{a.root}/paired_contrasts.csv", index=False); print("\n", C.round(4).to_string(index=False))
