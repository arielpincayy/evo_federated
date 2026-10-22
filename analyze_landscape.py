#!/usr/bin/env python
"""Diagnóstico del paisaje de fitness con los datos de E3: ¿hay estructura que la evolución pueda explotar?"""
import glob
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

recs = []
for f in sorted(glob.glob("runs/e3/*/seed*/population.csv")):
    m, s = f.split("/")[-3], int(f.split("seed")[-1].split("/")[0])
    recs.append(pd.read_csv(f).assign(method=m, seed=s))
d = pd.concat(recs, ignore_index=True)
g = d.genome.apply(json.loads).apply(pd.Series)
d = pd.concat([d, g], axis=1)
d["ch_max"] = d.channels.apply(max); d["ch_sum"] = d.channels.apply(sum)
d["act_elu"] = (d.activation == "elu").astype(int); d["bn"] = d.use_bn.astype(int)
feat = ["n_blocks", "ch_max", "ch_sum", "kernel", "act_elu", "dropout", "bn"]

print("== 1. Paisaje (random search + clusters Δ: 150 arquitecturas al azar por semilla, acc. en todos los nodos)")
r = d[d.method == "cs_rand"]
print(f"acc media={r.mon_acc_mean.mean():.3f}  sd entre arquitecturas={r.mon_acc_mean.std():.3f}  min={r.mon_acc_mean.min():.3f}  max={r.mon_acc_mean.max():.3f}")
for eps in (0.02, 0.05):
    frac = r.groupby("seed").apply(lambda x: (x.mon_acc_mean >= x.mon_acc_mean.max() - eps).mean())
    print(f"fracción de arquitecturas aleatorias a ≤{eps} de la mejor (por semilla): {frac.mean():.3f}")
print(f"params_norm: mediana={r.n_params_norm.median():.3f}  P10={r.n_params_norm.quantile(.1):.3f}  P90={r.n_params_norm.quantile(.9):.3f}")
print(f"Spearman(acc, params)={spearmanr(r.mon_acc_mean, r.n_params_norm)[0]:.3f}")

print("\n== 2. ¿Cuánta varianza de acc explican los genes? (random forest, CV 5; sobre las evaluaciones de random search)")
X, y = r[feat].values, r.mon_acc_mean.values
pred = cross_val_predict(RandomForestRegressor(200, random_state=0, n_jobs=8), X, y, cv=5)
print(f"R² (CV) = {1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum():.3f}")
imp = pd.Series(RandomForestRegressor(200, random_state=0, n_jobs=8).fit(X, y).feature_importances_, index=feat).sort_values(ascending=False)
print(imp.round(3).to_string())

print("\n== 3. Señal vs ruido: la misma arquitectura reevaluada en distintas generaciones (NSGA-II reevalúa padres con reps distintos)")
for m in ("cs_nsga", "rand_nsga"):
    x = d[d.method == m]
    gr = x.groupby(["seed", "genome"]).agg(n=("mon_acc_mean", "size"), mean_all=("mon_acc_mean", "mean"), sd_all=("mon_acc_mean", "std"),
                                             sd_rep=("acc_rep", "std"))
    multi = gr[gr.n >= 3]
    sd_between = gr.groupby("seed").mean_all.std().mean()
    print(f"{m}: SD dentro de la misma arquitectura (acc todos los nodos)={multi.sd_all.mean():.3f}  (acc estimada por reps)={multi.sd_rep.mean():.3f}  "
          f"SD entre arquitecturas={sd_between:.3f}  n_arquitecturas_con≥3_evaluaciones={len(multi)}")

print("\n== 4. Mejora de la evolución sobre su propia población inicial y frente a random search (HV con todos los nodos por generación)")
c = pd.read_csv("runs/e3/curves.csv")
t = c.pivot_table(index="generation", columns="method", values="hv_monitor")
print(t.round(4).to_string())
print("HV(gen 7) - HV(gen 0):", (t.iloc[-1] - t.iloc[0]).round(4).to_dict())

print("\n== 5. Frentes finales: ¿qué tipo de arquitecturas lo forman?")
fin = []
for f in sorted(glob.glob("runs/e3/*/seed*/final_front.csv")):
    fin.append(pd.read_csv(f).assign(method=f.split("/")[-3]))
fin = pd.concat(fin)
fin["ch_sum"] = fin.genome.apply(lambda s: sum(json.loads(s)["channels"])); fin["n_blocks"] = fin.genome.apply(lambda s: json.loads(s)["n_blocks"])
print(fin.groupby("method")[["test_acc_mean", "n_params_norm", "n_blocks", "ch_sum"]].mean().round(3).to_string())
print("acc test máxima del frente, media por método:", fin.groupby(["method", "init_seed"]).test_acc_mean.max().groupby("method").mean().round(3).to_dict())
