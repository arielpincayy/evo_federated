#!/usr/bin/env python
"""Analiza runs/noise/arch*.csv: ruido por nodos de entrenamiento, fiabilidad del ranking y distorsión frente a participación completa."""
import glob
import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob("runs/noise/arch*.csv")))
n_arch = d.arch.nunique()
truth = d[d.arm == "full"].set_index("arch").acc_all
d = d[d.arm != "full"]
rows = []
for arm, g in d.groupby("arm"):
    piv = g.pivot(index="arch", columns="draw", values="acc_all")
    mean = piv.mean(axis=1)
    sd_within = piv.std(axis=1, ddof=1).mean()
    sd_between = mean.std()
    taus = [kendalltau(piv[i], piv[j])[0] for i, j in itertools.combinations(piv.columns, 2)]  # test-retest entre sorteos
    common = truth.index.intersection(mean.index)
    rows.append({"arm": arm, "C": g.C.iloc[0], "family": g.family.iloc[0], "sd_within": sd_within, "sd_between": sd_between,
                 "noise/signal": sd_within / sd_between, "retest_kendall": np.mean(taus),
                 "kendall_vs_full(1 draw)": np.mean([kendalltau(piv[c].loc[common], truth.loc[common])[0] for c in piv.columns]),
                 "kendall_vs_full(mean draws)": kendalltau(mean.loc[common], truth.loc[common])[0],
                 "bias_vs_full": (mean.loc[common] - truth.loc[common]).mean(),
                 "rmse_vs_full(1 draw)": np.sqrt(np.mean([((piv[c].loc[common] - truth.loc[common]) ** 2).mean() for c in piv.columns]))})
res = pd.DataFrame(rows).sort_values(["family", "C"])
res.to_csv("runs/noise/summary.csv", index=False)
print(f"arquitecturas: {n_arch}; sorteos por arquitectura y arm: {d.draw.nunique()}; SD de la accuracy completa entre arquitecturas: {truth.std():.3f}; media {truth.mean():.3f}")
print(res.round(3).to_string(index=False))
