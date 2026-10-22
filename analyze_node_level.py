#!/usr/bin/env python
"""Proporción de nodos que mejoran / quedan igual / empeoran entre dos modelos o conjuntos de modelos, a partir de runs/node_level/.

Un nodo 'mejoró' si su accuracy con B es mayor que con A; 'igual' si coincide; 'empeoró' si es menor ('no mejoró' = igual + empeoró).
Para conjuntos de modelos (población inicial, hall of fame) se compara la accuracy media del nodo sobre el conjunto; como control
también la mejor del conjunto para cada nodo. Resultados en runs/node_level/summary_*.csv
"""
import glob
import itertools

import numpy as np
import pandas as pd

OUT = "runs/node_level"
EPS = 1e-9


def split(a, b):
    d = np.asarray(b) - np.asarray(a)
    return {"mejoró": float((d > EPS).mean()), "igual": float((np.abs(d) <= EPS).mean()), "empeoró": float((d < -EPS).mean()),
            "cambio_medio_pp": float(100 * d.mean()), "mejoró_>2pp": float((d > 0.02).mean()), "empeoró_>2pp": float((d < -0.02).mean())}


def wk(a, b):  # peor cuartil de nodos según A (ojo: regresión a la media favorece a los peores)
    a = np.asarray(a); q = a <= np.quantile(a, 0.25)
    d = np.asarray(b) - a
    return float((d[q] > EPS).mean())


# ------------------------------------------------------------------ E3: población inicial, hall of fame, refinado
files = sorted(glob.glob(f"{OUT}/e3_*_seed*.csv"))
rows = []
if files:
    d = pd.concat(pd.read_csv(f) for f in files)
    for (m, s), g in d.groupby(["method", "seed"]):
        def per_node(role, how):
            x = g[g.roles.str.contains(role)]
            p = x.pivot_table(index="node", columns="genome", values="acc")
            return (p.mean(axis=1) if how == "media" else p.max(axis=1)).sort_index()
        sets = {r: {h: per_node(r, h) for h in ("media", "mejor")} for r in ("initial", "hof", "hof_refined") if g.roles.str.contains(r).any()}
        for how in ("media", "mejor"):
            if "initial" in sets and "hof" in sets:
                a_, b_ = sets["initial"][how], sets["hof"][how]
                rows.append({"comparación": "evolución: población inicial → hall of fame", "agregación": how, "método": m, "semilla": s, **split(a_, b_), "peor_cuartil_mejoró": wk(a_, b_)})
            if "hof" in sets and "hof_refined" in sets:
                a_, b_ = sets["hof"][how], sets["hof_refined"][how]
                rows.append({"comparación": "refinamiento: hall of fame → hall of fame refinado", "agregación": how, "método": m, "semilla": s, **split(a_, b_), "peor_cuartil_mejoró": wk(a_, b_)})
    ev = pd.DataFrame(rows)
    # entre métodos (hall of fame de A frente al de B, misma semilla)
    hofs = {}
    for (m, s), g in d.groupby(["method", "seed"]):
        x = g[g.roles.str.contains("hof")].pivot_table(index="node", columns="genome", values="acc")
        hofs[(m, s)] = (x.mean(axis=1).sort_index(), x.max(axis=1).sort_index())
    pairs = [("cs_nsga", "rand_nsga"), ("rand_nsga", "rand_rand"), ("cs_nsga", "cs_rand"), ("cs_rand", "rand_rand"), ("cs_nsga", "rand_rand"), ("rand_nsga", "cs_rand")]
    bm = []
    for A, B in pairs:
        for s in sorted({s for (_, s) in hofs}):
            if (A, s) in hofs and (B, s) in hofs:
                for k, how in enumerate(("media", "mejor")):
                    bm.append({"comparación": f"entre métodos: {A} → {B}", "agregación": how, "método": f"{A}->{B}", "semilla": s,
                               **split(hofs[(A, s)][k], hofs[(B, s)][k]), "peor_cuartil_mejoró": wk(hofs[(A, s)][k], hofs[(B, s)][k])})
    full = pd.concat([ev, pd.DataFrame(bm)], ignore_index=True)
    full.to_csv(f"{OUT}/summary_e3_per_seed.csv", index=False)
    agg = full.groupby(["comparación", "agregación"])[["mejoró", "igual", "empeoró", "cambio_medio_pp", "mejoró_>2pp", "empeoró_>2pp", "peor_cuartil_mejoró"]].agg(["mean", "std"]).round(3)
    agg.to_csv(f"{OUT}/summary_e3.csv")
    print("=== E3 (media entre semillas; std entre semillas)")
    print(agg.to_string())
    print("\n=== Evolución por método (agregación=media)")
    print(ev[ev.agregación == "media"].groupby(["comparación", "método"])[["mejoró", "igual", "empeoró", "cambio_medio_pp"]].agg(["mean", "std"]).round(3).to_string())

# ------------------------------------------------------------------ configuraciones de entrenamiento
tf = f"{OUT}/train_configs.csv"
try:
    t = pd.read_csv(tf); rows = []
    for al, g in t.groupby("alpha"):
        p = g.pivot_table(index="node", columns="model", values="acc")
        base = "adam1e-3_e2_R5_C10"
        for c in p.columns:
            if c != base:
                rows.append({"alpha": al, "comparación": f"{base} → {c}", **split(p[base], p[c]), "peor_cuartil_mejoró": wk(p[base], p[c])})
        rows.append({"alpha": al, "comparación": "adam1e-2_e2_R20_C10 → adam1e-2_e2_R20_full (10 nodos → todos)", **split(p["adam1e-2_e2_R20_C10"], p["adam1e-2_e2_R20_full"]), "peor_cuartil_mejoró": wk(p["adam1e-2_e2_R20_C10"], p["adam1e-2_e2_R20_full"])})
    r = pd.DataFrame(rows); r.to_csv(f"{OUT}/summary_train.csv", index=False)
    print("\n=== Configuraciones de entrenamiento (1 arquitectura, validación)"); print(r.round(3).to_string(index=False))
except FileNotFoundError:
    pass

# ------------------------------------------------------------------ ruido: C y muestreo de nodos
nf = sorted(glob.glob(f"{OUT}/noise_arch*.csv"))
if nf:
    n = pd.concat(pd.read_csv(f) for f in nf); rows = []
    for arch, g in n.groupby("arch"):
        p = g.pivot_table(index="node", columns="model", values="acc")
        for dr in (0, 1):
            for C in (10, 20, 40):
                rows.append({"arch": arch, "draw": dr, "comparación": f"muestreo, C={C}: nodos aleatorios → clusters", **split(p[f"random_C{C}_d{dr}"], p[f"cluster_C{C}_d{dr}"])})
            for fam in ("cluster", "random"):
                for C in (20, 40):
                    rows.append({"arch": arch, "draw": dr, "comparación": f"{fam}: C=10 → C={C}", **split(p[f"{fam}_C10_d{dr}"], p[f"{fam}_C{C}_d{dr}"])})
                for C in (10, 20, 40):
                    rows.append({"arch": arch, "draw": dr, "comparación": f"{fam}: C={C} → todos los nodos", **split(p[f"{fam}_C{C}_d{dr}"], p["full"])})
    r = pd.DataFrame(rows); r.to_csv(f"{OUT}/summary_noise_per_arch.csv", index=False)
    s = r.groupby("comparación")[["mejoró", "igual", "empeoró", "cambio_medio_pp"]].agg(["mean", "std"]).round(3)
    s.to_csv(f"{OUT}/summary_noise.csv"); print("\n=== Estudio de ruido (media sobre arquitecturas y sorteos; std entre filas)"); print(s.to_string())

# ------------------------------------------------------------------ hall of fame: ¿alguna arquitectura mejora a todos los nodos?
if files:
    rows, bb = [], []
    for (m, s), g in d.groupby(["method", "seed"]):
        P = g.pivot_table(index="node", columns="genome", values="acc").sort_index(); roles = g.groupby("genome").roles.first()
        ini = [x for x in P.columns if "initial" in roles[x]]; hof = [x for x in P.columns if "hof" in roles[x].split("|")]
        refs = {"media_inicial": P[ini].mean(axis=1), "mejor_modelo_inicial": P[P[ini].mean().idxmax()], "mejor_inicial_por_nodo": P[ini].max(axis=1)}
        for h in hof:
            r_ = {"method": m, "seed": s, "genome": h, "en_inicial": h in ini, "acc_media": float(P[h].mean())}
            for k, ref in refs.items():
                r_[f"{k}_prop_mejora"] = float((P[h] > ref + EPS).mean()); r_[f"{k}_todos_mejoran"] = bool((P[h] > ref + EPS).all()); r_[f"{k}_ningun_peor"] = bool((P[h] >= ref - EPS).all())
            rows.append(r_)
        bi, bh = P[ini].mean().idxmax(), P[hof].mean().idxmax()
        bb.append({"method": m, "seed": s, "mismo_modelo": bi == bh, "acc_inicial": float(P[bi].mean()), "acc_hof": float(P[bh].mean()), "mejoran": float((P[bh] > P[bi] + EPS).mean()),
                   "igual": float((abs(P[bh] - P[bi]) <= EPS).mean()), "empeoran": float((P[bh] < P[bi] - EPS).mean())})
    h = pd.DataFrame(rows); h.to_csv(f"{OUT}/hof_per_model.csv", index=False)
    b = pd.DataFrame(bb); b.to_csv(f"{OUT}/hof_best_vs_best.csv", index=False)
    print(f"\n=== Hall of fame: {len(h)} arquitecturas en 20 corridas ({int(h.en_inicial.sum())} ya estaban en la población inicial)")
    for k in ("media_inicial", "mejor_modelo_inicial", "mejor_inicial_por_nodo"):
        print(f"  vs {k}: mejoran a todos los nodos {int(h[k+'_todos_mejoran'].sum())}; ningún nodo peor {int(h[k+'_ningun_peor'].sum())}; "
              f"mediana de nodos que mejoran {h[k+'_prop_mejora'].median():.2f}; máx {h[k+'_prop_mejora'].max():.2f}; con >=90% {int((h[k+'_prop_mejora'] >= .9).sum())}")
    x = b[~b.mismo_modelo]
    print(f"\nMejor del hall of fame = mejor inicial en {int(b.mismo_modelo.sum())} de {len(b)} corridas; en las {len(x)} restantes mejoran de media {x.mejoran.mean():.2f} "
          f"de los nodos (mín {x.mejoran.min():.2f}, máx {x.mejoran.max():.2f}); la accuracy media sube en {int((x.acc_hof > x.acc_inicial + 1e-9).sum())}, "
          f"baja en {int((x.acc_hof < x.acc_inicial - 1e-9).sum())}")
