#!/usr/bin/env python
"""Genera figuras y tablas LaTeX del informe a partir de runs/ (no escribe números a mano). Ejecutar desde la raíz del repo."""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "manuscript/report_assets"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C = {"cs_nsga": "#0072B2", "rand_nsga": "#D55E00", "cs_rand": "#009E73", "rand_rand": "#CC79A7", "cluster": "#0072B2", "random": "#D55E00"}
LAB = {"cluster_delta": "Clusters por $\\Delta$", "cluster_label_oracle": "Clusters por etiquetas (oráculo)", "cluster_random": "Clusters aleatorios",
       "random_nodes": "Nodos aleatorios", "cluster_delta_noholdout": "Clusters por $\\Delta$, sin retener", "random_nodes_noholdout": "Nodos aleatorios, sin retener"}
ORDER = list(LAB)
MET = {"cs_nsga": "CS-NSGA (clusters $\\Delta$ + NSGA-II)", "rand_nsga": "Nodos aleatorios + NSGA-II", "cs_rand": "Clusters $\\Delta$ + búsqueda aleatoria", "rand_rand": "Nodos aleatorios + búsqueda aleatoria"}


def tex_table(df, name, caption, label, colfmt):
    body = df.to_latex(index=False, escape=False, column_format=colfmt, float_format=lambda v: f"{v:.3f}")
    body = body.replace("\\begin{tabular}", "\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}", 1).replace("\\end{tabular}", "\\end{tabular}}", 1)
    open(f"{OUT}/{name}.tex", "w").write(f"\\begin{{table}}[H]\\centering\\small\\caption{{{caption}}}\\label{{{label}}}\n{body}\\end{{table}}\n")


# --- 1. diagnóstico de entrenamiento -------------------------------------------------------------------
t = pd.read_csv("runs/training_diagnostic.csv")
t["cfg"] = t.name.str.replace("fullpart", "todos los nodos").str.replace("_nobn", " sin BN") + " · " + t.epochs.astype(str) + " ép. · " + t.rounds.astype(str) + " r."
t = t.pivot_table(index="cfg", columns="alpha", values="acc_mean").sort_values(0.5)
fig, ax = plt.subplots(figsize=(6.4, 3.6)); y = np.arange(len(t))
ax.barh(y - 0.2, t[0.5], 0.4, color="#0072B2", label="$\\alpha=0.5$"); ax.barh(y + 0.2, t[0.1], 0.4, color="#D55E00", label="$\\alpha=0.1$")
ax.set_yticks(y); ax.set_yticklabels(t.index, fontsize=7); ax.set_xlabel("Accuracy media en todos los nodos"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_training.pdf"); plt.close()

# --- 2. E2: estimador ----------------------------------------------------------------------------------------
sets = [("Fashion-MNIST, $\\alpha=0.1$", "runs/estimator_fmnist_lr/a0.1/per_seed_summary.csv"),
        ("Fashion-MNIST, $\\alpha=0.5$", "runs/estimator_fmnist_lr/a0.5/per_seed_summary.csv"),
        ("CIFAR-10, $\\alpha=0.5$", "runs/estimator_e2/per_seed_summary.csv")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
rows = []
for ax, (title, f) in zip(axs, sets):
    d = pd.read_csv(f); g = d.groupby("arm")
    m, s = g.rmse.mean().reindex(ORDER), g.rmse.std().reindex(ORDER)
    ax.barh(range(len(ORDER)), m, xerr=s, color=["#0072B2", "#999999", "#CC79A7", "#D55E00", "#56B4E9", "#E69F00"], capsize=2)
    ax.set_title(title, fontsize=8); ax.set_xlabel("RMSE del estimador"); ax.invert_yaxis()
    for arm in ORDER:
        x = d[d.arm == arm]
        rows.append({"Régimen": title, "Variante": LAB[arm], "Sesgo": x.bias.mean(), "RMSE": x.rmse.mean(), "RMSE (DE)": x.rmse.std(), "$\\tau$": x.kendall.mean(), "Semillas": x.seed.nunique()})
axs[0].set_yticks(range(len(ORDER))); axs[0].set_yticklabels([LAB[a].replace("$\\Delta$", "Δ") for a in ORDER], fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_e2.pdf"); plt.close()
e2 = pd.DataFrame(rows)
tex_table(e2, "tab_e2", "Estimador de fitness frente a la accuracy del mismo modelo en todos los nodos (media sobre semillas; DE = desviación entre semillas; $\\tau$: Kendall dentro de cada sorteo).", "tab:e2", "llrrrrr")

# --- 3. E3 --------------------------------------------------------------------------------------------------------
p = pd.read_csv("runs/e3/per_run.csv"); cv = pd.read_csv("runs/e3/curves.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.0))
for i, m in enumerate(MET):
    x = p[p.method == m].sort_values("seed")
    axs[0].plot(x.seed + (i - 1) * 0.12, x.hv_test, "o", color=C[m], label=MET[m])
    axs[0].hlines(x.hv_test.mean(), -0.4, 4.4, color=C[m], lw=1, alpha=.6)
    k = cv[cv.method == m].groupby("generation").hv_monitor
    axs[1].plot(k.mean().index, k.mean(), color=C[m]); axs[1].fill_between(k.mean().index, k.mean() - k.std(), k.mean() + k.std(), color=C[m], alpha=.15)
axs[0].set_xlabel("Semilla"); axs[0].set_ylabel("HV del frente final en test")
axs[1].set_xlabel("Generación"); axs[1].set_ylabel("HV (acc. de todos los nodos)")
# diferencia pareada por generación frente a la búsqueda aleatoria con los mismos clusters (IC bootstrap sobre semillas)
pv = cv.pivot_table(index=["seed", "generation"], columns="method", values="hv_monitor").reset_index()
rng = np.random.RandomState(0)
for m, base in (("cs_nsga", "cs_rand"), ("rand_nsga", "rand_rand")):
    mu, lo, hi = [], [], []
    for g in sorted(pv.generation.unique()):
        d = (pv[pv.generation == g][m] - pv[pv.generation == g][base]).values
        bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
        mu.append(d.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
    gens = sorted(pv.generation.unique())
    axs[2].plot(gens, mu, color=C[m]); axs[2].fill_between(gens, lo, hi, color=C[m], alpha=.15)
axs[2].axhline(0, color="k", lw=.8); axs[2].set_xlabel("Generación"); axs[2].set_ylabel("Diferencia pareada de HV"); axs[2].set_title("NSGA-II $-$ búsq. aleatoria", fontsize=7)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7)
plt.tight_layout(rect=(0, 0.12, 1, 1)); plt.savefig(f"{OUT}/fig_e3.pdf"); plt.close()
con = pd.read_csv("runs/e3/paired_contrasts.csv")
rows = [{"Método": MET[m], "HV test": f"{g.hv_test.mean():.3f} $\\pm$ {g.hv_test.std():.3f}", "HV todos los nodos": f"{g.hv_monitor_final.mean():.3f}",
         "RMSE est.": g.est_rmse.mean(), "$\\tau$ est.": g.est_kendall.mean(), "Entren. locales": int(g.search_trainings.mean())} for m, g in p.groupby("method")]
tex_table(pd.DataFrame(rows), "tab_e3", "E3: frente final reevaluado en test (media $\\pm$ DE, 5 semillas) y calidad del estimador durante la búsqueda.", "tab:e3", "lllrrr")
NM = {"cs_nsga": "CS-NSGA", "rand_nsga": "NSGA-II nodos aleat.", "cs_rand": "búsq. aleat. clusters", "rand_rand": "búsq. aleat. nodos aleat."}
ac = cv.pivot_table(index=["seed", "generation"], columns="method", values="hv_monitor").groupby("seed").mean()
rngb = np.random.RandomState(0)


def bci(d):
    b = [rngb.choice(d, len(d)).mean() for _ in range(5000)]
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), int((d > 0).sum()), len(d)


rows = []
for x, y in (("cs_nsga", "rand_nsga"), ("rand_nsga", "rand_rand"), ("cs_nsga", "cs_rand"), ("cs_rand", "rand_rand"), ("rand_nsga", "cs_rand")):
    pt = p.pivot(index="seed", columns="method", values="hv_test"); pm = p.pivot(index="seed", columns="method", values="hv_monitor_final")
    for lab, d in (("HV test", (pt[x] - pt[y]).values), ("HV todos (final)", (pm[x] - pm[y]).values), ("Área bajo la curva", (ac[x] - ac[y]).values)):
        m_, lo, hi, w, n_ = bci(d)
        rows.append({"Contraste": f"{NM[x]} $-$ {NM[y]}", "Métrica": lab, "Diferencia": m_, "IC 95 \\%": f"[{lo:.3f}, {hi:.3f}]", "A favor": f"{w}/{n_}"})
tex_table(pd.DataFrame(rows), "tab_e3c", "E3: diferencias pareadas por semilla (IC bootstrap sobre 5 semillas; descriptivo).", "tab:e3c", "llrrr")

# --- 4. ruido ------------------------------------------------------------------------------------------------------
n = pd.read_csv("runs/noise/summary.csv")
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
for fam in ("cluster", "random"):
    x = n[n.family == fam].sort_values("C")
    axs[0].plot(x.C, x.sd_within, "o-", color=C[fam], label="Clusters $\\Delta$" if fam == "cluster" else "Nodos aleatorios")
    axs[1].plot(x.C, x["kendall_vs_full(1 draw)"], "o-", color=C[fam])
axs[0].axhline(n.sd_between.mean(), ls=":", color="gray"); axs[0].text(10, n.sd_between.mean() + .004, "SD entre arquitecturas", fontsize=7, color="gray")
axs[0].set_xlabel("Nodos de entrenamiento $C$"); axs[0].set_ylabel("SD de la misma arquitectura"); axs[0].legend(frameon=False)
axs[1].set_xlabel("Nodos de entrenamiento $C$"); axs[1].set_ylabel("$\\tau$ frente a participación completa"); axs[1].set_ylim(0.7, 0.95)
for a_ in axs: a_.set_xticks([10, 20, 40])
plt.tight_layout(); plt.savefig(f"{OUT}/fig_noise.pdf"); plt.close()
nt = n.assign(Variante=n.family.map({"cluster": "Clusters $\\Delta$", "random": "Nodos aleatorios"}) + ", $C=" + n.C.astype(str) + "$")
nt = nt[["Variante", "sd_within", "noise/signal", "retest_kendall", "kendall_vs_full(1 draw)", "bias_vs_full", "rmse_vs_full(1 draw)"]]
nt.columns = ["Variante", "SD misma arq.", "Ruido/señal", "$\\tau$ test-retest", "$\\tau$ vs completa", "Sesgo vs completa", "RMSE vs completa"]
tex_table(nt, "tab_noise", "Ruido por nodos de entrenamiento (20 arquitecturas $\\times$ 4 sorteos, Fashion-MNIST, $\\alpha=0.1$).", "tab:noise", "lrrrrrr")

# --- 5. E4 ---------------------------------------------------------------------------------------------------------
r = pd.read_csv("runs/e3/refine_per_run.csv"); rc = pd.read_csv("runs/e3/refine_contrasts.csv")
rows = []
for m, g in r.groupby("method"):
    d = rc[(rc.contraste == f"H1 refinado − original ({m})") & (rc["métrica"] == "hv")].iloc[0]
    rows.append({"Método": MET[m], "HV original": g.hv_original.mean(), "HV refinado": g.hv_refined.mean(), "Cambio pareado": d.media,
                 "IC 95 \\%": f"[{d.ic_bajo:.3f}, {d.ic_alto:.3f}]", "Mejoran": f"{int(d.gana)}/{int(d.n)}"})
tex_table(pd.DataFrame(rows), "tab_e4", "E4: HV del frente final en test antes y después del refinamiento multifidelidad ($C=40$ sobre $\\leq 12$ candidatos).", "tab:e4", "lrrrrr")


# --- 6. cohesión de clusters ---------------------------------------------------------------------------------
cm = pd.read_csv("runs/clusters/cluster_metrics.csv")
NAME = {"kmeans_delta_C10": "$k$-means sobre $\\Delta$ (usada)", "label_oracle_C10": "Oráculo con etiquetas", "average_delta_C10": "Enlace promedio sobre $\\Delta$ (tamaño libre)", "random_C10": "Aleatoria"}
rows = []
for al in (0.1, 0.5):
    for k, lab in NAME.items():
        x = cm[(cm.alpha == al) & (cm.scheme == k)]
        rows.append({"$\\alpha$": al, "Agrupación": lab, "Tamaños mín/med/máx": f"{x.size_min.mean():.1f} / {x.size_median.mean():.1f} / {x.size_max.mean():.1f}",
                     "$\\eta^2$": x.eta2.mean(), "Dist. etiquetas dentro / entre": f"{x.within_hellinger.mean():.2f} / {x.between_hellinger.mean():.2f}",
                     "Pureza": x.purity_dominant_class.mean(), "Var. relativa del estimador": x.var_ratio_vs_random.mean()})
tex_table(pd.DataFrame(rows), "tab_clusters", "Cohesión de las agrupaciones con $C=10$ (Fashion-MNIST, $N=100$, media de 5 semillas). Dist. etiquetas: distancia de Hellinger entre distribuciones de clases; Var. relativa: varianza de un proxy del estimador (distribución de etiquetas) frente a $C$ nodos al azar.", "tab:clusters", "rlcrcrr")


# --- 7. proporción de nodos que mejoran ---------------------------------------------------------------------
NL = "runs/node_level"
e3n = pd.read_csv(f"{NL}/summary_e3_per_seed.csv"); trn = pd.read_csv(f"{NL}/summary_train.csv"); nz = pd.read_csv(f"{NL}/summary_noise_per_arch.csv")
MN = {"cs_nsga": "CS-NSGA", "rand_nsga": "NSGA-II nodos aleat.", "cs_rand": "búsq. aleat. clusters", "rand_rand": "búsq. aleat. nodos aleat."}
bars = {"A": [], "B": [], "C": []}
for m in ("cs_nsga", "rand_nsga", "cs_rand", "rand_rand"):
    for how in ("media", "mejor"):
        x = e3n[(e3n["comparación"].str.startswith("evolución")) & (e3n.método == m) & (e3n.agregación == how)]
        bars["A"].append((f"{MN[m]} ({how})", x))
for A_, B_ in (("cs_nsga", "rand_nsga"), ("rand_nsga", "rand_rand"), ("cs_nsga", "cs_rand"), ("cs_rand", "rand_rand"), ("cs_nsga", "rand_rand"), ("rand_nsga", "cs_rand")):
    x = e3n[(e3n["comparación"] == f"entre métodos: {A_} → {B_}") & (e3n.agregación == "mejor")]
    bars["B"].append((f"{MN[A_]} → {MN[B_]}", x))
for al in (0.1, 0.5):
    bars["C"].append((f"Entrenamiento $\\alpha={al}$: Adam $10^{{-3}}$/5 r. → Adam $10^{{-2}}$/5 ép./20 r.", trn[(trn.alpha == al) & (trn["comparación"] == "adam1e-3_e2_R5_C10 → adam1e-2_e5_R20_C10")]))
    bars["C"].append((f"Entrenamiento $\\alpha={al}$: 10 nodos → todos", trn[(trn.alpha == al) & trn["comparación"].str.contains("full")][lambda t: t["comparación"].str.startswith("adam1e-2_e2_R20_C10")]))
for lab, key in (("Muestreo $C=10$: nodos aleat. → clusters", "muestreo, C=10: nodos aleatorios → clusters"), ("Nodos aleat.: $C=10\\to20$", "random: C=10 → C=20"), ("Nodos aleat.: $C=10\\to40$", "random: C=10 → C=40"), ("Nodos aleat.: $C=10\\to$ todos", "random: C=10 → todos los nodos")):
    bars["C"].append((lab, nz[nz["comparación"] == key]))
fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.0), gridspec_kw={"height_ratios": [len(bars["A"]), len(bars["B"]), len(bars["C"])]})
ttl = {"A": "Evolución: población inicial $\\to$ hall of fame (por nodo)", "B": "Entre métodos: hall of fame A $\\to$ B (mejor modelo del conjunto por nodo)", "C": "Entrenamiento y número de nodos (un modelo por nodo)"}
for ax, k in zip(axs, "ABC"):
    for i, (lab, x) in enumerate(bars[k]):
        v = [x["mejoró"].mean(), x["igual"].mean(), x["empeoró"].mean()]
        ax.barh(i, v[0], color="#0072B2"); ax.barh(i, v[1], left=v[0], color="#BBBBBB"); ax.barh(i, v[2], left=v[0] + v[1], color="#D55E00")
        ax.text(0.01, i, f"{100*v[0]:.0f}%", va="center", color="white", fontsize=7); ax.text(0.99, i, f"{100*v[2]:.0f}%", va="center", ha="right", color="white", fontsize=7)
    ax.set_yticks(range(len(bars[k]))); ax.set_yticklabels([l for l, _ in bars[k]], fontsize=6.5); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_title(ttl[k], fontsize=8)
axs[-1].set_xlabel("Proporción de nodos (azul: mejoró; gris: igual; naranja: empeoró)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_nodes.pdf"); plt.close()

def ms(x, c): return f"{100*x[c].mean():.0f}" if len(x) < 2 else f"{100*x[c].mean():.0f} $\\pm$ {100*x[c].std():.0f}"
rows = []
for k in "AB":
    for lab, x in bars[k]:
        rows.append({"Comparación": lab, "Mejoró \\%": ms(x, "mejoró"), "Igual \\%": ms(x, "igual"), "Empeoró \\%": ms(x, "empeoró"), "Cambio medio (pp)": f"{x.cambio_medio_pp.mean():+.1f}", "n": len(x)})
tex_table(pd.DataFrame(rows), "tab_nodes_e3", "Proporción de nodos que mejoran, quedan igual o empeoran en E3 (media $\\pm$ DE entre las 5 semillas; accuracy por nodo en prueba con modelos entrenados con los 100 nodos). ``media'' y ``mejor'': accuracy media o máxima del nodo sobre el conjunto de modelos.", "tab:nodese3", "lrrrrr")
rows = []
for lab, x in bars["C"]:
    rows.append({"Comparación": lab, "Mejoró \\%": ms(x, "mejoró"), "Igual \\%": ms(x, "igual"), "Empeoró \\%": ms(x, "empeoró"), "Cambio medio (pp)": f"{x.cambio_medio_pp.mean():+.1f}", "n": len(x)})
tex_table(pd.DataFrame(rows), "tab_nodes_c", "Proporción de nodos que mejoran en el diagnóstico de entrenamiento (una arquitectura y una partición, $n=1$) y en el estudio de ruido (media $\\pm$ DE entre arquitecturas y sorteos).", "tab:nodesc", "lrrrrr")

# --- 8. hall of fame: ¿alguna arquitectura mejora a todos los nodos? -----------------------------------------
hf = pd.read_csv(f"{NL}/hof_per_model.csv")
REFN = {"media_inicial": "Accuracy media de la población inicial (por nodo)", "mejor_modelo_inicial": "Mejor modelo inicial (por accuracy media)", "mejor_inicial_por_nodo": "Mejor modelo inicial de cada nodo"}
rows = [{"Referencia": v, "Mejoran a todos los nodos": int(hf[k + "_todos_mejoran"].sum()), "Ningún nodo peor": int(hf[k + "_ningun_peor"].sum()),
         "Mediana de nodos que mejoran": f"{100 * hf[k + '_prop_mejora'].median():.0f}\\%", "Máximo": f"{100 * hf[k + '_prop_mejora'].max():.0f}\\%", "$\\geq 90\\%$ de nodos": int((hf[k + "_prop_mejora"] >= .9).sum())} for k, v in REFN.items()]
tex_table(pd.DataFrame(rows), "tab_hof", f"Arquitecturas del hall of fame ({len(hf)} en 20 corridas, {int(hf.en_inicial.sum())} ya presentes en la población inicial) frente a distintas referencias de la población inicial, por nodo.", "tab:hof", "lrrrrr")

# --- números sueltos para el texto --------------------------------------------------------------------------
# --- E5: NSGA-III con un objetivo por cluster ----------------------------------------------------------
E5 = {"n3_cs": "NSGA-III, clusters (propuesta)", "n3_rand": "NSGA-III, nodos aleatorios", "n3_cs_lownoise": "NSGA-III, clusters, menor ruido",
      "rs_cs": "Búsqueda aleatoria, clusters", "n2_cs": "NSGA-II 2 obj., clusters", "n2_rand": "NSGA-II 2 obj., nodos aleatorios"}
r = pd.read_csv("runs/e5/per_run.csv").groupby("arm")[["best_acc_media", "best_peor_cluster", "best_peor_decil", "best_params_norm", "media_mejoró"]].agg(["mean", "std"]).reindex(list(E5))
e5 = pd.DataFrame({"Brazo": [E5[a] for a in E5]})
for c, n in [("best_acc_media", "Acc. media"), ("best_peor_cluster", "Peor cluster"), ("best_peor_decil", "Peor decil"), ("best_params_norm", "Params. norm."), ("media_mejoró", "Nodos que mejoran")]:
    e5[n] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(r[(c, "mean")], r[(c, "std")])]
tex_table(e5, "tab_e5", "E5: mejor modelo del hall of fame (media $\\pm$ desv. de 5 semillas; prueba, 100 nodos). ``Nodos que mejoran'' no es comparable entre familias.", "tab:e5", "lccccc")
c5 = pd.read_csv("runs/e5/paired_contrasts.csv"); c5 = c5[c5.métrica.isin(["best_acc_media", "best_peor_cluster", "best_params_norm"])]
c5["Contraste"] = c5.contraste.str.replace("_", "\\_"); c5["Métrica"] = c5.métrica.str.replace("_", "\\_")
c5["Dif. [IC 95\\%]"] = [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for d, lo, hi in zip(c5.diferencia, c5.ic_bajo, c5.ic_alto)]
tex_table(c5[["Contraste", "Métrica", "Dif. [IC 95\\%]"]], "tab_e5c", "E5: contrastes pareados por semilla (bootstrap sobre 5 semillas, descriptivo).", "tab:e5c", "llc")

d = pd.read_csv("runs/training_diagnostic.csv")
num = {"trainAdamLow": d[(d.name == "adam_0.001") & (d.epochs == 2) & (d.rounds == 5)].set_index("alpha").acc_mean.to_dict(),
       "trainBest": d[(d.name == "adam_0.01") & (d.epochs == 5) & (d.rounds == 20)].set_index("alpha").acc_mean.to_dict()}
json.dump(num, open(f"{OUT}/numbers.json", "w"), indent=1, default=float)
print("assets ok", sorted(glob.glob(f"{OUT}/*")))
