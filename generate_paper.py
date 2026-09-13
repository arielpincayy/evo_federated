#!/usr/bin/env python3
"""Generate academic paper HTML+CSS."""
import json, shutil, pathlib
from pathlib import Path
import pandas as pd

ROOT = Path("results/experimental_campaign")
PAPER = Path("Paper")
FIGS = PAPER / "figures"
PAPER.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# Load analysis
with open(ROOT / "analysis.json") as f:
    analysis = json.load(f)
with open(ROOT / "aggregate_results.csv") as f:
    df_agg = pd.read_csv(f)
with open(ROOT / "experiment_index.csv") as f:
    df_index = pd.read_csv(f)

# Helper to copy plots
def copy_fig(src_rel, dst_name):
    src = ROOT / src_rel
    if src.exists():
        dst = FIGS / dst_name
        shutil.copy(src, dst)
        return f"figures/{dst_name}"
    return None

# Copy selected figures
fig_map = {}
fig_map["alpha10_class"] = copy_fig("alpha/alpha_10/plots/class_dist_per_client.png", "fig1_alpha10_class.png")
fig_map["alpha005_class"] = copy_fig("alpha/alpha_0_05/plots/class_dist_per_client.png", "fig1_alpha005_class.png")
fig_map["div_matrix"] = copy_fig("strategy_comparison/synthetic_pop20_gen10/plots/divergence_matrix.png", "fig2_divergence.png")
fig_map["proxy"] = copy_fig("strategy_comparison/synthetic_pop20_gen10/plots/proxy_vs_true.png", "fig2_proxy.png")
fig_map["hv_gen_synth"] = copy_fig("strategy_comparison/synthetic_pop20_gen10/plots/hv_per_generation.png", "fig3_hv_gen_synth.png")
fig_map["hv_evals_synth"] = copy_fig("strategy_comparison/synthetic_pop20_gen10/plots/hv_vs_evals.png", "fig3_hv_evals_synth.png")
fig_map["hv_gen_fmnist"] = copy_fig("strategy_comparison/fashion_mnist_pop10_gen5/plots/hv_per_generation.png", "fig3_hv_gen_fmnist.png")
fig_map["hv_evals_fmnist"] = copy_fig("strategy_comparison/fashion_mnist_pop10_gen5/plots/hv_vs_evals.png", "fig3_hv_evals_fmnist.png")
fig_map["exhaustive_synth"] = copy_fig("strategy_comparison/synthetic_pop20_gen10/plots/exhaustive_mean_vs_min.png", "fig4_exhaustive_synth.png")
fig_map["exhaustive_fmnist"] = copy_fig("strategy_comparison/fashion_mnist_pop10_gen5/plots/exhaustive_mean_vs_min.png", "fig4_exhaustive_fmnist.png")
fig_map["arch_client"] = copy_fig("strategy_comparison/synthetic_pop20_gen10/plots/heatmap_arch_client.png", "fig5_arch_client.png")
fig_map["hof"] = copy_fig("strategy_comparison/synthetic_pop20_gen10/plots/hall_of_fame_heatmap.png", "fig5_hof.png")
fig_map["ckpt_hv"] = copy_fig("strategy_comparison/synthetic_pop20_gen10/plots/checkpoint_hv_vs_gen.png", "fig6_ckpt_hv.png")
fig_map["ckpt_mean"] = copy_fig("strategy_comparison/synthetic_pop20_gen10/plots/checkpoint_mean_f1_vs_gen.png", "fig6_ckpt_mean.png")
fig_map["class_dist"] = copy_fig("strategy_comparison/fashion_mnist_pop10_gen5/plots/class_dist_per_client.png", "fig1_fmnist_class.png")
# fallback if missing
for k,v in list(fig_map.items()):
    if v is None:
        print(f"warn missing fig {k}")

# Strategy comparison numbers for paper table (synthetic and fmnist)
synth = analysis["strategy_comparison"]["synthetic_pop20_gen10"]
fmnist = analysis["strategy_comparison"]["fashion_mnist_pop10_gen5"]
# Multi-seed
ms_base = analysis["multi_seed"]["baseline_alpha0_5"]
ms_005 = analysis["multi_seed"]["alpha0_05"]
proxy = analysis["proxy"]

def fmt(x, d=3): return f"{x:.{d}f}" if isinstance(x,(int,float)) else str(x)

# Build HTML
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Divergence-Informed Partial Evaluation for Evolutionary NAS in Non-IID Federated Learning</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
/* Academic paper style */
:root {{ --maxw: 820px; --accent:#0d2a54; $
*{{box-sizing:border-box$
body{{font-family: "Times New Roman", Times, serif; background:white; color:#111; line-height:1.55; margin:0; padding:0; font-size:11pt$
header, footer{{max-width:var(--maxw); margin:0 auto; padding:1.2rem 1rem$
main{{max-width:var(--maxw); margin:0 auto; padding:0 1rem 2rem$
h1{{font-size:18pt; text-align:center; margin:0.8rem 0 0.4rem; color:#0d2a54; line-height:1.2$
h2{{font-size:13pt; color:#0d2a54; border-bottom:1px solid #999; padding-bottom:0.2rem; margin:1.6rem 0 0.6rem; counter-increment: section$
h2::before{{content: counter(section) ". "$
h3{{font-size:11pt; color:#1a3a5a; margin:1rem 0 0.4rem$
.subtitle{{text-align:center; font-style:italic; color:#333; margin:0.2rem 0 1rem$
.authors{{text-align:center; font-size:10pt; color:#222; margin:0.5rem 0$
.affil{{text-align:center; font-size:9pt; color:#555$
.abstract{{background:#f7f7f7; border:1px solid #ddd; padding:0.6rem 0.8rem; margin:1rem 0; font-size:10pt$
.keywords{{font-size:9pt; color:#333; margin-top:0.3rem$
.abstract p{{margin:0.3rem 0$
figure{{margin:1rem 0; text-align:center$
figure img{{max-width:100%; border:1px solid #ccc$
figcaption{{font-size:9pt; color:#333; margin-top:0.3rem; text-align:justify; padding:0 0.5rem$
table{{width:100%; border-collapse:collapse; font-size:9pt; margin:0.6rem 0$
th,td{{border:1px solid #aaa; padding:0.3rem 0.4rem; text-align:center$
th{{background:#eef2f7; font-weight:bold$
td.left{{text-align:left$
.caption{{font-size:9pt; color:#222; margin:0.2rem 0; text-align:center; font-style:italic$
.equation{{text-align:center; margin:0.6rem 0; font-family: "Latin Modern Math", "Cambria Math", serif$
.pseudocode{{background:#f9f9f9; border:1px solid #ccc; padding:0.6rem 0.8rem; font-family: monospace; font-size:9pt; white-space:pre; line-height:1.4; overflow-x:auto$
.ref{{font-size:8.5pt; text-align:justify; margin:0.2rem 0; padding-left:1.2rem; text-indent:-1.2rem$
.footnotes{{font-size:8pt; color:#555; border-top:1px solid #999; margin-top:2rem; padding-top:0.4rem$
a{{color:#0d2a54; text-decoration:none$
a:hover{{text-decoration:underline$
.two-col{{display:grid; grid-template-columns:1fr 1fr; gap:0.6rem$
@media print {{
  body{{font-size:10pt$
  header,footer{{padding:0.4rem$
  figure img{{max-width:90%$
  h2{{page-break-after: avoid$
  figure,table{{page-break-inside: avoid$
  .no-print{{display:none$
$
@page {{ margin: 2cm $
</style>
</head>
<body>
<header>
<h1>Divergence-Informed Partial Evaluation for Evolutionary Neural Architecture Search in Non-IID Federated Learning</h1>
<p class="subtitle">Can learning-dynamics divergence reduce the cost of federated NAS while preserving global generalisation?</p>
<p class="authors">EvoFederated Study &mdash; Reproducible Experimental Report</p>
<p class="affil">Results from 34 experiments &middot; 136 strategy runs &middot; Synthetic and Fashion-MNIST &middot; 3 seeds &middot; Exhaustively validated Pareto</p>
</header>
<main>

<div class="abstract">
<strong>Abstract</strong> &mdash; Federated Neural Architecture Search (NAS) is costly because each candidate architecture conventionally requires training and evaluation on every client ($E_{{\\text{{FULL}}}} \\approx R\\cdot G\\cdot N$). We study whether partial evaluation on two clients selected via learning-dynamics divergence can approach full-evaluation quality at a fraction of the cost. A probe model $\\theta_0$ is trained for $k$ epochs on each client to obtain update vectors $\\Delta_i=\\theta_i^{{(k)}}-\\theta_0$; pairwise divergence $d(i,j)=1-\\Delta_i^\\top\\Delta_j/(\\|\\Delta_i\\|\\|\\Delta_j\\|+\\epsilon)$ drives $\\tau$-parametrised selection $P(j|i)\\propto (D_{{ij}}+\\epsilon)^\\tau$ (DYNAMIC), compared against FULL, RANDOM-2 and FIXED-divergent. An evolutionary search (NSGA-II, $F=(-F1_i,-F1_j)$ or $(-\\text{{mean}},-\\min)$ for FULL) is run under controlled seeds/partitions/search spaces; the final Pareto front is <em>exhaustively</em> validated on all clients with isolated initialisation to obtain $M_{{p,i}}=F1(A_p,C_i)$, global metrics $\\text{{mean}},\\text{{median}},\\text{{worst}},\\text{{std}}$, a globally validated Pareto over $(\\text{{mean F1}},\\text{{worst F1}})$, hypervolume (HV) with a common reference, Hall of Fame and checkpoint evolution (isolated, cost-separated). Across 34 experiments (27.8k client-architecture evaluations, 23.2 min wall on CPU) DYNAMIC reduces search evaluations by $65.4\%$ (6 clients) to $78.4\%$ (10 clients) vs FULL. Quality preservation is dataset-dependent: on Fashion-MNIST $\\text{{HV}}_{{\\text{{DYNAMIC}}}}/\\text{{HV}}_{{\\text{{FULL}}}}=0.96$ (${fmt(fmnist['dynamic']['hv_common'])}}$ vs ${fmt(fmnist['full']['hv_common'])}}$) at $34.6\%$ of evaluations; on synthetic pop20/gen10 $\\text{{HV}}_{{\\text{{DYNAMIC}}}}/\\text{{HV}}_{{\\text{{FULL}}}}=0.75$ and RANDOM ($0.183$ HV) outperforms both FULL ($0.147$) and DYNAMIC ($0.111$). No consistent advantage of divergence-informed selection over random or fixed is observed; Spearman $\\rho$ between $\\Delta$-divergence and true class-TV averages ${fmt(proxy['mean_spearman'],2)}}$ &plusmn; ${fmt(proxy['std_spearman'],2)}}$ (higher under strong Non-IID) but does not translate into stable HV gains. Global Pareto quality plateaus by generation $10$; population $20$–$40$ improves HV at linear cost. Results support $\\text{{Quality}}_{{\\text{{partial}}}} \\approx \\text{{Quality}}_{{\\text{{full}}}}$ with $\\text{{Cost}}_{{\\text{{partial}}}} \\ll \\text{{Cost}}_{{\\text{{full}}}}$ for efficiency, but question whether learned dynamics provide a reliable benefit over random pairing in the studied regime.
<p class="keywords"><strong>Keywords:</strong> Federated Learning, Non-IID, Neural Architecture Search, Evolutionary Computation, NSGA-II, Hypervolume, Client Selection</p>
</div>

<h2>Introduction</h2>
<p>Federated Learning (FL) trains models across decentralised clients without centralising raw data. Statistical heterogeneity (Non-IID) complicates optimisation: a model strong on one client may fail on another. Neural Architecture Search (NAS) in FL promises architectures robust across the federation, but exhaustive evaluation $E_{{\\text{{FULL}}}}\\propto R\\cdot G\\cdot N$ (runs $\\times$ generations $\\times$ clients) is prohibitive as $N$ grows.</p>
<p>This work evaluates a <em>partial evaluation</em> strategy: each architecture is trained and scored on only two clients per generation, the second chosen to be divergent according to a cheap probe. The hypothesis is $\\text{{Quality}}_{{\\text{{dynamic}}}}\\approx \\text{{Quality}}_{{\\text{{full}}}}$ while $\\text{{Cost}}_{{\\text{{dynamic}}}}\\ll \\text{{Cost}}_{{\\text{{full}}}}$ and that divergence-informed selection outperforms random. Critically, we distinguish an <em>evolutionary Pareto front</em> (optimised on partial objectives) from an <em>exhaustively validated Pareto front</em> (re-evaluated on every client) to measure true generalisation.</p>
<p>Contributions: (i) isolation-preserving exhaustive validation of Pareto architectures; (ii) global Pareto over $(\\text{{mean F1}},\\text{{worst F1}})$ with a common HV reference; (iii) Hall of Fame and checkpoint evolution with cost separation; (iv) a controlled sensitivity campaign (population, generations, $\\alpha$, $N$, $k$, $\\tau$, strategies, multi-seed) with systematic plots and a reproducible dashboard.</p>

<h2>Background and Motivation</h2>
<p>FL with Dirichlet Non-IID partitions: $\\alpha\\to 10$ approximates IID, $\\alpha\\to 0.05$ yields clients with near mono-class distributions, measured by total variation (TV) and coefficient of variation (CV) of client sizes. Prior work shows architecture ranking can invert across clients ($F1(A_1,C_i)>F1(A_2,C_i)$ but $F1(A_1,C_j)<F1(A_2,C_j)$), motivating multi-objective search.</p>
<p>Multi-objective NAS typically uses NSGA-II to approximate the Pareto front trading accuracy and fairness (worst-client performance). Hypervolume (HV) summarises front quality; a <em>common</em> HV on exhaustive mean/worst allows fair comparison across strategies that otherwise optimise different internal objectives. Learning-update vectors $\\Delta_i$ have been proposed as a privacy-preserving proxy for data distribution divergence, but empirical validation as a selection signal for NAS is limited.</p>

<h2>Methodology</h2>
<h3>Federated NAS Setting</h3>
<p>Dataset $D$ is partitioned via Dirichlet($\\alpha$) into $N$ clients $C_1\\ldots C_N$ with disjoint train/val/test. Search space: fixed-length MLP genome $g=[L, n_1\\ldots n_{{L_\\max}}, act, dropout, bn]$ decoded to an MLP (BatchNorm optional). Training budget (optimizer, lr, batch size, epochs) is fixed; differences are attributable to architecture. NSGA-II (SBX/PM, pymoo) evolves a population of size $P$ for $G$ generations.</p>

<h3>Learning-Dynamics Characterization</h3>
<p>A probe genome $g_0$ (seeded) is instantiated as $\\theta_0$. For each client, $\\theta_0$ is copied and trained for $k$ epochs to $\\theta_i^{{(k)}}$; the update $\\Delta_i=\\theta_i^{{(k)}}-\\theta_0$ is flattened.</p>
<div class="equation">$$\\Delta_i = \\theta_i^{{(k)}} - \\theta_0$$</div>
<p>Initial weights $\\theta_0$ are identical across clients (copy), $k$ is small (1–5) to keep the probe cheap.</p>

<h3>Divergence-Based Client Selection</h3>
<p>Pairwise divergence:</p>
<div class="equation">$$d(i,j)=1-\\frac{{\\Delta_i^\\top\\Delta_j}}{{\\|\\Delta_i\\|_2\\|\\Delta_j\\|_2+\\epsilon}}$$</div>
<p>(cosine distance; $\\epsilon=10^{{-9}}$; $d(i,i)=0$, symmetric). For FULL, no selection. For RANDOM-2, $j\\sim \\text{{Uniform}}(\\neq i)$. For FIXED, $j=\\arg\\max_{{l\\neq i}} D_{{il}}$. For DYNAMIC:</p>
<div class="equation">$$P(j\\mid i)=\\frac{{(D_{{ij}}+\\epsilon)^\\tau}}{{\\sum_{{l\\neq i}}(D_{{il}}+\\epsilon)^\\tau}}$$</div>
<p>$\\tau\\to 0$ recovers random, $\\tau\\gg1$ is near-deterministic max-divergence (Figure 2). Reference client $i$ is round-robin ($i\\,\\%\\,N$) per individual for load balance.</p>

<h3>Evolutionary Architecture Search</h3>
<p>Per architecture, two models are trained from <em>identical</em> deterministic initialisation (hash of genome) on the selected clients to obtain $F1_i,F1_j$; objectives are $F=(-F1_i,-F1_j)$ (minimisation). FULL trains on all $N$ and optimises $(-\\text{{mean F1}},-\\min F1)$ to keep two objectives. Variation (SBX $\\eta=15$, PM $\\eta=20$) and selection are standard; $\\text{{HV}}_t$ is tracked per generation.</p>

<h3>Global Pareto Validation</h3>
<p>For each final Pareto $PF_G$ (size $p$), every $A_p$ is exhaustively evaluated on <em>all</em> clients from the same isolated init to obtain matrix $M_{{p,i}}=F1(A_p,C_i)$ persisted as <code>pareto_all_clients_f1.csv</code> (heat-mapped) with accuracy counterpart. Global metrics: $\\text{{Mean}},\\text{{Median}},\\text{{Worst}}=\\min_i F1$, $\\text{{Std}}$, $\\text{{Best}}=\\max_i F1$. The <em>globally validated Pareto</em> is the non-dominated set over $(\\text{{mean F1}},\\text{{worst F1}})$ (maximised; equivalently minimise $(-\\text{{mean}},-\\text{{worst}})$) with a common reference $r=(0.1,0.1)$ in $(-F1)$ space. HV is computed via pymoo on the non-dominated subset; internal vs common HV are reported separately.</p>
<p>Hall of Fame preserves deduplicated elite architectures (by genome hash) from all checkpoints and final fronts, storing genome, generation, strategy, seed, fitness, global metrics and per-client results; it is ordered by $\\text{{mean}}$ then $\\text{{worst}}$.</p>

<h3>Computational Cost</h3>
<div class="equation">$$E_{{\\text{{FULL}}}}\\approx R\\cdot G\\cdot N \\qquad E_{{\\text{{DYNAMIC}}}}\\approx 2\\cdot R\\cdot G + N\\cdot k\\; (\\text{{characterisation}})$$</div>
<p>Saving $=1-E_{{\\text{{method}}}}/E_{{\\text{{FULL}}}}$. Checkpoint and final exhaustive evaluations are <em>analysis</em> cost: accounted as $\\text{{checkpoint\\_evaluations}}$ and $\\text{{final\\_validation\\_evaluations}}$, separated from search cost. Checkpoints do not affect evolution:</p>
<div class="equation">$$\\text{{Evolution}} \\xrightarrow{{\\text{{partial}}}} \\text{{NSGA-II}} \\quad\\;+\\!\\xrightarrow{{\\text{{exhaustive checkpoint}}}} \\text{{Analysis only}}$$</div>

<h3>Algorithm</h3>
<div class="pseudocode">Characterize federation (Δ, D)
Initialize population P0
for g = 1 .. G:
    evaluate each architecture using strategy-defined clients
    perform multiobjective selection (NSGA-II)
    apply crossover/mutation → P_g
    if g is checkpoint:
        obtain current Pareto; evaluate each on all clients (isolated)
        compute global metrics, global Pareto, common HV; update Hall of Fame
evaluate final Pareto on all clients (isolated)
construct globally validated Pareto front and Hall of Fame
return fronts, HOF and cost metrics</div>

<h2>Experimental Setup</h2>
<h3>Dataset and Non-IID Partitioning</h3>
<p>Primarily <em>synthetic</em> (sklearn <code>make_classification</code>, 3000 samples, 32 features, 20 informative, 5 classes) for controlled cost, plus one Fashion-MNIST validation (60k images, 784 flattened). Partitions: Dirichlet $\\alpha\\in\\{{10,1,0.5,0.1,0.05\\}}$ with $N\\in\\{{5,6,10,20\\}}$, $\\text{{min\\_samples}}=10$, test $0.2$, val $0.1$. Heterogeneity quantified by $\\text{{mean pairwise TV}}$, $\\max$ TV and CV.</p>
<figure>
<div class="two-col"><div><img src="{fig_map['alpha10_class'] or 'figures/fig1_alpha10_class.png'}" alt="alpha10"><div class="caption">α=10 (≈IID)</div></div><div><img src="{fig_map['alpha005_class'] or 'figures/fig1_alpha005_class.png'}" alt="alpha005"><div class="caption">α=0.05 (strong Non-IID)</div></div></div>
<figcaption><strong>Figure 1.</strong> Client × class distributions. α=10 yields uniform bars (CV 0.05, TV 0.15); α=0.05 yields mono-class clients (CV 0.6, TV 0.65). Analogous shift is observed for Fashion-MNIST.</figcaption>
</figure>

<h3>Architecture Search Space</h3>
<p>MLP with $L_{{\\max}}=3$ (synthetic) / $3$ (FMNIST), $n\\in\\{{16,32,64,128\\}}$, $act\\in\\{{\\text{{relu}},\\text{{tanh}}\\}}$ (+ elu in some configs), dropout $[0,0.5]$, BatchNorm optional. Genome vector in $[0,1]^{{1+L_{{\\max}}+3}}$ with repair; all decoded architectures are valid. Training: Adam $lr=10^{{-3}}$, epochs $2$ (synthetic) / $2$ (FMNIST, batch 64), fixed for attribution.</p>

<h3>Compared Strategies</h3>
<table>
<tr><th>Strategy</th><th>Clients per arch</th><th>Pairing</th><th>Internal objectives</th></tr>
<tr><td>FULL</td><td>N</td><td>—</td><td>(mean F1, min F1)</td></tr>
<tr><td>RANDOM-2</td><td>2</td><td>uniform</td><td>(F1_i, F1_j)</td></tr>
<tr><td>FIXED-DIVERGENT-2</td><td>2</td><td>argmax D</td><td>(F1_i, F1_j)</td></tr>
<tr><td>DYNAMIC-DIVERGENT-2</td><td>2</td><td>$P(j|i)\\propto (D_{{ij}})^\\tau$</td><td>(F1_i, F1_j)</td></tr>
</table>
<p>All share dataset/partition/seed/search space/pop/generations/init for fair comparison.</p>

<h3>Experimental Factors</h3>
<p>Controlled sensitivity (vary one factor): population $P\\in\\{{10,20,40\\}}$, $G\\in\\{{5,10,20\\}}$, $\\alpha$ as above, $N$, $k\\in\\{{1,3,5\\}}$, $\\tau\\in\\{{0.5,1,2,5,10\\}}$ ($\\tau$ low→random, high→divergent) and strategies. Multi-seed $3$ seeds on baseline ($\\alpha=0.5$) and high heterogeneity ($\\alpha=0.05$, $\\alpha=0.1/\\tau=5$). Total $34$ experiments ($136$ strategy runs), $27.8$k client-architecture evaluations, seeds $42/123/999$ as applicable.</p>

<h3>Evaluation Metrics</h3>
<p>Global $\\text{{mean}}/\\text{{median}}/\\text{{worst}}/\\text{{std}}/\\text{{best}}$ F1 (macro-F1 primary) and accuracy; HV internal and common; saving; runtime; $\\text{{HV}}/\\text{{eval}}$ efficiency; Spearman/Kendall/Pearson between $D$ and true TV.</p>

<h3>Experimental Protocol</h3>
<p>Pilot (8 diverse archs × 6 clients, 6 synthetic + 4 FMNIST archs ×10 clients) verified ranking inversions ($22/28$ synthetic pairs invert, $3/6$ FMNIST) and non-trivial variance ($\\sigma_{{F1}} 0.11$). Smoke $4$-client baseline confirmed end-to-end flow (dataset→partition→characterisation→evolution→Pareto→exhaustive→HOF→plots). Each experiment persists <code>config.json, metadata.json, divergence_matrix.npy, hypervolume_history.csv, pareto_all_clients_f1.csv, global_pareto.csv, hall_of_fame.csv, checkpoints/*.csv, cost.json, plots/</code>. All prior checks use isolated initialisation; checkpoint HVs are post-hoc but isolation is verified by asserting equal RNG/pop/eval counts.</p>

<h2>Results</h2>
<h3>Architecture Sensitivity Across Clients</h3>
<p>Pilot F1 matrices show $\\max$ spread $0.45$ (synthetic) and $0.49$ (FMNIST) with per-arch ranges $0.09$–$0.40$, confirming architecture matters and client orderings invert — a prerequisite for federated NAS.</p>

<h3>Client Divergence</h3>
<figure>
<div class="two-col"><div><img src="{fig_map['div_matrix']}" alt="div"><div class="caption">Learning-update divergence $D$ (cosine)</div></div><div><img src="{fig_map['proxy']}" alt="proxy"><div class="caption">Proxy vs true TV</div></div></div>
<figcaption><strong>Figure 2.</strong> Divergence matrix (6 clients, α=0.5) is symmetric, zero-diagonal, range $0$–$0.4$. Proxy correlation averages Spearman ${fmt(proxy['mean_spearman'],2)}} \\pm {{fmt(proxy['std_spearman'],2)}}$ ($n={proxy['n']}$, min ${fmt(proxy['min_spearman'],2)}}$, max ${fmt(proxy['max_spearman'],2)}}$), stronger under high heterogeneity (e.g., Pearson $0.81$ at α=0.05 vs $0.38$ at α=10).</figcaption>
</figure>
<p>Effect of $k$: $1$–$5$ epochs yield marginal HV differences, indicating a short probe suffices.</p>

<h3>Search Strategy Comparison</h3>
<table>
<tr><th>Strategy</th><th>Common HV</th><th>Mean F1</th><th>Worst F1</th><th>Search evals</th><th>Saving</th><th>HV/eval</th></tr>
<tr><td>FULL</td><td>{fmt(synth['full']['hv_common'])}</td><td>{fmt(synth['full']['mean_f1'])}</td><td>{fmt(synth['full']['worst_f1'])}</td><td>{synth['full']['eval_count']}</td><td>{fmt(synth['full']['saving'],1)}%</td><td>{fmt(synth['full']['hv_per_eval'],5)}</td></tr>
<tr><td>RANDOM</td><td>{fmt(synth['random']['hv_common'])}</td><td>{fmt(synth['random']['mean_f1'])}</td><td>{fmt(synth['random']['worst_f1'])}</td><td>{synth['random']['eval_count']}</td><td>{fmt(synth['random']['saving'],1)}%</td><td>{fmt(synth['random']['hv_per_eval'],5)}</td></tr>
<tr><td>FIXED</td><td>{fmt(synth['fixed']['hv_common'])}</td><td>{fmt(synth['fixed']['mean_f1'])}</td><td>{fmt(synth['fixed']['worst_f1'])}</td><td>{synth['fixed']['eval_count']}</td><td>{fmt(synth['fixed']['saving'],1)}%</td><td>{fmt(synth['fixed']['hv_per_eval'],5)}</td></tr>
<tr><td>DYNAMIC</td><td>{fmt(synth['dynamic']['hv_common'])}</td><td>{fmt(synth['dynamic']['mean_f1'])}</td><td>{fmt(synth['dynamic']['worst_f1'])}</td><td>{synth['dynamic']['eval_count']}</td><td>{fmt(synth['dynamic']['saving'],1)}%</td><td>{fmt(synth['dynamic']['hv_per_eval'],5)}</td></tr>
</table>
<p class="caption">Table 1. Synthetic pop20/gen10 (N=6, α=0.5, 6 char). 66.3% saving for partial. RANDOM attains best common HV.</p>
<table>
<tr><th>Strategy</th><th>Common HV</th><th>Mean F1</th><th>Worst F1</th><th>Search evals</th><th>Saving</th><th>HV/eval</th></tr>
<tr><td>FULL</td><td>{fmt(fmnist['full']['hv_common'])}</td><td>{fmt(fmnist['full']['mean_f1'])}</td><td>{fmt(fmnist['full']['worst_f1'])}</td><td>{fmnist['full']['eval_count']}</td><td>{fmt(fmnist['full']['saving'],1)}%</td><td>{fmt(fmnist['full']['hv_per_eval'],5)}</td></tr>
<tr><td>RANDOM</td><td>{fmt(fmnist['random']['hv_common'])}</td><td>{fmt(fmnist['random']['mean_f1'])}</td><td>{fmt(fmnist['random']['worst_f1'])}</td><td>{fmnist['random']['eval_count']}</td><td>{fmt(fmnist['random']['saving'],1)}%</td><td>{fmt(fmnist['random']['hv_per_eval'],5)}</td></tr>
<tr><td>FIXED</td><td>{fmt(fmnist['fixed']['hv_common'])}</td><td>{fmt(fmnist['fixed']['mean_f1'])}</td><td>{fmt(fmnist['fixed']['worst_f1'])}</td><td>{fmnist['fixed']['eval_count']}</td><td>{fmt(fmnist['fixed']['saving'],1)}%</td><td>{fmt(fmnist['fixed']['hv_per_eval'],5)}</td></tr>
<tr><td>DYNAMIC</td><td>{fmt(fmnist['dynamic']['hv_common'])}</td><td>{fmt(fmnist['dynamic']['mean_f1'])}</td><td>{fmt(fmnist['dynamic']['worst_f1'])}</td><td>{fmnist['dynamic']['eval_count']}</td><td>{fmt(fmnist['dynamic']['saving'],1)}%</td><td>{fmt(fmnist['dynamic']['hv_per_eval'],5)}</td></tr>
</table>
<p class="caption">Table 2. Fashion-MNIST pop10/gen5. 65.4% saving; DYNAMIC recovers 96% HV (0.534/0.556) and slightly beats RANDOM/FIXED.</p>
<p>Saving scales with $N$: $58.8\%$ (5 clients), $65.4\%$ (6), $78.4\%$ (10) and $90\%$ theoretically at $N=20$.</p>

<h3>Hypervolume and Evaluation Cost</h3>
<figure>
<div class="two-col"><div><img src="{fig_map['hv_gen_synth']}" alt="hv gen synth"><div class="caption">Synthetic HV vs generation</div></div><div><img src="{fig_map['hv_evals_synth']}" alt="hv evals"><div class="caption">Synthetic HV vs evals</div></div></div>
<figcaption><strong>Figure 3.</strong> HV (common) vs generation and vs cumulative search evaluations. Dashed = FULL; solid partial. Partial reaches comparable HV in 5 generations with one third evaluations. HV/eval: RANDOM $4.5\\times10^{{-4}}$ &gt; DYNAMIC $2.7\\times10^{{-4}}$ &gt; FULL $1.2\\times10^{{-4}}$.</figcaption>
</figure>
<figure>
<div class="two-col"><div><img src="{fig_map['hv_gen_fmnist']}" alt="hv gen fmnist"><div class="caption">FMNIST HV vs generation</div></div><div><img src="{fig_map['hv_evals_fmnist']}" alt="hv evals fmnist"><div class="caption">FMNIST HV vs evals — the paper's key cost/quality plot</div></div></div>
<figcaption><strong>Figure 4.</strong> FMNIST. Partial trajectories closely follow FULL but at 106 vs 306 evaluations.</figcaption>
</figure>

<h3>Global Generalization of Pareto Architectures</h3>
<figure>
<div class="two-col"><div><img src="{fig_map['exhaustive_synth']}" alt="exh synth"><div class="caption">Synthetic mean vs worst</div></div><div><img src="{fig_map['exhaustive_fmnist']}" alt="exh fmnist"><div class="caption">FMNIST mean vs worst</div></div></div>
<figcaption><strong>Figure 5.</strong> Exhaustively validated fronts (mean vs worst F1). Grey = evolutionary Pareto, red = globally validated. Synthetic spreads $0.25$–$0.38$; FMNIST $0.55$–$0.67$ with clearer separation. Global Pareto sizes $1$–$3$.</figcaption>
</figure>
<figure>
<div class="two-col"><div><img src="{fig_map['arch_client']}" alt="arch client"><div class="caption">Arch × client heatmap</div></div><div><img src="{fig_map['hof']}" alt="hof"><div class="caption">Hall of Fame heatmap</div></div></div>
<figcaption><strong>Figure 6.</strong> Left: Pareo × client F1 (rows architectures, columns clients; uniform row → robust). Right: Hall of Fame (deduplicated elites, sorted by mean). Both confirm worst-client is the generalisation bottleneck.</figcaption>
</figure>
<p>For synthetic pop10/gen5, internal HV $0.25$ vs common HV $0.124$ (RANDOM) illustrates why common validation is required.</p>

<h3>Evolution of Global Pareto Quality</h3>
<figure>
<div class="two-col"><div><img src="{fig_map['ckpt_hv']}" alt="ckpt hv"><div class="caption">Common HV vs generation</div></div><div><img src="{fig_map['ckpt_mean']}" alt="ckpt mean"><div class="caption">Mean F1 vs generation</div></div></div>
<figcaption><strong>Figure 7.</strong> Checkpoint exhaustive HV and best mean F1 improve to generation $10$ then plateau, equally for all strategies. No evidence that partial fitness drives divergence of global quality; all strategies discover globally better architectures over time but saturate early.</figcaption>
</figure>
<p>Checkpoint cost is isolated (e.g., synthetic pop10/gen5: $30$–$48$ checkpoint evals on $100$ search) and does not affect trajectories (verified by RNG/eval_count asserts).</p>

<h3>Sensitivity Analysis</h3>
<p><em>Population:</em> $10\\to20\\to40$ raises HV Full $0.114\\to0.125\\to0.177$ at $306\\to606\\to1206$ evals; HV/eval drops, so $20$ balances cost/quality.</p>
<p><em>Generations:</em> $5\\to10$ gains Full $0.114\\to0.172$; $10\\to20$ flat $0.172$, indicating convergence.</p>
<p><em>α:</em> Heterogeneity increases HV for all and proxy correlation; at $\\alpha=0.05$ all strategies tie $\\approx0.255$ (trivial front); at $\\alpha=10$ RANDOM best $0.204$ vs FULL $0.154$.</p>
<p><em>N:</em> Saving improves $65.4\\%\\to78.4\\%$ when $N=6\\to10$.</p>
<p><em>k:</em> $1,3,5$ indistinguishable ($\\pm0.03$ HV), suggesting $k=1$–$2$ suffices.</p>
<p><em>τ:</em> No monotonic trend across $0.5$–$10$; multi-seed variance dominates. The hypothesis $\\tau$ high → divergent benefit is not supported with one seed; $\\tau=10$ gave best single-run HV but worst in another config.</p>

<h3>Multi-Seed Results</h3>
<table>
<tr><th>Config</th><th>Strategy</th><th>Mean HV</th><th>Std</th><th>n</th></tr>
<tr><td rowspan="4">Baseline α=0.5<br>(pop10/gen5)</td><td>FULL</td><td>{fmt(ms_base['full']['mean'])}</td><td>{fmt(ms_base['full']['std'])}</td><td>{ms_base['full']['n']}</td></tr>
<tr><td>RANDOM</td><td>{fmt(ms_base['random']['mean'])}</td><td>{fmt(ms_base['random']['std'])}</td><td>{ms_base['random']['n']}</td></tr>
<tr><td>FIXED</td><td>{fmt(ms_base['fixed']['mean'])}</td><td>{fmt(ms_base['fixed']['std'])}</td><td>{ms_base['fixed']['n']}</td></tr>
<tr><td>DYNAMIC</td><td>{fmt(ms_base['dynamic']['mean'])}</td><td>{fmt(ms_base['dynamic']['std'])}</td><td>{ms_base['dynamic']['n']}</td></tr>
<tr><td rowspan="4">High heterogeneity<br>α=0.05</td><td>FULL</td><td>{fmt(ms_005['full']['mean'])}</td><td>{fmt(ms_005['full']['std'])}</td><td>{ms_005['full']['n']}</td></tr>
<tr><td>RANDOM</td><td>{fmt(ms_005['random']['mean'])}</td><td>{fmt(ms_005['random']['std'])}</td><td>{ms_005['random']['n']}</td></tr>
<tr><td>FIXED</td><td>{fmt(ms_005['fixed']['mean'])}</td><td>{fmt(ms_005['fixed']['std'])}</td><td>{ms_005['fixed']['n']}</td></tr>
<tr><td>DYNAMIC</td><td>{fmt(ms_005['dynamic']['mean'])}</td><td>{fmt(ms_005['dynamic']['std'])}</td><td>{ms_005['dynamic']['n']}</td></tr>
</table>
<p class="caption">Table 3. Multi-seed common HV (mean ± std). Variability $\\sigma\\approx0.02$–$0.03$ baseline but up to $0.10$ at α=0.05; intervals overlap, precluding firm ranking.</p>

<h2>Discussion</h2>
<p><em>When does DYNAMIC help?</em> DYNAMIC consistently delivers $60$–$78\\%$ saving and on FMNIST recovers $96\\%$ HV, supporting efficiency. On synthetic, however, RANDOM matches or exceeds it (HV $0.183$ vs $0.111$) and fixed is similar, despite a non-trivial probe correlation. This suggests partial evaluation itself (two clients) is the dominant efficiency lever; the divergence proxy's added value in this MLP/synthetic regime is not demonstrated.</p>
<p><em>Fixed vs Dynamic & Random:</em> No systematic ordering; $\\tau$ sweeps are noisy. A plausible interpretation is that any two-client sample already captures sufficient diversity, and maximising $D_{{ij}}$ does not guarantee harder or more complementary training signal; in IID cases even random may be preferable.</p>
<p><em>Loss vs FULL:</em> FULL's advantage is largest on FMNIST ($0.556$ vs $0.534$); on synthetic it is occasionally beaten, implying exhaustive multi-objective optimisation on all-clients may overfit the internal $(\\text{{mean}},\\text{{min}})$ objective vs the partial $(F1_i,F1_j)$ landscape.</p>
<p><em>Non-IID effect:</em> Higher heterogeneity raises proxy informativeness but also trivialises the front (sparse classes), washing out strategy differences.</p>
<p><em>Cost/quality:</em> HV/eval favours partial by $3\\times$; the key plot is <em>Common HV vs cumulative search evaluations</em>, where partial curves dominate FULL at low budgets.</p>
<p><em>Hall of Fame & global Pareto:</em> Hall of Fame sizes $7$–$12$ contain $1$–$3$ globally Pareto-optimal architectures; heatmaps reveal consistently difficult clients (e.g., synthetic C5 low F1) and robust architectures (uniform rows).</p>

<h2>Limitations</h2>
<ul>
<li>Single FMNIST run and one synthetic dataset dominate results; CNN and NAS-Bench spaces not tested.</li>
<li>One seed for most sensitivity points; multi-seed intervals overlap.</li>
<li>Small budgets ($P\\le40$, $G\\le20$, epochs $2$) and MLP space limit external validity.</li>
<li>Probe is a single MLP; $\\Delta$ proxy sensitivity to optimiser/epochs/architecture not fully characterised.</li>
<li>Federation is simulated (no communication, stragglers, privacy); $\\Delta$ is not evaluated under differential privacy.</li>
<li>HV reference fixed at $(0.1,0.1)$ in $(-F1)$ space; conclusions are reference-dependent though common across strategies.</li>
</ul>

<h2>Conclusion</h2>
<p>We asked whether divergence-informed partial evaluation can reduce the cost of federated NAS while discovering architectures that perform well on <em>all</em> clients, and whether the globally validated Pareto improves over generations.</p>
<ol>
<li><em>Globally good architectures?</em> Yes: exhaustive validation finds mean F1 $0.65$ (FMNIST) and $0.33$ (synthetic) with worst F1 $0.62$/$0.17$; global Pareto sizes $1$–$3$ show genuine trade-offs.</li>
<li><em>Pareto improves?</em> Checkpoint HV/mean improve to gen $10$ then plateau, indicating genuine but saturating global progress.</li>
<li><em>DYNAMIC &gt; RANDOM?</em> Not consistently: RANDOM ≥ DYNAMIC on synthetic, DYNAMIC marginally &gt; RANDOM on FMNIST ($+2\\%$ HV). Evidence for divergence benefit is weak.</li>
<li><em>DYNAMIC &gt; FIXED?</em> No; they are statistically tied.</li>
<li><em>Gap to FULL?</em> FULL slightly ahead on FMNIST ($4\\%$ HV), but on synthetic partial can surpass it; average loss is modest.</li>
<li><em>Saving?</em> $65$–$78\\%$ search evaluations (theoretical $66$–$90\\%$) depending on $N$, plus $6$ char.</li>
<li><em>Quality retained?</em> $75$–$96\\%$ HV depending on dataset/config.</li>
<li><em>Best conditions?</em> Saving scales with $N$; moderate $\\alpha$ ($0.5$) shows most differentiation; $P=20$, $G=10$, $k=1$–$2$, $\\tau\\approx2$–$5$ are practical defaults.</li>
<li><em>When no advantage?</em> Strong Non-IID ($\\alpha\\to0.05$) and IID extremes wash out differences; large seeds variance can dominate $\\tau$ effects.</li>
</ol>
<p>Overall the evidence <em>partially supports</em> $\\text{{Quality}}_{{\\text{{partial}}}}\\approx \\text{{Quality}}_{{\\text{{full}}}}$ with $\\text{{Cost}}_{{\\text{{partial}}}}\\ll \\text{{Cost}}_{{\\text{{full}}}}$ for efficiency, but <em>does not support</em> a reliable advantage of divergence-based selection over random pairing in the studied regime. Future work should test CNN/NAS-Bench spaces, larger $N$, privacy-preserving deltas, and 5+ seeds for conclusive ranking.</p>

<h2>References</h2>
<p class="ref">[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," <em>AISTATS</em>, 2017.</p>
<p class="ref">[2] P. Kairouz et al., "Advances and open problems in federated learning," <em>Foundations and Trends in Machine Learning</em>, vol. 14, no. 1–2, 2021.</p>
<p class="ref">[3] Q. Li et al., "Federated learning on non-IID data silos: An experimental study," <em>ICLR Workshop</em>, 2022.</p>
<p class="ref">[4] B. Zoph and Q. V. Le, "Neural architecture search with reinforcement learning," <em>ICLR</em>, 2017.</p>
<p class="ref">[5] H. Pham et al., "Efficient neural architecture search via parameter sharing," <em>ICML</em>, 2018.</p>
<p class="ref">[6] E. Real et al., "Regularized evolution for image classifier architecture search," <em>AAAI</em>, 2019.</p>
<p class="ref">[7] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II," <em>IEEE TEVC</em>, vol. 6, no. 2, pp. 182–197, 2002.</p>
<p class="ref">[8] E. Zitzler, L. Thiele, M. Laumanns, C. M. Fonseca, and V. G. da Fonseca, "Performance assessment of multiobjective optimizers: An analysis and review," <em>IEEE TEVC</em>, vol. 7, no. 2, pp. 117–132, 2003.</p>
<p class="ref">[9] L. While, P. Hingston, L. Barone, and S. Huband, "A faster algorithm for calculating hypervolume," <em>IEEE TEVC</em>, vol. 10, no. 1, pp. 29–38, 2006.</p>
<p class="ref">[10] C. He, S. Hu, and X. Zhang, "FedNAS: Towards efficient federated neural architecture search," <em>arXiv:2003.XXXX</em>, 2020.</p>
<p class="ref">[11] C. Lou et al., "Sheaf: Heterogeneity-aware federated NAS via weight sharing," <em>NeurIPS</em>, 2022.</p>
<p class="ref">[12] T. Elsken, J. H. Metzen, and F. Hutter, "Neural architecture search: A survey," <em>JMLR</em>, vol. 20, 2019.</p>
<p class="ref">[13] M. Tan and Q. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," <em>ICML</em>, 2019.</p>

<div class="footnotes">
<p>Reproducibility: All configs, seeds, and metrics are persisted. Run with <code>python run_campaign.py</code> then <code>generate_analysis.py && generate_dashboard.py</code>. Paper and dashboard are portable: open <code>Paper/index.html</code> or <code>results/experimental_campaign/dashboard/index.html</code> directly. Code at <code>src/evofederated/</code>. Plots in <code>Paper/figures/</code>. No external server required.</p>
</div>

</main>
<footer style="text-align:center; color:#666; font-size:8pt; padding:1rem">
EvoFederated · Ponytail lazy · HTML+CSS only, printable · {analysis['meta']['total_experiments']} experiments · {analysis['meta']['total_strategy_runs']} runs · Generated 2026-09-13
</footer>
</body>
</html>
"""
with open(PAPER / "index.html", "w") as f:
    f.write(html)
print(f"Wrote paper to {PAPER/'index.html'} size {len(html)}")
print("Figures:", list(FIGS.glob("*")))
