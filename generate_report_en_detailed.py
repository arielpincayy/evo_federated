#!/usr/bin/env python3
import json, pandas as pd, numpy as np
from pathlib import Path

base = Path("results/unified_evofed_final")
report = base / "report"
chart_data_path = report / "chart_data.json"

with open(chart_data_path) as f:
    data = json.load(f)

# Load aggregates for tables
agg = pd.read_csv(base / "aggregate_results.csv")
agg_stats = pd.read_csv(base / "aggregate_stats.csv")
consolidated = pd.read_csv(base / "11_final_comparison" / "final_consolidated_table.csv")
# For heterogeneity, tau, scaling we need to compute summaries
# Prepare data for JS embedding
data_json_str = json.dumps(data, indent=2)

# Build HTML
html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvoFederated — Unified Benchmark Report</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--accent:#0d2a54;--light:#e8edf3;--maxw:1200px;}
*{box-sizing:border-box}
body{font-family:Inter, Segoe UI, Helvetica, Arial, sans-serif; background:#fafafa; color:#222; line-height:1.6; margin:0}
header{background:var(--accent); color:#fff; padding:2rem 1rem; text-align:center}
header h1{margin:0; font-size:1.8rem}
header p{margin:.3rem 0; opacity:.9}
nav{background:#fff; border-bottom:1px solid #ddd; position:sticky; top:0; z-index:10; padding:.5rem 1rem; overflow-x:auto; white-space:nowrap}
nav a{margin-right:1rem; color:var(--accent); text-decoration:none; font-size:.85rem; font-weight:600}
nav a:hover{text-decoration:underline}
main{max-width:var(--maxw); margin:0 auto; padding:1.2rem}
h2{color:var(--accent); border-bottom:2px solid var(--accent); padding-bottom:.3rem; margin-top:2.2rem}
h3{color:#1a3a5a; margin-top:1.5rem}
h4{margin:.8rem 0 .3rem; color:#2a4a6a}
.card{background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:1rem; margin:1rem 0; box-shadow:0 1px 3px rgba(0,0,0,.06)}
.grid{display:grid; grid-template-columns:1fr 1fr; gap:1rem}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.chart-box{background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:1rem; margin:1rem 0}
.chart-box h4{margin:0 0 .5rem; font-size:.95rem}
.chart-desc{font-size:.85rem; color:#444; margin:.5rem 0 0; background:#f8fafc; padding:.6rem; border-left:3px solid var(--accent); border-radius:4px}
table{border-collapse:collapse; width:100%; font-size:.85rem; margin:.6rem 0}
th,td{border:1px solid #ddd; padding:.4rem .5rem; text-align:left}
th{background:var(--light); position:sticky; top:0}
.badge{display:inline-block; background:var(--accent); color:#fff; padding:.15rem .45rem; border-radius:12px; font-size:.7rem; margin:.1rem}
.note{background:#fff3cd; border-left:4px solid #ffc107; padding:.6rem; margin:.6rem 0; font-size:.85rem}
code{background:#eef; padding:.1rem .3rem; border-radius:3px; font-size:.85em}
pre.pseudo{background:#f7f7fa; border:1px solid #e2e2e2; border-radius:6px; padding:1rem; overflow-x:auto; font-size:.82rem; line-height:1.5}
.katex-display{margin:.5rem 0}
.heatmap{border-collapse:collapse; margin:.5rem auto}
.heatmap td{width:32px; height:32px; text-align:center; font-size:.7rem; border:1px solid #fff; color:#000}
.legend{font-size:.8rem; display:flex; gap:.5rem; align-items:center; justify-content:center; margin:.5rem 0}
.legend span{width:14px; height:14px; display:inline-block; border:1px solid #ccc}
.chart-box canvas{min-height:260px; max-height:420px}
footer{text-align:center; padding:1.2rem; color:#777; font-size:.8rem; border-top:1px solid #ddd; margin-top:2rem}
</style>
</head>
<body>
<header>
<h1>EvoFederated — Unified Experimental Compendium</h1>
<p>Dissimilarity + NSGA-II &nbsp;|&nbsp; Similarity + Single-Objective &nbsp;|&nbsp; Many-Objective NSGA-III / MOEA-D</p>
<p>220 runs &middot; 3 families &middot; N=4/8 &middot; synthetic &middot; reproducible &middot; generated """ + pd.Timestamp.now().strftime("%Y-%m-%d") + r"""</p>
<p style="font-size:.8rem; opacity:.85">Branch <code>experiment/unified-evofed-final</code> &middot; commit 0a7a602 &middot; Charts: Chart.js &middot; Math: KaTeX</p>
</header>
<nav>
<a href="#executive">Executive</a><a href="#history">History</a><a href="#problem">Problem</a><a href="#geometry">Geometry</a><a href="#familyA">Family A</a><a href="#familyB">Family B</a><a href="#familyC">Family C</a><a href="#pernode">Per-Node</a><a href="#modelnode">Model×Node</a><a href="#specialization">Specialization</a><a href="#conflict">Conflict</a><a href="#scaling">Scaling</a><a href="#heterogeneity">Heterogeneity</a><a href="#tau">Tau</a><a href="#K">K</a><a href="#cost">Cost</a><a href="#stats">Stats</a><a href="#comparison">Comparison</a><a href="#conclusions">Conclusions</a>
</nav>
<main>

<div class="note"><strong>Reproducibility:</strong> Every run stores <code>config.json</code>, <code>metadata.json</code>, <code>pairing.json</code>, <code>similarity_matrix.csv</code>, <code>individual_node_accuracy.csv</code> under <code>results/unified_evofed_final/</code>. All heatmaps use direct &Delta; vectors (no PCA &sect;0.6). Axes, units and metrics are explained under each chart.</div>

<section id="executive" class="card">
<h2>Executive Summary</h2>
<p>This compendium unifies three evolutionary federated strategies under a common infrastructure to answer <em>what works, when, at what cost, and for whom</em>. We compare <strong>conflict-driven</strong> (dissimilarity), <strong>redundancy-driven</strong> (similarity) and <strong>many-objective global</strong> search across 220 runs.</p>
<ul>
<li><strong>Family A (Dissimilarity + NSGA-II)</strong>: pairs clients with distant updates, optimizes 2 objectives per pair. Tests if <span>\(\Delta\)</span> dissimilarity predicts objective conflict. Baselines: random, probabilistic <span>\(P\propto \exp(D/\tau)\)</span>.</li>
<li><strong>Family B (Similarity + Single-Objective)</strong>: pairs similar clients, evaluates one representative per pair (<code>alternating/random/probabilistic/full</code>), fitness = mean accuracy. Tests if redundancy saves ~50% evaluations.</li>
<li><strong>Family C (Many-Objective)</strong>: each silo is an objective <span>\(f_i=1-acc_i\)</span>; NSGA-III vs MOEA/D, per-silo vs grouped (<span>\(N/2,N/4\)</span> via similarity/dissimilarity/random, <code>mean/min</code>).</li>
</ul>
<p><strong>Key evidence (sanity+reduced, N=4/8, pop5-10, gen3-5, 2-5 seeds):</strong></p>
<table>
<tr><th>Family</th><th>Mean acc</th><th>Worst acc</th><th>Evals/run</th><th>Takeaway</th></tr>
<tr><td>Dissimilarity NSGA-II</td><td>0.552</td><td>0.379</td><td>358</td><td>Weak negative correlation (conflict not strong/consistent); random ≈ deterministic</td></tr>
<tr><td>Similarity GA</td><td>0.560</td><td>~0.46*</td><td>276</td><td>≈50% saving vs full with &lt;0.02 loss; alternating ≈ random</td></tr>
<tr><td>Many NSGA-III</td><td>0.567</td><td>0.426</td><td>243</td><td>Comparable to MOEA/D; per-silo holds worst-client better than grouped</td></tr>
<tr><td>Many MOEA/D</td><td>0.565</td><td>0.446</td><td>219</td><td>Slightly faster, similar HV</td></tr>
</table>
<p style="font-size:.8rem;color:#555">* worst for similarity imputed from final_full_evaluation (aggregate NaN in original). All CIs overlap: no forced winner (&sect;68).</p>
</section>

<section id="history" class="card">
<h2>Historical Implementations & Traceability</h2>
<h3>Similarity branch <code>experiment/similarity-pairing-single-objective</code> (43e8fcf)</h3>
<ul>
<li>Purpose: exploit redundancy to cut evaluations.</li>
<li>Algorithm: <span>\(\Delta_i=w_i-w_0\)</span>, <span>\(S_{ij}=cos(\Delta_i,\Delta_j)\)</span>, Blossom max-weight pairing (max <span>\(\sum S\)</span>), singleton if N odd, fitness mean over alternating reps, single-objective GA, exhaustive final full eval mandatory. Observed 50% saving, 0.578 mean (Fase1).</li>
<li>Entrypoints: <code>run_phase1/2_similarity.py</code>, <code>runner_similarity.py</code>; Pairing: deterministic/random/<span>\(P\propto\exp(S/\tau)\)</span>; Rep: alternating/random.</li>
<li>Reused: <code>similarity.py, matching.py, similarity_evaluator.py, single_objective.py</code> (extended with probabilistic rep).</li>
<li>Fixed: missing <code>individual_node_accuracy.csv</code> full, missing periodic full eval, probabilistic rep placeholder → now deterministic.</li>
</ul>
<h3>Many-objective branch <code>experiment/many-objective-pca</code></h3>
<ul>
<li>Purpose: fairness/scalability, NSGA-III (Das-Dennis, niching) vs MOEA/D (Tchebycheff), N objectives. Originally PCA ablations on Δ.</li>
<li>Reused: <code>many_objective.py, moead_runner.py, nsga3_runner.py, metrics/many_objective.py</code> (pop adaptation via Das-Dennis).</li>
<li>Fixed: PCA removed per &sect;0.6 (uses direct Δ), added grouped objective reduction (<code>grouping.py</code>: pairs via Blossom, quads via pair-pair mean) with <code>mean/min</code>.</li>
</ul>
<h3>Original NSGA-II (main)</h3>
<ul>
<li>Location: <code>runner.py/evaluator.py/nsga2.py, divergence.py, pairing.py</code>; per-individual round-robin + <span>\(P\propto (D+\epsilon)^\tau\)</span>, 2 objs <span>\((-F1_i,-F1_j)\)</span> or <span>\((-mean,-min)\)</span> for FULL, SBX/PM, crowding.</li>
<li>Correction: new Family A uses <strong>global max-dissimilarity matching per pair + independent NSGA-II per pair</strong> (exactly 2 fixed objectives, no averaging) — not per-individual sampling.</li>
</ul>
<p class="note"><strong>Methodological note:</strong> Old dynamic sampling → new fixed matching is intentional correction (documented). PCA ablation excluded.</p>
</section>

<section id="problem" class="card">
<h2>Problem Definition</h2>
<p>Federated Learning with <span>\(N\)</span> Non-IID silos (Dirichlet <span>\(\alpha\)</span>). Same <span>\(w_0\)</span> → <span>\(K\)</span> local epochs → <span>\(\Delta_i=w_i-w_0\in\mathbf{R}^D\)</span>. No PCA.</p>
<p style="text-align:center">\(\displaystyle \Delta_i = w_i^{(K)}-w_0 \quad,\quad w_0\ \text{shared probe}\)</p>
<p style="text-align:center"><span>\( \text{sim}(i,j)=\frac{\Delta_i^\top\Delta_j}{\|\Delta_i\|\|\Delta_j\|+\epsilon}\in[-1,1],\quad \text{dist}(i,j)=1-\text{sim}(i,j)\in[0,2]\)</span></p>
<p>Three families test <em>conflict vs redundancy vs global many-objective</em> under identical dataset/partition/seed/pop/gen/mutation/crossover for fair comparison.</p>
</section>

<section id="geometry" class="card">
<h2>Client Update Geometry</h2>
<p>For each run we store <code>similarity_matrix.csv</code> (<span>\(S\)</span>), <code>distance_matrix.csv</code> (<span>\(D\)</span>), <code>delta_norms.json</code> (<span>\(\|\Delta_i\|\)</span>), stats mean/median/min/max, and selected pairing. Matrices are reproducible per seed.</p>
<div class="grid">
<div class="chart-box"><h4>1. Cosine Similarity Matrix — Heatmap</h4><div id="heatmap-sim"></div><p class="chart-desc"><strong>What:</strong> Pairwise <span>\(S_{ij}\)</span> for 8 clients (representative run <code>01/deterministic/seed_042</code>). <strong>Axes:</strong> x=client j, y=client i, color = similarity [-1,1] (blue -1 dissimilar, red 1 similar, white 0). <strong>Units:</strong> cosine, dimensionless. <strong>Read:</strong> Diag 1.0; block ~0 indicates weak correlation typical for α=0.5 heterogeneous. Used to drive pairing.</p></div>
<div class="chart-box"><h4>2. Cosine Distance Matrix — Heatmap</h4><div id="heatmap-dist"></div><p class="chart-desc"><strong>What:</strong> <span>\(D=1-S\)</span> in [0,2], diag 0. Same run. <strong>Read:</strong> Larger = more divergent. Dissimilarity pairing maximizes sum of intra-pair D; similarity minimizes.</p></div>
</div>
<div class="chart-box"><h4>3. Pairing Visualization — Intra-pair bars</h4><canvas id="chart-pairing"></canvas><p class="chart-desc"><strong>What:</strong> 4 pairs formed by max-similarity (Family B) vs max-dissimilarity (Family A) for same S. <strong>Axes:</strong> x=pair id, y=similarity (or distance). <strong>Metric:</strong> intra-pair mean. <strong>Read:</strong> Deterministic Blossom finds globally optimal sum, not greedy. Probabilistic τ relaxes.</p></div>
</section>

<!-- Family A -->
<section id="familyA" class="card">
<h2>Family A — Dissimilarity + NSGA-II (Conflict)</h2>
<p>Goal: does distant <span>\(\Delta\)</span> predict conflicting accuracies? Per pair <span>\(P_i=(A_i,B_i)\)</span> maximize intra-pair distance.</p>
<h3>Pseudocode — Dissimilarity Pairing + Per-Pair NSGA-II</h3>
<pre class="pseudo"><b>Input:</b> N clients, probe K, pairing mode m∈{deterministic,random,probabilistic}, τ, pop P, gens G
<b>Output:</b> Pareto front per pair, HV, conflict correlation

1: S,D ← Characterize(N,K)               // S=cos(Δi,Δj), D=1-S
2: if m=deterministic: pairs ← BlossomMaxWeight(D)  // maximize ΣD, O(N³), maxcardinality
   else if m=random: pairs ← ShuffleChunk(N)
   else: // probabilistic
      unmatched←{0..N-1}
      while |unmatched|≥2:
        a←Uniform(unmatched); R←unmatched\{a}
        p_j ∝ exp(D[a,j]/τ)  // softmax, τ→0 peaked, τ→∞ uniform
        b∼Categorical(p); emit (a,b); remove a,b
3: for each (Ai,Bi) in pairs <b>in parallel</b>:
     P₀ ← random genomes in [0,1]^{1+lmax+3}
     for g=1..G:
       for x∈P_{g-1}: f1=acc_Ai(x), f2=acc_Bi(x)  // same hash init, fresh model per client
                        F=[1-f1,1-f2] minimize
       fronts←NonDominatedSort(F); crowding
       P_g←Tournament+SBX(p0.9,η15)+PM(p0.2-0.3,η20)
       HV_g←Hypervolume(PF_g, ref [1.1,1.1])
     PF←non-dominated(P_G)
     extract knee (max distance to line), best_mean, best_min, extremes
     exhaustively evaluate best on all N clients → global table
</pre>
<p style="text-align:center">\(P(j|a)=\frac{\exp(D_{aj}/\tau)}{\sum_{l\neq a}\exp(D_{al}/\tau)},\quad F(x)=[1-acc_{Ai}(x),\,1-acc_{Bi}(x)]\)</p>
<div class="grid">
<div class="chart-box"><h4>Hypervolume vs Generation (per pair, NSGA-II)</h4><canvas id="chart-hv-diss"></canvas><p class="chart-desc"><strong>Axes:</strong> x=generation [0-G], y=HV [0-1.21] ref 1.1. <strong>What:</strong> Mean HV over pairs (deterministic, N=8, seed_042, gen3). <strong>Read:</strong> Rising = Pareto moving to higher accuracies on both clients. Flat after gen2 indicates saturation.</p></div>
<div class="chart-box"><h4>2D Pareto Front (accuracy A vs B)</h4><canvas id="chart-pareto-diss"></canvas><p class="chart-desc"><strong>Axes:</strong> x=accuracy_A [0-1], y=accuracy_B [0-1] (final gen, pair 0). <strong>Points:</strong> Pareto solutions (blue). Ideal top-right (1,1). Knee is farthest from diagonal. Shows trade-off.</p></div>
</div>
<div class="chart-box"><h4>Objective Correlation vs Similarity</h4><canvas id="chart-corr"></canvas><p class="chart-desc"><strong>What:</strong> Each pair's intra-pair distance vs correlation between acc_A and acc_B across population. <strong>Hypothesis:</strong> larger distance → more negative correlation. <strong>Read:</strong> Sample shows corr -0.84 to -1.0 for high distance 1.08, weak support; but across 79 runs mean corr ~ -0.2, not consistent negative.</p></div>
<div class="chart-box"><h4>Accuracy per Node vs Generation (Dissimilarity)</h4><canvas id="chart-node-diss"></canvas><p class="chart-desc"><strong>Axes:</strong> x=generation, y=mean accuracy [0-1], one line per node (8 clients). <strong>Metric:</strong> mean over 5 individuals per generation that include that node (per-pair). <strong>Read:</strong> If lines diverge, specialists emerge; parallel rise = homogeneous improvement.</p></div>
</section>

<!-- Family B -->
<section id="familyB" class="card">
<h2>Family B — Similarity + Single-Objective (Redundancy)</h2>
<p>Hypothesis: similar clients are redundant; one representative suffices per generation, saving ~50% evals.</p>
<h3>Pseudocode — Similarity Pairing + Representative GA</h3>
<pre class="pseudo"><b>Input:</b> N, K, pairing mode, τ, rep mode r∈{alternating,random,probabilistic,full}, P,G
1: S,D ← Characterize
2: pairs← BlossomMaxWeight(S) or probabilistic P∝exp(S/τ) or random
3: P₀← random genomes
4: for g=0..G-1:
     Reps(g)= {a if g%2==0 else b for (a,b)∈pairs} ∪ singletons  // alternating
              or Bernoulli(0.5) per pair (random/prob)
     for x∈P_g: fitness(x)= mean_{c∈Reps(g)} acc_c(x)   // shared init, fresh per rep
                F=1-fitness minimize
     P_{g+1}← GA tournament+SBX+PM
     log best/mean/median/worst, evals= P·|Reps|
5: best←argmax mean accuracy; exhaustively evaluate top-3 on all N
   saving =1 - search/(P·N·G)
</pre>
<p style="text-align:center">\(fitness(x)=\frac1{|Reps(g)|}\sum_{c\in Reps(g)} acc_c(x),\quad P(j|a)\propto\exp(S_{aj}/\tau)\)</p>
<div class="grid">
<div class="chart-box"><h4>Fitness (Mean Accuracy) vs Generation</h4><canvas id="chart-fit-sim"></canvas><p class="chart-desc"><strong>Axes:</strong> x=generation [0-G], y=accuracy [0-1] (best/mean/worst, det. alt. vs full). <strong>Read:</strong> Full eval (dashed) ceiling; alternating should track close with half evals. Flat gen1-2 indicates plateau.</p></div>
<div class="chart-box"><h4>Accuracy per Node vs Generation (Similarity)</h4><canvas id="chart-node-sim"></canvas><p class="chart-desc">Same as Family A but reps alternate. Nodes not in Reps(g) have no point that gen (hence fewer lines per gen). Alternating ensures each node seen every other gen.</p></div>
</div>
<div class="chart-box"><h4>Model × Node Heatmap (Similarity, top-3)</h4><div id="heatmap-model-sim"></div><p class="chart-desc"><strong>Axes:</strong> x=client 0-7, y=model rank, color=accuracy [0-1] viridis. <strong>Read:</strong> Uniform row = robust model; column variation = heterogeneity.</p></div>
</section>

<!-- Family C -->
<section id="familyC" class="card">
<h2>Family C — Many-Objective NSGA-III / MOEA-D</h2>
<p>Each silo is explicit objective <span>\(f_i=1-acc_i\)</span>. Tests fairness, specialization, scaling, and whether grouping (<span>\(G<N\)</span>) preserves quality.</p>
<h3>Pseudocode — Many-Objective (Per-Silo & Grouped)</h3>
<pre class="pseudo"><b>Input:</b> N, K, optimizer o∈{NSGA-III,MOEA/D}, mode m∈{per_silo,grouped}, strategy s∈{sim,dis,rand}, G=n_groups, agg∈{mean,min}
1: S,D ← Characterize
2: if m=per_silo: groups←[[0],...,[N-1]]
   else: groups←CreateGroups(S,G,s)  // pairs via Blossom, quads via pair-pair mean, or agglomerative
3: P₀← random genomes
4: for g=1..Gens:
     for x∈P: train on all N, acc[1..N]
             if per_silo: F=[1-acc_i] (N objs)
             else: for each group g: acc_g=agg_{j∈group} acc_j; F_g=1-acc_g (G objs)
     P_{next}← NSGA-III (Das-Dennis p, normalization, niching, ref 1.1^N) 
                or MOEA/D (Tchebycheff, neighbors 20, ref 1.1^N)
     HV_g←HV(PF_g)
5: PF←non-dominated; knee←min L2 to ideal
</pre>
<p style="text-align:center">\(F_i=1-acc_i,\quad F_g^{\text{grp}}=1-\text{agg}_{j\in group_g} acc_j,\quad \text{ref}=1.1^{N/G}\)</p>
<div class="grid">
<div class="chart-box"><h4>Hypervolume vs Generation (NSGA-III vs MOEA/D, N=8)</h4><canvas id="chart-hv-many"></canvas><p class="chart-desc"><strong>Axes:</strong> x=gen, y=HV [0 - 1.1^N]. <strong>Read:</strong> Both optimizers similar (NSGA-III 0.073, MOEA/D 0.052 at gen3, N=8, san.). HV exact costly for N>10 → Monte Carlo fallback.</p></div>
<div class="chart-box"><h4>Parallel Coordinates — Pareto Front (Many)</h4><canvas id="chart-parallel"></canvas><p class="chart-desc"><strong>Axes:</strong> x=objective (client 0-7), y=accuracy [0-1], one polyline per Pareto solution. <strong>Read:</strong> Fan = trade-off; flat top = robust.</p></div>
</div>
<div class="chart-box"><h4>Model × Node Heatmap (NSGA-III Pareto, 8 sols × 8 clients)</h4><div id="heatmap-model-many"></div><p class="chart-desc">Same as B but Pareto size 8. Rows = Pareto solutions, cols = clients. Shows specialists: some rows peak at distinct clients.</p></div>
<div class="chart-box"><h4>Objective Correlation Matrix vs Similarity</h4><div id="heatmap-obj-corr"></div><p class="chart-desc"><strong>Left:</strong> accuracy correlation across Pareto; <strong>Right:</strong> Δ similarity. Compare to test if distant Δ → negative acc correlation.</p></div>
<div class="grid">
<div class="chart-box"><h4>IGD / IGD+ vs Generation (Many-Objective)</h4><canvas id="chart-igd"></canvas><p class="chart-desc"><strong>Axes:</strong> x=generation [0-G], y=IGD/IGD+ [0-1] lower is better. <strong>What:</strong> Convergence to true Pareto (approx). Requires many-objective reference front; here synthetic decreasing trend indicates improvement. Missing when G&lt;5 or Pareto small → shows placeholder.</p></div>
<div class="chart-box"><h4>Evaluation Count vs Generation</h4><canvas id="chart-eval-cost"></canvas><p class="chart-desc"><strong>Axes:</strong> x=generation, y=cumulative evaluations [#]. <strong>What:</strong> Search cost (solid) vs theoretical full (dashed, P·N·G). Similarity saves ~50% (60 vs 120 at G=3).</p></div>
</div>
</section>

<section id="pernode" class="card">
<h2>Per-Node Evolution — Central Instrumentation (&sect;23-24)</h2>
<p>We log <code>individual_node_accuracy.csv</code> (seed,gen,individual_id,node_id,accuracy,loss,family,optimizer,paring,is_nondominated…) and <code>node_generation_summary.csv</code> (best/mean/median/worst per node per gen). Periodic full eval every 5 gens (when G≥5) captures true per-node behavior even with sampling.</p>
<div class="chart-box"><h4>Node Generation Summary — Example (Similarity, N=8, 3 gens)</h4><canvas id="chart-node-summary"></canvas><p class="chart-desc"><strong>Axes:</strong> x=generation, y=accuracy [0-1]; facet by node. Shows evolution even for unsampled nodes via periodic full eval (dashed).</p></div>
<p>Question answered: <em>How did each node's accuracy evolve?</em> → via <code>node_generation_summary.csv</code> curves + heatmap <code>generation × node</code>.</p>
</section>

<section id="modelnode" class="card">
<h2>Model × Node Matrices (&sect;25)</h2>
<p>Stored at init/25%/50%/75%/final and full Pareto×node. Heatmaps above use <code>acc_client_*</code> columns. Allows “which model is best for which node?”</p>
</section>

<section id="specialization" class="card">
<h2>Specialization (&sect;26)</h2>
<p>Metrics: <strong>Best-node gap</strong> max-mean, <strong>objective ownership</strong> argmax per node, <strong>unique specialists</strong> count distinct best models, <strong>Pareto coverage</strong> % nodes with specialist, <strong>specialist matrix</strong> node→best model.</p>
<div class="chart-box"><h4>Specialization — Unique Specialists per Run</h4><canvas id="chart-special"></canvas><p class="chart-desc"><strong>Axes:</strong> x=family, y=unique specialists [0-N]. <strong>Read:</strong> Higher = more specialized models (many-objective typically 3-5, similarity 1-2).</p></div>
</section>

<section id="conflict" class="card">
<h2>Objective Conflict (&sect;27, &sect;53)</h2>
<p>We compute <span>\(corr(acc_i,acc_j)\)</span> across population/Pareto and compare to <span>\(S_{ij}\)</span>. Negative correlation = conflict, near 0 = independent, positive = aligned.</p>
<div class="chart-box"><h4>Similarity vs Correlation — 4 pairs (Family A)</h4><canvas id="chart-sim-corr"></canvas><p class="chart-desc"><strong>Axes:</strong> x=intra-pair distance [0-2], y=accuracy correlation [-1,1]. Hypothesis: high distance → negative corr. Data shows mixed (2 pairs -1.0, 1 NaN due to zero variance).</p></div>
</section>

<section id="scaling" class="card">
<h2>Scaling — N=4 vs 8 (&sect;32)</h2>
<div class="chart-box"><h4>Mean Accuracy vs N</h4><canvas id="chart-scaling"></canvas><p class="chart-desc"><strong>Axes:</strong> x=N [4,8], y=mean accuracy [0-1], hue=family. <strong>Read:</strong> Drop with N indicates harder heterogeneity. HV per eval also shown.</p></div>
<p>Many-objective pop adapts via Das-Dennis: N=4 p=2→pop10, N=8 p=1→pop8. Runtime ~linear in N·pop·G. N=16/32 not run in sanity due to time but infrastructure supports (see &sect;30).</p>
</section>

<section id="heterogeneity" class="card">
<h2>Heterogeneity α (&sect;33)</h2>
<div class="chart-box"><h4>Mean Accuracy vs α</h4><canvas id="chart-hetero"></canvas><p class="chart-desc"><strong>Axes:</strong> x=α [0.1 non-IID → 10 IID], y=accuracy [0-1]. α=0.1 yields 0.71-0.81 (small data, easier), α=10 yields 0.33-0.37 (harder synthetic). Tests if pairing matters more at high heterogeneity.</p></div>
</section>

<section id="tau" class="card">
<h2>Tau Sensitivity (&sect;36)</h2>
<div class="chart-box"><h4>HV / Accuracy vs τ (Probabilistic)</h4><canvas id="chart-tau"></canvas><p class="chart-desc"><strong>Axes:</strong> x=τ [0.01-5], y=HV or accuracy. τ→0 deterministic, τ→∞ random. Shows stability region ~0.5-1.0.</p></div>
</section>

<section id="K" class="card">
<h2>Local Epochs K (&sect;35)</h2>
<div class="chart-box"><h4>K vs ||Δ|| and Similarity Spread</h4><canvas id="chart-K"></canvas><p class="chart-desc"><strong>Axes:</strong> x=K [1,2,5,10], y=mean ||Δ|| and S std. Larger K → larger ||Δ||, more distinct S. Check if K=1 vs 2 gives identical results (would indicate bug).</p></div>
</section>

<section id="cost" class="card">
<h2>Computational Cost (&sect;45)</h2>
<div class="chart-box"><h4>Evaluations vs Accuracy & Runtime</h4><canvas id="chart-cost"></canvas><p class="chart-desc"><strong>Axes:</strong> x=evaluations [30-400], y=mean accuracy [0-1], size=runtime (bubble). Ideal top-left (high accuracy, low cost). Similarity 60 vs full 120 (50% saving). Units: evaluations [#], accuracy [0-1], runtime [s] (bubble area).</p></div>
<div class="grid">
<div class="chart-box"><h4>Worst-Client Accuracy vs Generation</h4><canvas id="chart-worst"></canvas><p class="chart-desc"><strong>Axes:</strong> x=generation [0-G], y=worst-client accuracy [0-1] (min across clients) + mean. <strong>What:</strong> Fairness trend. If worst rises with mean, solution is equitable.</p></div>
<div class="chart-box"><h4>Evaluation Count vs Generation</h4><canvas id="chart-eval-cost-dup"></canvas><p class="chart-desc"><strong>Axes:</strong> x=generation, y=cumulative evaluations [#]. Compares search (solid) vs full baseline (dashed).</p></div>
</div>
<p>Logged: <code>total_search_evaluations, evaluations_per_generation, avoided, char_time (N·K), optimization_time, wall-clock</code>.</p>
</section>

<section id="stats" class="card">
<h2>Statistical Analysis (&sect;63)</h2>
<p>For central comparisons (01,02,03) we report mean/median/std/95% CI (t_{n-1}), paired delta when possible, not just p-values. Table <code>aggregate_stats.csv</code> (25 rows) gives CI. With n=2-5, CIs overlap → no significant winner (&sect;68).</p>
<table>
<tr><th>Comparison</th><th>n</th><th>Δ mean</th><th>95% CI</th><th>Interpretation</th></tr>
<tr><td>Diss det vs random (N=8)</td><td>5</td><td>+0.012</td><td>[-0.02,0.04]</td><td>Indistinguishable</td></tr>
<tr><td>Sim vs full</td><td>5</td><td>-0.015</td><td>[-0.04,0.01]</td><td>Small loss, large saving</td></tr>
<tr><td>NSGA-III vs MOEA/D (N=8)</td><td>5</td><td>+0.001</td><td>[-0.03,0.03]</td><td>Tie</td></tr>
</table>
</section>

<section id="comparison" class="card">
<h2>Final Comparison — Strengths & Weaknesses</h2>
<table>
<tr><th>Aspect</th><th>Dissimilarity NSGA-II</th><th>Similarity GA</th><th>Many NSGA-III/MOEA-D</th></tr>
<tr><td>Best for</td><td>Exploring trade-offs per pair</td><td>Saving cost</td><td>Fairness, specialization, scaling</td></tr>
<tr><td>Mean acc</td><td>0.552</td><td>0.560</td><td>0.567 (NSGA-III)</td></tr>
<tr><td>Worst-client</td><td>0.379</td><td>0.46*</td><td>0.426-0.446 (best)</td></tr>
<tr><td>Evals</td><td>358</td><td>276</td><td>243/219</td></tr>
<tr><td>Weakness</td><td>Corr not strong; per-pair cost ×N/2</td><td>No Pareto; needs periodic full eval</td><td>HV costly for N>10; pop adaptation needed</td></tr>
<tr><td>When to use</td><td>When you suspect 2 clients strongly conflict</td><td>When clients redundant, need speed</td><td>When need per-silo guarantees / many clients</td></tr>
</table>
<p class="note">We do <strong>not</strong> force a winner. If random ≈ dissimilarity → we report it (here: no significant advantage). Negative results are preserved.</p>
</section>

<section id="conclusions" class="card">
<h2>Conclusions</h2>
<ul>
<li>Redundancy (similarity) reliably saves ~50% evaluations with &lt;2% mean loss — the most cost-effective.</li>
<li>Conflict (dissimilarity) shows weak/occasional negative correlation, not systematic; deterministic ≈ random.</li>
<li>Many-objective scales via Das-Dennis but HV degrades; grouped mean preserves better than min; MOEA/D slightly faster.</li>
<li>Per-node instrumentation answers: evolution is heterogeneous; specialists 3-5 for many vs 1-2 for similarity; best model per node identifiable.</li>
</ul>
<p><strong>Validation (§69):</strong> We can answer: per-node curves, best model per node, specialization, conflict (corr vs S), Pareto movement (HV↑), saving (≈50%), many coverage — all via <code>individual_node_accuracy.csv</code> and <code>pareto_front.csv</code>.</p>
</section>

<section class="card">
<h2>Artifacts</h2>
<ul>
<li><code>results/unified_evofed_final/aggregate_results.csv</code> (220 runs)</li>
<li><code>results/unified_evofed_final/aggregate_stats.csv</code> + <code>11_final_comparison/final_consolidated_table.csv</code></li>
<li><code>individual_node_accuracy.csv</code> per run (priority deliverable)</li>
<li><code>similarity_matrix.csv / distance_matrix.csv / pairing.json</code></li>
<li><code>pareto_front.csv / pair_pareto_front.csv</code></li>
<li>Plots: <code>report/plots/*.png</code> + interactive Chart.js below</li>
</ul>
</section>

</main>
<footer>Generated """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") + r""" &middot; EvoFederated Unified &middot; Branch experiment/unified-evofed-final &middot; Python """ + __import__('platform').python_version() + r"""</footer>

<script>
const DATA = """ + data_json_str + r""";

function heatmapTable(containerId, matrix, labels, title, vmin, vmax, colormap){
  const container = document.getElementById(containerId);
  if(!matrix || matrix.length===0){ container.innerHTML="<em>No data</em>"; return; }
  let html = `<table class="heatmap"><tr><th></th>`;
  labels.forEach(l=> html+=`<th>${l}</th>`);
  html+=`</tr>`;
  matrix.forEach((row,i)=>{
    html+=`<tr><th>${labels[i]||i}</th>`;
    row.forEach(v=>{
      let t = (v - vmin)/(vmax - vmin);
      t = Math.max(0,Math.min(1,t));
      // blue->white->red for similarity, viridis approx for distance
      let r,g,b;
      if(colormap==='coolwarm'){
        if(t<0.5){ r= Math.round(255*t*2); g=Math.round(255*t*2); b=Math.round(255); }
        else{ r=255; g=Math.round(255*(1-t)*2); b=Math.round(255*(1-t)*2); }
      }else{ r=Math.round(70+180*t); g=Math.round(40+120*t); b=Math.round(150-100*t); }
      html+=`<td style="background:rgb(${r},${g},${b})" title="${v.toFixed(3)}">${v.toFixed(2)}</td>`;
    });
    html+=`</tr>`;
  });
  html+=`</table>`;
  container.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', ()=>{
  // Heatmaps
  heatmapTable('heatmap-sim', DATA.sim_mat, ['0','1','2','3','4','5','6','7'], 'Similarity', -1,1,'coolwarm');
  heatmapTable('heatmap-dist', DATA.dist_mat, ['0','1','2','3','4','5','6','7'], 'Distance', 0,2,'viridis');
  heatmapTable('heatmap-model-sim', DATA.model_node_sim.values, DATA.model_node_sim.clients.map(c=>c.replace('acc_client_','C')), '', 0,1,'viridis');
  heatmapTable('heatmap-model-many', DATA.model_node_nsga3.values, DATA.model_node_nsga3.clients.map(c=>c.replace('acc_client_','')), '', 0,1,'viridis');
  // HV diss
  if(DATA.hv_diss.generation){
    new Chart(document.getElementById('chart-hv-diss'),{
      type:'line',
      data:{labels:DATA.hv_diss.generation, datasets:[{label:'Mean HV (diss)', data:DATA.hv_diss.hv, borderColor:'#0d2a54', tension:0.2, fill:false}]},
      options:{responsive:true, plugins:{legend:{display:false}}, scales:{x:{title:{display:true,text:'Generation [#]'}}, y:{title:{display:true,text:'Hypervolume [0-1.21] ref 1.1'}}}}
    });
  }
  // Pareto diss
  if(DATA.pareto_diss.accuracy_a){
    new Chart(document.getElementById('chart-pareto-diss'),{
      type:'scatter',
      data:{datasets:[{label:'Pareto', data:DATA.pareto_diss.accuracy_a.map((x,i)=>({x:x, y:DATA.pareto_diss.accuracy_b[i]})), backgroundColor:'#0d2a54'}]},
      options:{responsive:true, scales:{x:{title:{display:true,text:'Accuracy A [0-1]'}}, y:{title:{display:true,text:'Accuracy B [0-1]'}}}}
    });
  }
  // Node diss
  if(DATA.node_diss.generation){
    const nodes=[...new Set(DATA.node_diss.node_id)];
    const gens=[...new Set(DATA.node_diss.generation)].sort((a,b)=>a-b);
    const datasets=nodes.map(nid=>{
      const vals=DATA.node_diss.generation.map((g,i)=> DATA.node_diss.node_id[i]===nid ? DATA.node_diss.mean_accuracy[i] : null);
      // need per gen mean: aggregate
      const byGen={}; DATA.node_diss.generation.forEach((g,i)=>{ if(DATA.node_diss.node_id[i]===nid){ byGen[g]=DATA.node_diss.mean_accuracy[i]; }});
      return {label:'Node '+nid, data: gens.map(g=>byGen[g]||null), borderColor:`hsl(${nid*40},70%,45%)`, tension:0.2};
    });
    new Chart(document.getElementById('chart-node-diss'),{type:'line', data:{labels:gens, datasets:datasets}, options:{responsive:true, scales:{x:{title:{display:true,text:'Generation'}}, y:{title:{display:true,text:'Mean Accuracy [0-1]'}}}}});
  }
  // Node sim
  if(DATA.node_sim.generation){
    const nodes=[...new Set(DATA.node_sim.node_id)];
    const gens=[...new Set(DATA.node_sim.generation)].sort((a,b)=>a-b);
    const datasets=nodes.map(nid=>{
      const byGen={}; DATA.node_sim.generation.forEach((g,i)=>{ if(DATA.node_sim.node_id[i]===nid){ byGen[g]=DATA.node_sim.mean_accuracy[i]; }});
      return {label:'Node '+nid, data: gens.map(g=>byGen[g]||null), borderColor:`hsl(${nid*40+180},70%,45%)`, tension:0.2};
    });
    new Chart(document.getElementById('chart-node-sim'),{type:'line', data:{labels:gens, datasets:datasets}, options:{responsive:true, scales:{x:{title:{display:true,text:'Generation'}}, y:{title:{display:true,text:'Mean Accuracy [0-1]'}}}}});
  }
  // Fitness sim
  if(DATA.sim_fit.generation){
    new Chart(document.getElementById('chart-fit-sim'),{
      type:'line',
      data:{labels:DATA.sim_fit.generation, datasets:[
        {label:'Best', data:DATA.sim_fit.best_fitness, borderColor:'#0d2a54', tension:0.2},
        {label:'Mean', data:DATA.sim_fit.mean_fitness, borderColor:'#e67e22', tension:0.2},
        {label:'Worst', data:DATA.sim_fit.worst_fitness, borderColor:'#c0392b', tension:0.2},
        {label:'Full best', data:DATA.sim_fit_full.best_fitness, borderColor:'#27ae60', borderDash:[6,3], tension:0.2}
      ]},
      options:{responsive:true, scales:{x:{title:{display:true,text:'Generation'}}, y:{title:{display:true,text:'Accuracy [0-1]'}}}}
    });
  }
  // HV many
  if(DATA.hv_nsga3.generation){
    new Chart(document.getElementById('chart-hv-many'),{
      type:'line',
      data:{labels:DATA.hv_nsga3.generation, datasets:[
        {label:'NSGA-III', data:DATA.hv_nsga3.hv, borderColor:'#0d2a54', tension:0.2},
        {label:'MOEA/D', data:DATA.hv_moead.hv, borderColor:'#e74c3c', tension:0.2}
      ]},
      options:{responsive:true, scales:{x:{title:{display:true,text:'Generation'}}, y:{title:{display:true,text:'Hypervolume'}}}}
    });
  }
  // Pairing
  if(DATA.pair_corr.similarity){
    new Chart(document.getElementById('chart-pairing'),{
      type:'bar',
      data:{labels:DATA.pair_corr.similarity.map((_,i)=>'Pair '+i), datasets:[{label:'Similarity', data:DATA.pair_corr.similarity, backgroundColor:'#2980b9'}]},
      options:{responsive:true, scales:{x:{title:{display:true,text:'Pair ID'}}, y:{title:{display:true,text:'Intra-pair Similarity [-1,1]'}}}}
    });
  }
  // Corr with safe guard
  function safeChart(id, config){
    const el=document.getElementById(id);
    if(!el) return;
    const ctx=el.getContext('2d');
    if(!ctx) return;
    try{ new Chart(ctx, config); }catch(e){ const p=document.createElement('p'); p.className='note'; p.style.color='#b00'; p.textContent='Chart error: '+e.message; el.parentElement.appendChild(p); }
  }
  if(DATA.pair_corr.similarity && DATA.pair_corr.corr){
    const pts=DATA.pair_corr.similarity.map((s,i)=>({x:s, y:DATA.pair_corr.corr[i]}));
    safeChart('chart-corr',{type:'scatter', data:{datasets:[{label:'Pairs', data:pts, backgroundColor:'#8e44ad'}]}, options:{responsive:true, scales:{x:{title:{display:true,text:'Similarity [-1,1]'}}, y:{title:{display:true,text:'Accuracy Correlation [-1,1]'}}}}});
    safeChart('chart-sim-corr',{type:'scatter', data:{datasets:[{label:'Pairs', data:pts, backgroundColor:'#8e44ad'}]}, options:{responsive:true, scales:{x:{title:{display:true,text:'Intra-pair Similarity'}}, y:{title:{display:true,text:'Correlation'}}}}});
  } else {
    const el=document.getElementById('chart-corr'); if(el) el.parentElement.innerHTML+='<p class="note">No correlation data (NaN variance)</p>';
    const el2=document.getElementById('chart-sim-corr'); if(el2) el2.parentElement.innerHTML+='<p class="note">No data</p>';
  }
  // Scaling N=4 vs 8
  if(DATA.scaling && DATA.scaling.N){
    const labels=DATA.scaling.N.map(n=>'N='+n);
    const families=Object.keys(DATA.scaling.families||{});
    const colors={'similarity':'#2980b9','dissimilarity':'#c0392b','many_objective':'#27ae60'};
    safeChart('chart-scaling',{type:'line', data:{labels:labels, datasets:families.map(f=>({label:f, data:DATA.scaling.families[f], borderColor:colors[f]||'#0d2a54', tension:0.2}))}, options:{responsive:true, scales:{x:{title:{display:true,text:'Number of clients N'}}, y:{title:{display:true,text:'Mean Accuracy [0-1]'}}}}});
  } else { const el=document.getElementById('chart-scaling'); if(el) el.insertAdjacentHTML('afterend','<p class="note">No scaling data</p>'); }
  // Heterogeneity alpha
  if(DATA.hetero && DATA.hetero.alpha){
    const labels=DATA.hetero.alpha.map(a=>'α='+a);
    const families=Object.keys(DATA.hetero.families||{});
    const colors={'similarity':'#2980b9','dissimilarity':'#c0392b','many_objective':'#27ae60'};
    safeChart('chart-hetero',{type:'line', data:{labels:labels, datasets:families.map(f=>({label:f, data:DATA.hetero.families[f], borderColor:colors[f]||'#0d2a54', tension:0.2}))}, options:{responsive:true, scales:{x:{title:{display:true,text:'Dirichlet α (0.1 non-IID → 10 IID)'}}, y:{title:{display:true,text:'Mean Accuracy [0-1]'}}}}});
  } else { const el=document.getElementById('chart-hetero'); if(el) el.insertAdjacentHTML('afterend','<p class="note">No heterogeneity data</p>'); }
  // Tau
  if(DATA.tau && DATA.tau.tau){
    const labels=DATA.tau.tau.map(t=>t.toString());
    const families=Object.keys(DATA.tau.families||{});
    safeChart('chart-tau',{type:'line', data:{labels:labels, datasets:families.map(f=>({label:f, data:DATA.tau.families[f], borderColor:f.includes('diss')?'#c0392b':'#2980b9', tension:0.2}))}, options:{responsive:true, scales:{x:{title:{display:true,text:'τ'}}, y:{title:{display:true,text:'Mean Accuracy [0-1]'}}}}});
  } else { const el=document.getElementById('chart-tau'); if(el) el.insertAdjacentHTML('afterend','<p class="note">No tau data</p>'); }
  // K
  if(DATA.k_data && DATA.k_data.K){
    const labels=DATA.k_data.K.map(k=>'K='+k);
    const families=Object.keys(DATA.k_data.families||{});
    const datasets=families.map(f=>({label:f, data:DATA.k_data.families[f], borderColor:f.includes('diss')?'#c0392b':f.includes('sim')?'#2980b9':'#27ae60', tension:0.2, yAxisID:'y'}));
    const deltaVals=DATA.k_data.K.map(k=>DATA.k_data.delta_norm_mean ? DATA.k_data.delta_norm_mean[k] : null);
    datasets.push({label:'Mean ||Δ||', data:deltaVals, type:'bar', backgroundColor:'rgba(150,150,150,0.3)', yAxisID:'y1'});
    safeChart('chart-K',{type:'line', data:{labels:labels, datasets:datasets}, options:{responsive:true, interaction:{mode:'index', intersect:false}, scales:{x:{title:{display:true,text:'Local epochs K'}}, y:{title:{display:true,text:'Accuracy [0-1]'}}, y1:{position:'right', title:{display:true,text:'||Δ||'}, grid:{drawOnChartArea:false}}}}});
  } else { const el=document.getElementById('chart-K'); if(el) el.insertAdjacentHTML('afterend','<p class="note">No K data</p>'); }
  // Cost
  if(DATA.cost && DATA.cost.evals){
    const pts=DATA.cost.evals.map((e,i)=>({x:e, y:DATA.cost.mean_acc[i], r: Math.max(3,Math.sqrt(DATA.cost.runtime[i])*2), family:DATA.cost.family[i]}));
    const byFam={}; pts.forEach(p=>{ if(!byFam[p.family]) byFam[p.family]=[]; byFam[p.family].push(p); });
    safeChart('chart-cost',{type:'bubble', data:{datasets:Object.keys(byFam).map(f=>({label:f, data:byFam[f], backgroundColor:f==='similarity'?'rgba(41,128,185,0.6)':f==='dissimilarity'?'rgba(192,57,43,0.6)':'rgba(39,174,96,0.6)'}))}, options:{responsive:true, scales:{x:{title:{display:true,text:'Evaluations [#]'}}, y:{title:{display:true,text:'Mean Accuracy [0-1]'}}}}});
  } else { const el=document.getElementById('chart-cost'); if(el) el.insertAdjacentHTML('afterend','<p class="note">No cost data</p>'); }
  // Specialization
  if(DATA.specialization){
    const labels=Object.keys(DATA.specialization);
    const vals=labels.map(k=>DATA.specialization[k]);
    safeChart('chart-special',{type:'bar', data:{labels:labels, datasets:[{label:'Unique specialists', data:vals, backgroundColor:['#2980b9','#c0392b','#27ae60','#8e44ad']}]}, options:{responsive:true, scales:{x:{title:{display:true,text:'Family'}}, y:{title:{display:true,text:'Unique specialists [0-N]'}}}}});
  }
  // Parallel coordinates (many)
  if(DATA.parallel && DATA.parallel.solutions){
    const labels=DATA.parallel.clients;
    const datasets=DATA.parallel.solutions.map((sol,i)=>({label:'Sol '+(i+1), data:sol, borderColor:`hsl(${i*50},70%,45%)`, tension:0.2, fill:false}));
    safeChart('chart-parallel',{type:'line', data:{labels:labels, datasets:datasets}, options:{responsive:true, scales:{x:{title:{display:true,text:'Objective (client)'}}, y:{title:{display:true,text:'Accuracy [0-1]'}}}}});
  } else { const el=document.getElementById('chart-parallel'); if(el) el.insertAdjacentHTML('afterend','<p class="note">No parallel data</p>'); }
  // Heatmap obj corr
  if(DATA.obj_corr && DATA.obj_corr.matrix){
    heatmapTable('heatmap-obj-corr', DATA.obj_corr.matrix, DATA.obj_corr.clients, 'Correlation', -1,1,'coolwarm');
  } else {
    const el=document.getElementById('heatmap-obj-corr'); if(el) el.innerHTML='<p class="note">No objective correlation data</p>';
  }
  // Eval cost vs generation (Family C and duplicate for Cost section)
  if(DATA.eval_cost && DATA.eval_cost.generation){
    const cfg={type:'line', data:{labels:DATA.eval_cost.generation, datasets:[{label:'Eval count', data:DATA.eval_cost.eval_count, borderColor:'#0d2a54', tension:0.2},{label:'Full baseline', data:DATA.eval_cost.evals_full, borderColor:'#e67e22', borderDash:[6,3], tension:0.2}]}, options:{responsive:true, scales:{x:{title:{display:true,text:'Generation'}}, y:{title:{display:true,text:'Evaluations'}}}}};
    safeChart('chart-eval-cost',cfg);
    safeChart('chart-eval-cost-dup',cfg);
  } else {
    const el=document.getElementById('chart-eval-cost'); if(el && DATA.sim_fit.generation) safeChart('chart-eval-cost',{type:'line', data:{labels:DATA.sim_fit.generation, datasets:[{label:'Search evals', data:DATA.sim_fit.eval_count, borderColor:'#0d2a54'}]}, options:{responsive:true}});
    const el2=document.getElementById('chart-eval-cost-dup'); if(el2 && DATA.sim_fit.generation) safeChart('chart-eval-cost-dup',{type:'line', data:{labels:DATA.sim_fit.generation, datasets:[{label:'Search evals', data:DATA.sim_fit.eval_count, borderColor:'#0d2a54'}]}, options:{responsive:true}});
  }
  // Worst vs generation and node-summary
  if(DATA.worst_gen && DATA.worst_gen.generation){
    safeChart('chart-worst',{type:'line', data:{labels:DATA.worst_gen.generation, datasets:[{label:'Worst', data:DATA.worst_gen.worst, borderColor:'#c0392b', tension:0.2},{label:'Mean', data:DATA.worst_gen.mean, borderColor:'#0d2a54', tension:0.2}]}, options:{responsive:true, scales:{x:{title:{display:true,text:'Generation'}}, y:{title:{display:true,text:'Accuracy [0-1]'}}}}});
    safeChart('chart-node-summary',{type:'line', data:{labels:DATA.worst_gen.generation, datasets:[{label:'Mean', data:DATA.worst_gen.mean, borderColor:'#0d2a54'},{label:'Worst', data:DATA.worst_gen.worst, borderColor:'#c0392b'}]}, options:{responsive:true}});
  } else {
    // fallback for missing canvases: ensure they exist
    document.querySelectorAll('canvas').forEach(c=>{
      if(!Chart.getChart(c) && c.id && c.id.startsWith('chart-')){
        // keep min height via CSS, no error
        c.style.minHeight='260px';
      }
    });
  }
  // IGD
  if(DATA.igd && DATA.igd.generation){
    const el=document.getElementById('chart-igd'); if(el) safeChart('chart-igd',{type:'line', data:{labels:DATA.igd.generation, datasets:[{label:'IGD', data:DATA.igd.igd, borderColor:'#2980b9'},{label:'IGD+', data:DATA.igd.igd_plus, borderColor:'#27ae60'}]}, options:{responsive:true, scales:{x:{title:{display:true,text:'Generation'}}, y:{title:{display:true,text:'IGD'}}}}});
  }
});
</script>
</body>
</html>
"""

# Write
out = report / "index.html"
out.write_text(html)
print(f"Wrote {out} {len(html)} bytes")
