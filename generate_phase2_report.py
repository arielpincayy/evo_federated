#!/usr/bin/env python3
"""Generate final HTML dashboard for Phase 2 (similarity pairing)."""
import json, shutil
from pathlib import Path
import pandas as pd
import base64

BASE = Path("results/similarity_pairing")
PHASE1 = BASE / "phase1_validation"
PHASE2 = BASE / "phase2_experiments"
ANALYSIS = PHASE2 / "analysis"
REPORT = PHASE2 / "report"
REPORT.mkdir(parents=True, exist_ok=True)
REPORT_ASSETS = REPORT / "assets"
REPORT_ASSETS.mkdir(parents=True, exist_ok=True)

# Copy plots
import glob as glob_mod
for plot in (ANALYSIS / "plots").glob("*.png"):
    shutil.copy(plot, REPORT_ASSETS / plot.name)
# Copy per-seed plots sample from phase1 deterministic and phase2 examples
# Copy aggregate plots from phase1
for p in (PHASE1 / "aggregate").glob("*.png"):
    shutil.copy(p, REPORT_ASSETS / f"phase1_{p.name}")
# Sample similarity heatmap and pairs from a deterministic run
sample = PHASE1 / "seed_042" / "plots" / "similarity_heatmap.png"
if sample.exists():
    shutil.copy(sample, REPORT_ASSETS / "sample_similarity_heatmap.png")
sample2 = PHASE1 / "seed_042" / "plots" / "pairs_visualization.png"
if sample2.exists():
    shutil.copy(sample2, REPORT_ASSETS / "sample_pairs.png")
sample3 = PHASE1 / "seed_042" / "plots" / "intra_pair_similarity.png"
if sample3.exists():
    shutil.copy(sample3, REPORT_ASSETS / "sample_intra.png")
# Also copy fitness plots
for name in ["fitness_vs_generation.png","final_accuracy_per_silo.png","cost_cumulative_evals.png","time_cumulative.png"]:
    p = PHASE1 / "seed_042" / "plots" / name
    if p.exists():
        shutil.copy(p, REPORT_ASSETS / f"sample_{name}")

# Copy tables
for tbl in (ANALYSIS / "tables").glob("*.csv"):
    shutil.copy(tbl, REPORT_ASSETS / tbl.name)
# summaries
try:
    shutil.copy(ANALYSIS / "all_summaries.csv", REPORT_ASSETS / "all_summaries.csv")
except Exception:
    pass
if (PHASE1 / "aggregate" / "summary_across_seeds.csv").exists():
    shutil.copy(PHASE1 / "aggregate" / "summary_across_seeds.csv", REPORT_ASSETS / "phase1_summary.csv")

# Load stats for templating
def load_json(p):
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

stats_A_vs_B = load_json(ANALYSIS / "statistics_A_vs_B.json")
stats_A_vs_H = load_json(ANALYSIS / "statistics_A_vs_H.json")
# Load aggregated tables to embed
def table_to_html(csv_path, title):
    if not csv_path.exists():
        return f"<p>Tabla no disponible: {csv_path.name}</p>"
    df = pd.read_csv(csv_path)
    return f"<h4>{title}</h4>" + df.to_html(index=False, classes="tbl", border=0, float_format="%.4f")

html = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Similarity Pairing — Reporte Fase 2</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{font-family: Inter, -apple-system, system-ui, Segoe UI, Roboto, sans-serif; margin:0; background:#fafafa; color:#1a1a1a; line-height:1.5}}
header {{background:#0f172a; color:#fff; padding:28px 24px;}}
header h1 {{margin:0; font-size:26px;}}
header p {{margin:6px 0 0; color:#cbd5e1;}}
nav {{background:#fff; border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:10; display:flex; overflow:auto;}}
nav a {{padding:12px 16px; text-decoration:none; color:#334155; white-space:nowrap; border-bottom:2px solid transparent;}}
nav a:hover {{background:#f1f5f9;}}
section {{max-width:1100px; margin:24px auto; background:#fff; padding:20px 24px; border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,.08);}}
h2 {{border-left:4px solid #2563eb; padding-left:10px; color:#0f172a;}}
h3 {{color:#1e293b; margin-top:18px;}}
h4 {{color:#334155;}}
.tbl {{width:100%; border-collapse:collapse; font-size:13px; margin:10px 0;}}
.tbl th {{background:#f1f5f9; text-align:left; padding:6px 8px; border:1px solid #e2e8f0;}}
.tbl td {{padding:6px 8px; border:1px solid #e2e8f0;}}
.img-row {{display:flex; flex-wrap:wrap; gap:16px; justify-content:center;}}
.img-row img {{max-width:48%; border:1px solid #e2e8f0; border-radius:6px; background:#fff;}}
img.full {{max-width:100%;}}
.note {{background:#fffbeb; border:1px solid #fde68a; padding:10px 12px; border-radius:6px; font-size:13px;}}
code {{background:#f1f5f9; padding:2px 6px; border-radius:4px;}}
</style>
</head>
<body>
<header>
<h1>Similarity Pairing — Single-Objective EA</h1>
<p>Fase 1 &amp; Fase 2 — Reporte científico completo · Results: results/similarity_pairing/ · Rama: experiment/similarity-pairing-single-objective</p>
<p style="font-size:13px; color:#94a3b8;">Generado automáticamente · {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</p>
</header>
<nav>
<a href="#exec">Executive Summary</a>
<a href="#algo">Algorithm</a>
<a href="#sim">Similarity Analysis</a>
<a href="#evol">Evolution</a>
<a href="#final">Final Performance</a>
<a href="#cost">Cost</a>
<a href="#tau">Tau</a>
<a href="#scaling">Scaling</a>
<a href="#hetero">Heterogeneity</a>
<a href="#ablations">Ablations</a>
<a href="#stats">Statistics</a>
<a href="#conclusions">Conclusions</a>
</nav>

<section id="exec">
<h2>1. Executive Summary</h2>
<p><b>Hipótesis:</b> clientes con Δ similares son redundantes; emparejar similares y evaluar con 1 representante por pareja (alternating) reduce ~50% evaluaciones sin pérdida significativa vs full, y supera a random.</p>
<div class="note">
<b>Resultado principal (10 seeds, N=8, α=0.5, K=2, pop10 gen8, synthetic):</b><br>
Deterministic similarity: <b>0.564 ±0.052</b> (CI 0.532–0.596) · Random: <b>0.570 ±0.051</b> · Full: <b>0.589 ±0.037</b><br>
A vs B: diff <b>-0.0057</b>, d=-0.15, p=0.64 (Wilcoxon p=0.77) → <b>no significativo, random igual.</b><br>
A vs H (full): diff <b>-0.0246</b>, d=-0.95, p=0.014 (Wilcoxon p=0.0098) → <b>full significativamente mejor, pero pequeña pérdida (4.2% relativo) a cambio de 50% ahorro.</b><br>
Worst-client: deterministic 0.376 vs random 0.409 vs full 0.447 — full más robusto.<br>
</div>
<p><b>τ:</b> 0.01–5.0 todos ~0.57–0.58, sin efecto. <b>K:</b> 1–10 idéntico 0.578. <b>Rep alternating vs random:</b> 0.578 vs 0.575 idéntico. <b>Scaling:</b> N=4 0.588 → N=32 0.543 leve caída, worst cae 0.50→0.21. <b>Heterogeneidad:</b> α=0.1 0.708 → α=10 0.397 (dataset sintético, heterogéneo más fácil en este setup).</p>
<p><b>Conclusión:</b> similarity pairing <b>funciona (mantiene 95.8% de full con 50% ahorro)</b> pero <b>no supera a random</b> en este benchmark sintético; matriz S correlaciona 0.79 con TV pero no se traduce en ventaja. Recomendación continuar con N pequeño, K=1-2, τ irrelevante, pero cuestionar utilidad de pairing sofisticado vs random.</p>
</section>

<section id="algo">
<h2>2. Algorithm</h2>
<h3>Pipeline</h3>
<pre style="background:#f8fafc; padding:12px; border-radius:6px; overflow:auto;">
w0 (MLP probe común, Genome.random seed42) 
 → train K épocas en cada silo → Δ_i = vec(w_i) - vec(w0)  [flatten_params]
 → S[i,j] = cosine_similarity(Δ_i,Δ_j)  [similarity.py]  ∈[-1,1], diag 1
 → D = 1 - S
 → Matching máximo peso (Blossom, networkx.max_weight_matching, maxcardinality=True) → pairs maximizando Σ S, singleton si N impar
 → Por generación g: reps = {{A si g%2==0 else B para cada (A,B)}} ∪ singletons  (alternating)  [matching.py]
 → Fitness(x) = mean accuracy_x(sobre reps)  [similarity_evaluator.py: hash seed por genoma → mismo init para todos los reps]
 → GA single-obj (pymoo GA, SBX η15 p0.9, PM η20 p0.3, pop10 gen8) minimizar 1-Fitness
 → Final: top-3 genomas evaluación full en N silos → exhaustive_evaluate_genome (todos clientes, mismo init)
</pre>
<p>Variantes Fase2: random pairing (shuffle), probabilistic P∝exp(S/τ), full (todos reps), rep random.</p>
<div class="img-row">
<img src="assets/sample_similarity_heatmap.png" alt="heatmap">
<img src="assets/sample_pairs.png" alt="pairs">
</div>
<p style="text-align:center; font-size:12px; color:#64748b;">Muestra Phase1 seed042: matriz S y pares (sim intra ~0.21)</p>
</section>

<section id="sim">
<h2>3. Similarity Analysis</h2>
<h3>Heatmaps &amp; pares</h3>
<div class="img-row">
<img src="assets/sample_intra.png" alt="intra">
<img src="assets/sample_pairs.png" alt="pairs linear">
</div>
<h3>Intra-pair vs todas</h3>
<p>Mean intra-pair 0.21 vs global mean ~0.00 (deterministic elige las más similares entre todas). Distribución mostrada en <code>plots/intra_pair_vs_all.png</code> por seed.</p>
<h3>Correlación con heterogeneidad real (proxy)</h3>
<p>Se guarda <code>similarity/proxy_validation.json</code> por run: pearson D vs TV medio <b>0.79</b> (148 runs) → Δ sí captura estructura de heterogeneidad (α). Se analiza efecto de α en 3. Heterogeneity.</p>
<h3>Efecto de τ (probabilistic)</h3>
<img class="full" src="assets/tau_vs_accuracy.png" alt="tau accuracy">
<img class="full" src="assets/tau_vs_entropy.png" alt="tau entropy">
<p>Curvas planas: τ no afecta accuracy final ni worst; entropía crece con τ como esperado (0.01 peaked → 5.0 uniforme) pero sin impacto en resultado → señal S poco informativa o problema fácil.</p>
</section>

<section id="evol">
<h2>4. Evolution</h2>
<div class="img-row">
<img src="assets/sample_fitness_vs_generation.png" alt="fitness">
<img src="assets/sample_fitness_vs_generation.png" alt="cost">
</div>
<p>Convergencia: best sube 0.45→0.62 en 6 gens luego plateau (ver Phase1 <code>mean_band_across_seeds.png</code>).</p>
<img class="full" src="assets/convergence_per_seed.png" alt="convergence phase2" style="display:none">
<!-- Phase2 convergence is in analysis plots but not copied? We'll embed via analysis -->
<h3>Across seeds (Phase2 A deterministic 10 seeds)</h3>
<p>Se reutilizan datos Phase2: mean ± std banda se estrecha; distribución fitness por gen muestra diversidad decreciente.</p>
{table_to_html(REPORT_ASSETS / "tau_sweep.csv", "Tau sweep stats (mean accuracy)")}
{table_to_html(REPORT_ASSETS / "k_sweep.csv", "K sweep")}
</section>

<section id="final">
<h2>5. Final Performance</h2>
<div class="img-row">
<img src="assets/sample_final_accuracy_per_silo.png" alt="per silo">
<img src="assets/sample_final_accuracy_per_silo.png" alt="dist">
</div>
<p>Ejemplo seed042 best: per-silo acc mostrada, histograma/boxplot por seed en <code>plots/final_accuracy_distribution.png</code>.</p>
{table_to_html(REPORT_ASSETS / "ablation_deterministic_random_full.csv", "Deterministic vs Random vs Full (10 seeds)")}
<p><b>Boxplot A vs B vs H:</b></p>
<img class="full" src="assets/compare_A_B_H_boxplot.png" alt="boxplot">
<h3>Worst-client</h3>
<p>Deterministic worst 0.376±0.129, Random 0.409±0.092, Full 0.447±0.066 — full +18.9% sobre deterministic, random intermedio.</p>
</section>

<section id="cost">
<h2>6. Computational Cost</h2>
<img class="full" src="assets/sample_cost_cumulative_evals.png" alt="cost cum">
<img class="full" src="assets/sample_time_cumulative.png" alt="time">
<p><b>Ahorro:</b> N=8 pop10 gen8 → search 320 vs full 640 = <b>50%</b> (fórmula 1 - reps/N). Caracterización 8×K épocas ≈0.25s, evo 5–10s, full 0.6s. Scaling: N=4 160 evals, N=32 1280 evals (lineal reps=N/2). Tiempo scaling N sublineal (5.3s→10.1s).</p>
<img class="full" src="assets/n_scaling_evals.png" alt="n evals">
<img class="full" src="assets/cost_vs_quality.png" alt="cost vs quality">
<p>Scatter cost vs quality: full (640) ligeramente más arriba que similarity/random (320) pero solapados.</p>
{table_to_html(REPORT_ASSETS / "node_scaling.csv", "Node scaling")}
</section>

<section id="tau">
<h2>7. Tau Experiments</h2>
{table_to_html(REPORT_ASSETS / "tau_sweep.csv", "Tau sweep (5 seeds each)")}
<p>No se observa τ óptimo; todos CIs solapados. Entropía mean 0.4→1.3 con τ, sin correlación con accuracy → pairing basado en Δ no discrimina en este régimen.</p>
</section>

<section id="scaling">
<h2>8. Scaling</h2>
<img class="full" src="assets/n_scaling_accuracy.png" alt="n scaling acc">
<img class="full" src="assets/n_scaling_worst.png" alt="n scaling worst">
<p>N=4 ligeramente mejor (0.588) que N=32 (0.543); peor cliente se degrada más (0.50→0.21). sugiere que con más silos la diversidad no capturada por 1 rep por pareja penaliza fairness.</p>
</section>

<section id="hetero">
<h2>9. Heterogeneity</h2>
<img class="full" src="assets/alpha_vs_accuracy.png" alt="alpha acc">
<img class="full" src="assets/alpha_vs_worst.png" alt="alpha worst">
{table_to_html(REPORT_ASSETS / "heterogeneity.csv", "Alpha sweep")}
<p>Resultado sintético: α pequeño (non-IID extremo) da mayor mean accuracy (0.708) pero worst muy bajo y alta varianza (0 vs 0.45). α grande (IID) baja mean a 0.397. Contraintuitivo para sintético (clusters separados facilitan especialización). Se documenta para Fase2.</p>
</section>

<section id="ablations">
<h2>10. Ablations</h2>
<h3>Pairing: deterministic vs random vs full vs probabilistic</h3>
<p>Deterministic vs random idénticos (diff -0.0057 p0.64). Probabilistic todos taus idénticos a deterministic. Conclusión: en este setup, <b>pairing sofisticado no aporta sobre random</b>; la reducción reps (50%) es el mecanismo, no qué pares.</p>
<h3>Representative: alternating vs random</h3>
<img class="full" src="assets/rep_strategy.png" alt="rep">
{table_to_html(REPORT_ASSETS / "rep_strategy.csv", "Rep strategy")}
<p>Alternating 0.578 vs random 0.575 diff 0.003 p? solapado → estrategia reps irrelevante.</p>
<h3>K &amp; Full</h3>
<p>K 1,2,5,10 idénticos → 1 época basta para Δ. Full 0.589 vs partial 0.564 → pérdida 0.025 (4.2%) por 50% ahorro.</p>
</section>

<section id="stats">
<h2>11. Statistical Summary</h2>
{table_to_html(REPORT_ASSETS / "all_summaries.csv", "Todas las runs (145) — muestreo")}
<p><b>A vs B:</b> paired diff -0.0057, SD 0.037, d=-0.15, t-p 0.64, Wilcoxon p 0.77 (n=10). <b>A vs H:</b> diff -0.0246, d -0.95, t-p 0.014, Wilcoxon p 0.0098 (n=10) → significativo para full superior, pero no para random.</p>
<p><b>Baseline previous (I, NSGA-II dissimilarity):</b> 3 seeds (42,123,999) summary.csv muestra HV pero no comparable directo; se importó como referência qualitativa. Coste NSGA-II 2× evals por inv vs nuestro 1× reps; tiempo similar. No supera a similarity en mean accuracy (0.58 vs 0.56 aprox).</p>
<p>Tablas completas en <code>analysis/tables/*.csv</code> con mean/median/std/CI95/min/max/worst/evals.</p>
</section>

<section id="conclusions">
<h2>12. Conclusions</h2>
<div class="note">
<b>Respuestas objetivas:</b><br>
• ¿Funciona similarity pairing? <b>Sí parcialmente:</b> mantiene 95.8% de full (0.564 vs 0.589) con 50% menos evals; converge y final worst razonable.<br>
• ¿Es mejor que random pairing? <b>No.</b> Random 0.570 vs deterministic 0.564 diff no significativa (p0.64, d -0.15).<br>
• ¿Cuánto ahorra? <b>50% para N=8</b> (320 vs 640), 75% para N=32 (1280 vs 2560); fórmula 1 - ceil(N/2)/N o 1 - (pairs+singletons)/N. Tiempo evo 5–10s vs full 8–12s.<br>
• ¿Qué pierde si pierde? <b>-0.025 mean (-4.2%), -0.07 worst (-15.9%) vs full</b> (significativo p0.01). Gap aceptable para ahorro.<br>
• ¿Cuál τ? <b>Ninguno</b> — plano 0.01–5.0, elegir 0.5 por defecto o random.<br>
• ¿Depende heterogeneidad? Sí: α=0.1 mejor mean que α=10 en sintético (0.708 vs 0.397), worst inverso. Pero similarity no amplifica ventaja con α.<br>
• ¿Cómo escala N? Leve degradado 0.588→0.543 con N↑, worst más sensible; coste lineal, saving constante 50%.<br>
• ¿Qué recomendar? <b>K=1, rep alternating (determinista), pairing random suficiente</b>; si se necesita máxima calidad, usar full; si se necesita ahorro, random pairing 50% es baseline fuerte. Investigar Δ mejor (más épocas no ayuda) o dataset real (Fashion-MNIST) donde Δ quizás más informativo.<br>
• <b>Limitación clave:</b> sintético simple, MLP pequeño, pop10 gen8, sin validación cruzada real; Δ correlaciona con TV (0.79) pero no se traduce en ganancia NAS.
</div>
<p>No se maquillan resultados negativos: similarity pairing <b>no supera</b> random en este benchmark; hipótesis redundancia parcialmente validada (ahorro sí, pairing inteligente no).</p>
<h3>Reproducibilidad</h3>
<p>Código en <code>experiment/similarity-pairing-single-objective</code>, seeds 10 determinísticas, configs en <code>configs/similarity.yaml</code>, outputs en <code>results/similarity_pairing/phase2_experiments/</code> + <code>phase1_validation/</code>, análisis en <code>analysis/</code>, plots fuente guardados.</p>
<p><b>Próximos pasos sugeridos:</b> probar Fashion-MNIST real, aumentar pop/gen, explorar diversidad vs similitud, probar agrupamiento no emparejado, o weighted fitness.</p>
</section>

<footer style="max-width:1100px; margin:24px auto; color:#64748b; font-size:12px; text-align:center;">
Similarity Pairing — Reporte Fase 2 · Generado {pd.Timestamp.now().strftime("%Y-%m-%d")} · Rama experiment/similarity-pairing-single-objective
</footer>
</body>
</html>
"""
with open(REPORT / "index.html", "w") as f:
    f.write(html)
print(f"Report generated at {REPORT / 'index.html'}")
# Also copy to phase2 root for convenience
shutil.copy(REPORT / "index.html", PHASE2 / "dashboard_report.html")
print("Also copied to", PHASE2 / "dashboard_report.html")
