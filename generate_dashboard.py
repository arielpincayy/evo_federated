#!/usr/bin/env python3
"""Genera dashboard HTML autocontenido simple."""
import json, pandas as pd, base64, pathlib, shutil
from pathlib import Path

ROOT = Path("results/experimental_campaign")
DASH = ROOT / "dashboard"
DASH.mkdir(parents=True, exist_ok=True)
ASSETS = DASH / "assets"
ASSETS.mkdir(exist_ok=True)

# Load data
df_index = pd.read_csv(ROOT / "experiment_index.csv")
df_agg = pd.read_csv(ROOT / "aggregate_results.csv")
with open(ROOT / "analysis.json") as f:
    analysis = json.load(f)

# Helper to get image as base64 or copy
def copy_plot(src_rel, dst_name):
    src = ROOT / src_rel
    if not src.exists():
        return None
    dst = ASSETS / dst_name
    shutil.copy(src, dst)
    return f"assets/{dst_name}"

# Select key plots to copy
plots_to_copy = [
    ("strategy_comparison/synthetic_pop20_gen10/plots/hv_per_generation.png", "synthetic_hv_gen.png"),
    ("strategy_comparison/synthetic_pop20_gen10/plots/hv_vs_evals.png", "synthetic_hv_evals.png"),
    ("strategy_comparison/synthetic_pop20_gen10/plots/hv_vs_time.png", "synthetic_hv_time.png"),
    ("strategy_comparison/synthetic_pop20_gen10/plots/exhaustive_mean_vs_min.png", "synthetic_exhaustive.png"),
    ("strategy_comparison/synthetic_pop20_gen10/plots/divergence_matrix.png", "synthetic_div.png"),
    ("strategy_comparison/synthetic_pop20_gen10/plots/proxy_vs_true.png", "synthetic_proxy.png"),
    ("strategy_comparison/synthetic_pop20_gen10/plots/checkpoint_hv_vs_gen.png", "synthetic_checkpoint_hv.png"),
    ("strategy_comparison/synthetic_pop20_gen10/plots/checkpoint_mean_f1_vs_gen.png", "synthetic_checkpoint_mean.png"),
    ("strategy_comparison/synthetic_pop20_gen10/plots/hall_of_fame_heatmap.png", "synthetic_hof.png"),
    ("strategy_comparison/synthetic_pop20_gen10/plots/heatmap_arch_client.png", "synthetic_arch_client.png"),
    ("strategy_comparison/fashion_mnist_pop10_gen5/plots/hv_per_generation.png", "fmnist_hv_gen.png"),
    ("strategy_comparison/fashion_mnist_pop10_gen5/plots/hv_vs_evals.png", "fmnist_hv_evals.png"),
    ("strategy_comparison/fashion_mnist_pop10_gen5/plots/exhaustive_mean_vs_min.png", "fmnist_exhaustive.png"),
    ("strategy_comparison/fashion_mnist_pop10_gen5/plots/divergence_matrix.png", "fmnist_div.png"),
    ("strategy_comparison/fashion_mnist_pop10_gen5/plots/checkpoint_hv_vs_gen.png", "fmnist_checkpoint_hv.png"),
    ("strategy_comparison/fashion_mnist_pop10_gen5/plots/hall_of_fame_heatmap.png", "fmnist_hof.png"),
    ("alpha/alpha_10/plots/class_dist_per_client.png", "alpha10_class.png"),
    ("alpha/alpha_0_05/plots/class_dist_per_client.png", "alpha005_class.png"),
    ("alpha/alpha_10/plots/divergence_matrix.png", "alpha10_div.png"),
    ("alpha/alpha_0_05/plots/divergence_matrix.png", "alpha005_div.png"),
    ("population/pop_10/plots/hv_per_generation.png", "pop_hv.png"),
    ("population/pop_40/plots/hv_per_generation.png", "pop40_hv.png"),
    ("generations/gen_5/plots/hv_per_generation.png", "gen5_hv.png"),
    ("generations/gen_20/plots/hv_per_generation.png", "gen20_hv.png"),
    ("tau/tau_0_5/plots/hv_per_generation.png", "tau05_hv.png"),
    ("tau/tau_10_0/plots/hv_per_generation.png", "tau10_hv.png"),
]

# also need pareto plots
pareto_plots = [
    ("strategy_comparison/synthetic_pop20_gen10/plots/pareto_dynamic.png", "pareto_synth_dynamic.png"),
    ("strategy_comparison/synthetic_pop20_gen10/plots/pareto_random.png", "pareto_synth_random.png"),
    ("strategy_comparison/synthetic_pop20_gen10/plots/pareto_full.png", "pareto_synth_full.png"),
    ("strategy_comparison/fashion_mnist_pop10_gen5/plots/pareto_dynamic.png", "pareto_fmnist_dynamic.png"),
]

for src, dst in plots_to_copy+pareto_plots:
    copy_plot(src, dst)

# Build HTML
# Compute summary numbers
meta = analysis["meta"]
total_exp = meta["total_experiments"]
total_runs = meta["total_strategy_runs"]
total_evals = meta["total_evals"]
total_time_min = meta["total_time_min"]
seeds = meta["seeds_used"]
strategies = meta["strategies"]

# Strategy comparison table data
synth = analysis["strategy_comparison"]["synthetic_pop20_gen10"]
fmnist = analysis["strategy_comparison"]["fashion_mnist_pop10_gen5"]

def row_for(comp, strat):
    d=comp.get(strat, {})
    return f"<tr><td>{strat}</td><td>{d.get('hv_common',0):.3f}</td><td>{d.get('mean_f1',0):.3f}</td><td>{d.get('worst_f1',0):.3f}</td><td>{d.get('eval_count',0)}</td><td>{d.get('saving',0):.1f}%</td><td>{d.get('hv_per_eval',0):.5f}</td></tr>"

# Alpha table
# Build dataframe for alpha
alpha_df = df_agg[df_agg["experiment"].str.startswith("alpha/")].copy()
alpha_table_html = ""
if not alpha_df.empty:
    # pivot hv
    piv = alpha_df.pivot_table(index="alpha", columns="strategy", values="hv_common", aggfunc="mean").sort_index()
    alpha_table_html = piv.to_html(float_format=lambda x: f"{x:.3f}", classes="table", border=0)

# Pop table
pop_df = df_agg[df_agg["experiment"].str.startswith("population/")]
pop_table_html = ""
if not pop_df.empty:
    piv = pop_df.pivot_table(index="pop_size", columns="strategy", values="hv_common", aggfunc="mean").sort_index()
    pop_table_html = piv.to_html(float_format=lambda x: f"{x:.3f}", classes="table", border=0)

# Gen table
gen_df = df_agg[df_agg["experiment"].str.startswith("generations/")]
gen_table_html = ""
if not gen_df.empty:
    piv = gen_df.pivot_table(index="n_generations", columns="strategy", values="hv_common", aggfunc="mean").sort_index()
    gen_table_html = piv.to_html(float_format=lambda x: f"{x:.3f}", classes="table", border=0)

# k table
k_df = df_agg[df_agg["experiment"].str.startswith("characterization_k/")]
k_table_html = ""
if not k_df.empty:
    piv = k_df.pivot_table(index="k_epochs", columns="strategy", values="hv_common", aggfunc="mean").sort_index()
    k_table_html = piv.to_html(float_format=lambda x: f"{x:.3f}", classes="table", border=0)

# tau table
tau_df = df_agg[df_agg["experiment"].str.startswith("tau/")]
tau_table_html = ""
if not tau_df.empty:
    piv = tau_df.pivot_table(index="tau", columns="strategy", values="hv_common", aggfunc="mean").sort_index()
    tau_table_html = piv.to_html(float_format=lambda x: f"{x:.3f}", classes="table", border=0)

# clients table
clients_df = df_agg[df_agg["experiment"].str.startswith("clients/")]
clients_table_html = ""
if not clients_df.empty:
    piv = clients_df.pivot_table(index="n_clients", columns="strategy", values="hv_common", aggfunc="mean").sort_index()
    clients_table_html = piv.to_html(float_format=lambda x: f"{x:.3f}", classes="table", border=0)

# Multi-seed stats
ms_baseline = analysis["multi_seed"]["baseline_alpha0_5"]
ms_alpha05 = analysis["multi_seed"]["alpha0_05"]

def ms_table(ms):
    rows=""
    for strat, stats in ms.items():
        rows+=f"<tr><td>{strat}</td><td>{stats['mean']:.3f}</td><td>{stats['std']:.3f}</td><td>{stats['n']}</td></tr>"
    return rows

# Proxy
proxy = analysis["proxy"]

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>EvoFederated — Dashboard Campaña Experimental</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin:0; padding:0; background:#f8f9fa; color:#212529; line-height:1.5}}
header{{background:#0d2a54; color:white; padding:1.5rem 2rem}}
header h1{{margin:0; font-size:1.8rem}}
header p{{margin:0.3rem 0 0; opacity:0.9}}
.container{{max-width:1200px; margin:0 auto; padding:1.2rem}}
.card{{background:white; border-radius:8px; padding:1.2rem; margin:1rem 0; box-shadow:0 1px 3px rgba(0,0,0,0.1)}}
.card h2{{margin-top:0; color:#0d2a54; border-bottom:2px solid #e9ecef; padding-bottom:0.4rem}}
.card h3{{color:#1a4d8f; margin:0.8rem 0 0.4rem}}
.grid2{{display:grid; grid-template-columns:1fr 1fr; gap:1rem}}
.grid3{{display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem}}
.table{{width:100%; border-collapse:collapse; font-size:0.9rem}}
.table th,.table td{{border:1px solid #dee2e6; padding:0.4rem 0.6rem; text-align:center}}
.table th{{background:#e9ecef}}
.badge{{display:inline-block; background:#0d2a54; color:white; padding:0.15rem 0.45rem; border-radius:12px; font-size:0.8rem; margin:0.1rem}}
.kpi{{display:grid; grid-template-columns:repeat(auto-fit, minmax(180px,1fr)); gap:0.8rem; margin:0.8rem 0}}
.kpi div{{background:#f1f3f5; padding:0.8rem; border-radius:6px; text-align:center}}
.kpi strong{{display:block; font-size:1.4rem; color:#0d2a54}}
img{{max-width:100%; border:1px solid #dee2e6; border-radius:4px; background:white}}
.small{{font-size:0.85rem; color:#555}}
ul{{margin:0.4rem 0}}
a{{color:#0d2a54}}
code{{background:#e9ecef; padding:0.1rem 0.3rem; border-radius:3px}}
</style>
</head>
<body>
<header>
<h1>EvoFederated — Dashboard Campaña Experimental</h1>
<p>NAS evolutiva en FL Non-IID · {total_exp} experimentos · {total_runs} evaluaciones estrategia · {total_time_min:.1f} min tiempo total</p>
<p class="small">Generado {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")} — datos en <code>results/experimental_campaign/</code> · Hall of Fame + Validación Exhaustiva + Checkpoints</p>
</header>
<div class="container">

<div class="card">
<h2>Resumen ejecución</h2>
<div class="kpi">
<div><strong>{total_exp}</strong>experimentos</div>
<div><strong>{total_runs}</strong>runs estrategia</div>
<div><strong>{total_evals}</strong>evals cliente-arquitectura</div>
<div><strong>{total_time_min:.1f} min</strong>tiempo (suma)</div>
<div><strong>~487 s</strong>FMNIST más largo</div>
<div><strong>26+33</strong>orig OK + retry OK</div>
</div>
<p><strong>Seeds:</strong> {seeds} &nbsp; <strong>Estrategias:</strong> {', '.join(strategies)} &nbsp; <strong>Datasets:</strong> synthetic (32) + fashion-mnist (1)</p>
<p><strong>Baselines comparados:</strong> <span class="badge">FULL</span> <span class="badge">RANDOM-2</span> <span class="badge">FIXED-DIVERGENT-2</span> <span class="badge">DYNAMIC-DIVERGENT-2 (τ=2)</span></p>
<p><strong>Presupuesto:</strong> pop 10-40, gen 5-20, α ∈ {{10,1,0.5,0.1,0.05}}, clientes 5/6/10, k ∈ {{1,3,5}}, τ ∈ {{0.5,1,2,5,10}}, 3 seeds en configs clave.</p>
<p class="small"><strong>Bugs corregidos (cambio mínimo):</strong> 1) <code>BatchNorm batch=1</code> → skip batch unitaria en <code>client.train</code> <code>src/evofederated/clients/client.py:84</code>; 2) <code>int64 JSON</code> → <code>default=str</code> en <code>runner.py:128</code> y conversión <code>int()</code> en <code>datasets.py:168</code>.</p>
<p class="small">Estructura: <code>results/experimental_campaign/&lt;fase&gt;/&lt;config&gt;/</code> cada uno con <code>config.json, summary.csv, hypervolume_history.csv, divergence_matrix.npy, plots/</code>. Ver <code>experiment_index.csv</code> y <code>aggregate_results.csv</code>.</p>
</div>

<div class="card">
<h2>1. Comparación estrategias (config razonable)</h2>
<h3>Synthetic — pop20_gen10, 6 clientes, α=0.5, k=2, τ=2 (coste medio)</h3>
<table class="table">
<tr><th>Estrategia</th><th>HV común</th><th>mean F1</th><th>worst F1</th><th>Evals</th><th>Saving</th><th>HV/eval</th></tr>
{row_for(synth,"full")}
{row_for(synth,"random")}
{row_for(synth,"fixed")}
{row_for(synth,"dynamic")}
</table>
<p class="small">Ratio DYNAMIC/FULL: HV={synth.get('ratio_dynamic_full_hv',0):.2f}, Evals={synth.get('ratio_dynamic_full_evals',0):.2f} (66% ahorro, 75% HV). RANDOM supera a FULL en HV (0.183 vs 0.147) y también a DYNAMIC.</p>
<h3>Fashion-MNIST — pop10_gen5, 6 clientes, α=0.5 (dataset real)</h3>
<table class="table">
<tr><th>Estrategia</th><th>HV común</th><th>mean F1</th><th>worst F1</th><th>Evals</th><th>Saving</th><th>HV/eval</th></tr>
{row_for(fmnist,"full")}
{row_for(fmnist,"random")}
{row_for(fmnist,"fixed")}
{row_for(fmnist,"dynamic")}
</table>
<p class="small">Ratio DYNAMIC/FULL: HV={fmnist.get('ratio_dynamic_full_hv',0):.2f}, Evals={fmnist.get('ratio_dynamic_full_evals',0):.2f} (65% ahorro, 96% HV). En FMNIST, FULL sigue líder pero DYNAMIC recupera muy bien y supera a RANDOM/FIXED.</p>
</div>

<div class="card">
<h2>2. Hypervolume vs coste</h2>
<div class="grid2">
<div><h3>Synthetic HV vs generación</h3><img src="assets/synthetic_hv_gen.png" alt="hv gen synth"><p class="small">Full converge lento; Random/Dynamic alcanzan HV similar en 5 gens con 1/3 evals. Gen20 no mejora (HV flat → convergencia temprana).</p></div>
<div><h3>Synthetic HV vs evals</h3><img src="assets/synthetic_hv_evals.png" alt="hv evals synth"><p class="small">HV/eval: Random 4.5e-4 > Dynamic 2.7e-4 > Full 1.2e-4. Eficiencia: partial 3× más efficiente.</p></div>
</div>
<div class="grid2">
<div><h3>FMNIST HV vs generación</h3><img src="assets/fmnist_hv_gen.png" alt="hv gen fmnist"><p class="small">FULL lidera pero DYNAMIC casi igual con 65% menos evals.</p></div>
<div><h3>FMNIST HV vs evals</h3><img src="assets/fmnist_hv_evals.png" alt="hv evals fmnist"><p class="small">Pendiente similar: partial alcanza 0.53 HV con 106 evals vs FULL 0.55 con 306.</p></div>
</div>
<div class="grid2">
<div><h3>Exhaustivo mean vs min — Synthetic</h3><img src="assets/synthetic_exhaustive.png" alt="exhaustive synth"></div>
<div><h3>FMNIST</h3><img src="assets/fmnist_exhaustive.png" alt="exhaustive fmnist"></div>
</div>
<p class="small">En FMNIST, puntos más dispersos y con mayor F1 (0.55-0.67) que synthetic (0.25-0.38). DYNAMIC no domina claramente.</p>
</div>

<div class="card">
<h2>3. Non-IID α</h2>
<div class="grid2">
<div><img src="assets/alpha10_class.png" alt="alpha10 class"><p class="small">α=10 ≈ IID: barras uniformes, CV tamaño 0.05, TV medio ~0.15</p></div>
<div><img src="assets/alpha005_class.png" alt="alpha005 class"><p class="small">α=0.05 fuerte Non-IID: algunos clientes casi mono-clase, CV 0.6+, TV 0.65</p></div>
</div>
<div class="grid2">
<div><img src="assets/alpha10_div.png" alt="alpha10 div"><p class="small">Divergencia α=10: distancias bajas, correlación proxy 0.38-0.44</p></div>
<div><img src="assets/alpha005_div.png" alt="alpha005 div"><p class="small">α=0.05: matriz contrastada, proxy Pearson 0.81 — divergencia Δ muy informativa cuando heterogeneidad es alta.</p></div>
</div>
<h3>HV común por α</h3>
{alpha_table_html}
<p class="small">Tendencia: a α↓ (más heterogéneo), HV sube para todas (espacio más amplio). RANDOM y DYNAMIC empatan con FULL en α=0.05 (0.255). En α=10 (IID) RANDOM mejor (0.204 vs FULL 0.154). DYNAMIC nunca claramente superior a RANDOM en synthetic; en FMNIST α=0.5, DYNAMIC 0.533 vs RANDOM 0.521 pequeña ventaja.</p>
</div>

<div class="card">
<h2>4. Sensibilidad</h2>
<h3>Población (gen=5, α=0.5)</h3>
{pop_table_html}
<p class="small">Pop 10→20→40: HV Full 0.114→0.125→0.176 (mejora), pero evals 306→606→1206 (casi lineal). HV/eval cae: no vale la pena 40 si coste es crítico. Dynamic escala similar: 0.092→0.110→0.153.</p>
<h3>Generaciones (pop=10)</h3>
{gen_table_html}
<p class="small">Gen 5→10 mejora Full 0.114→0.172 pero gen10→20 flat (0.172). Random/Fixed/Dynamic también plateau en gen10. Convergencia temprana → 5-10 gens suficiente para synthetic.</p>
<h3>Clientes (pop10 gen5 α=0.5)</h3>
{clients_table_html}
<p class="small">5 clientes: saving 58.8% (5 vs 10+char), 10 clientes: saving 78.4% (escala mejor con N). HV cae con N↑ (más objetivos difíciles): Full 0.146 (5c) →0.101 (10c).</p>
<h3>k (épocas caracterización)</h3>
{k_table_html}
<p class="small">k=1,3,5: diferencias mínimas (±0.03 HV). k=1 ya da proxy útil; k=5 no mejora consistentemente. Ahorro de caracterización: usar k=1-2.</p>
<h3>τ (presión divergente)</h3>
{tau_table_html}
<p class="small">τ 0.5-10: variación alta pero no monotónica. Mejor τ parece 10 (HV 0.146) y peor 5 (0.056) para Dynamic en α=0.5 baseline. Alta sensibilidad a seed, no conclusión firme con 1 seed. En multi-seed α0.1 τ5, dynamic 0.208 (bueno) vs random 0.178.</p>
</div>

<div class="card">
<h2>5. Pareto fronts representativos</h2>
<div class="grid3">
<div><h3>Synthetic Dynamic</h3><img src="assets/pareto_synth_dynamic.png" alt="pareto synth dynamic"></div>
<div><h3>Synthetic Random</h3><img src="assets/pareto_synth_random.png" alt="pareto synth random"></div>
<div><h3>Synthetic Full</h3><img src="assets/pareto_synth_full.png" alt="pareto synth full"></div>
</div>
<div style="margin-top:0.8rem"><h3>FMNIST Dynamic</h3><img src="assets/pareto_fmnist_dynamic.png" style="max-width:600px" alt="pareto fmnist"></div>
<p class="small">Full optimiza (mean,min) vs partial (F1_i,F1_j). Validación común muestra solapamiento; en FMNIST frentes más amplios y mejor F1.</p>
</div>

<div class="card">
<h2>5b. Global Pareto Generalization (Validación Exhaustiva)</h2>
<div class="grid2">
<div><h3>Synthetic — Mean vs Worst (exhaustivo)</h3><img src="assets/synthetic_exhaustive.png" alt="global pareto synth"><p class="small">Rojo = Pareto global (mean vs worst). Barras de error muestran trade-off robustez.</p></div>
<div><h3>FMNIST — Mean vs Worst</h3><img src="assets/fmnist_exhaustive.png" alt="global fmnist"><p class="small">F1 mayor en FMNIST (0.6) vs synthetic (0.28). Full y partial solapan pero Full ligero liderazgo en FMNIST (96% HV conservado).</p></div>
</div>
<div class="grid2">
<div><h3>Checkpoint HV vs generación (Synthetic pop20)</h3><img src="assets/synthetic_checkpoint_hv.png" alt="ckpt hv"><p class="small">HV común exhaustivo por checkpoint: muestra si Pareto global mejora con t. Idealmente pendiente positiva.</p></div>
<div><h3>Checkpoint Mean F1 vs generación</h3><img src="assets/synthetic_checkpoint_mean.png" alt="ckpt mean"><p class="small">Best mean F1 también plateau en gen10, consistente con HV interno.</p></div>
</div>
<div class="grid2">
<div><h3>Architecture × Client heatmap (Synthetic)</h3><img src="assets/synthetic_arch_client.png" alt="arch client"><p class="small">Cada fila una arquitectura Pareto exhaustiva; columnas clientes. Observa robustez: filas uniformes = buena generalización.</p></div>
<div><h3>Hall of Fame heatmap (Synthetic)</h3><img src="assets/synthetic_hof.png" alt="hof"><p class="small">Mejores arquitecturas acumuladas (sin duplicados). Hall of Fame size ~8-12, global pareto size 1-3.</p></div>
</div>
<p class="small">Estructura nuevos artefactos: <code>pareto_all_clients_f1.csv</code> (M p,i), <code>global_pareto.csv</code> (distinción Evolutionary vs Exhaustively Validated), <code>hall_of_fame.csv</code> (best mean/worst, compromiso, miembros Pareto global), <code>checkpoints/checkpoint_gen*.csv</code> + <code>hv_vs_generation_checkpoints.csv</code>. Coste separado: <code>search_evaluations / checkpoint_evaluations / final_validation_evaluations</code> en <code>cost.json</code>.</p>
</div>

<div class="card">
<h2>6. Divergencia Δ</h2>
<div class="grid2">
<div><img src="assets/synthetic_div.png" alt="synth div"><p class="small">Matriz D cosine, diag 0, simétrica, rango 0-0.4</p></div>
<div><img src="assets/synthetic_proxy.png" alt="proxy"><p class="small">Proxy vs True TV: Spearman mean {proxy['mean_spearman']:.2f} (±{proxy['std_spearman']:.2f}), rango [{proxy['min_spearman']:.2f},{proxy['max_spearman']:.2f}] n={proxy['n']}. Correlación positiva, fuerte cuando α bajo (0.81) y moderada en IID (0.38).</p></div>
</div>
<p class="small">Efecto k: cambios pequeños en D (cosine). Stable con k=1.</p>
</div>

<div class="card">
<h2>7. Multi-seed estabilidad</h2>
<table class="table">
<tr><th>Config</th><th>Estrategia</th><th>mean HV</th><th>std</th><th>n</th></tr>
<tr><td rowspan="4">Baseline α0.5 (pop10 gen5)</td><td>full</td><td>{ms_baseline.get('full',{}).get('mean',0):.3f}</td><td>{ms_baseline.get('full',{}).get('std',0):.3f}</td><td>{ms_baseline.get('full',{}).get('n',0)}</td></tr>
<tr><td>random</td><td>{ms_baseline.get('random',{}).get('mean',0):.3f}</td><td>{ms_baseline.get('random',{}).get('std',0):.3f}</td><td>{ms_baseline.get('random',{}).get('n',0)}</td></tr>
<tr><td>fixed</td><td>{ms_baseline.get('fixed',{}).get('mean',0):.3f}</td><td>{ms_baseline.get('fixed',{}).get('std',0):.3f}</td><td>{ms_baseline.get('fixed',{}).get('n',0)}</td></tr>
<tr><td>dynamic</td><td>{ms_baseline.get('dynamic',{}).get('mean',0):.3f}</td><td>{ms_baseline.get('dynamic',{}).get('std',0):.3f}</td><td>{ms_baseline.get('dynamic',{}).get('n',0)}</td></tr>
<tr><td rowspan="4">Alta heterogeneidad α0.05</td><td>full</td><td>{ms_alpha05.get('full',{}).get('mean',0):.3f}</td><td>{ms_alpha05.get('full',{}).get('std',0):.3f}</td><td>{ms_alpha05.get('full',{}).get('n',0)}</td></tr>
<tr><td>random</td><td>{ms_alpha05.get('random',{}).get('mean',0):.3f}</td><td>{ms_alpha05.get('random',{}).get('std',0):.3f}</td><td>{ms_alpha05.get('random',{}).get('n',0)}</td></tr>
<tr><td>fixed</td><td>{ms_alpha05.get('fixed',{}).get('mean',0):.3f}</td><td>{ms_alpha05.get('fixed',{}).get('std',0):.3f}</td><td>{ms_alpha05.get('fixed',{}).get('n',0)}</td></tr>
<tr><td>dynamic</td><td>{ms_alpha05.get('dynamic',{}).get('mean',0):.3f}</td><td>{ms_alpha05.get('dynamic',{}).get('std',0):.3f}</td><td>{ms_alpha05.get('dynamic',{}).get('n',0)}</td></tr>
</table>
<p class="small">Baseline α0.5: std 0.03-0.04 (30% variación) — alta variabilidad con 1 seed es engañosa. α0.05: Full std 0.03, Random/Dynamic varían más (0.1-0.2). No hay dominancia estable.</p>
</div>

<div class="card">
<h2>7b. Quality vs Cost (análisis central)</h2>
<div class="grid2">
<div><h3>Synthetic HV vs evals</h3><img src="assets/synthetic_hv_evals.png" alt="hv evals"><p class="small">HV común vs cumulative search evaluations (sin checkpoints). Pendiente partial > Full.</p></div>
<div><h3>FMNIST HV vs evals</h3><img src="assets/fmnist_hv_evals.png" alt="hv evals fmnist"><p class="small">La gráfica más importante: Full 306 evals → HV 0.555; Dynamic 106 evals → HV 0.533 (96% quality, 65% saving). Box muestra calidad/coste.</p></div>
</div>
<p class="small"><strong>Ecuaciones coste:</strong> E<sub>FULL</sub>≈R·G·N, E<sub>DYNAMIC</sub>≈2·R·G (+ k·N caracterización). Saving = 1−E_method/E_FULL. Checkpoints se contabilizan separado como coste científico, no operativo.</p>
</div>

<div class="card">
<h2>8. Principales hallazgos</h2>
<ul>
<li><strong>¿Dynamic reduce evals?</strong> Sí, consistentemente: 65% (6 clientes), 78% (10c), 58% (5c). E_dynamic/E_full ≈ 0.34-0.41. Incluyendo caracterización, saving neto 58-78%.</li>
<li><strong>¿Cuánto HV pierde/gana?</strong> En FMNIST DYNAMIC conserva 96% HV (0.533/0.555). En synthetic pop20 DYNAMIC conserva 75% (0.110/0.147). En algunos configs Random gana: pop10 gen5 Random 0.124 > Full 0.114. No hay pérdida sistemática grande, pero tampoco ganancia clara.</li>
<li><strong>¿Dynamic > Random?</strong> No consistente. Synthetic baseline: Random 0.124 > Dynamic 0.092. Pop20: Random 0.183 > Dynamic 0.110. FMNIST: Dynamic 0.533 ≈ Random 0.521 (ligera ventaja 2%). En α sweeps, a veces Random mejor, a veces Dynamic. Diferencia pequeña frente a variabilidad seed.</li>
<li><strong>¿Aporta divergencia vs aleatorio?</strong> Débil evidencia. Proxy correlación media 0.55, alta en Non-IID fuerte, pero no se traduce en HV consistente. Fixed vs Dynamic también similares.</li>
<li><strong>Fixed vs Dynamic?</strong> Muy similares, a veces Fixed mejor (α0.1: Fixed 0.210 vs Dynamic 0.208; gen10: Fixed 0.147 vs Dynamic 0.094).</li>
<li><strong>Non-IID:</strong> Heterogeneidad ↑ aumenta HV para todos y mejora correlación proxy. En α=0.05, todos empatan en 0.255 (frente trivial). En IID Random brilla.</li>
<li><strong>Población:</strong> Mayor pop mejora HV pero coste lineal; pop20-40 mejor compromiso que 10 si presupuesto permite.</li>
<li><strong>Generaciones:</strong> Convergencia en ~5-10 gens; gen20 no aporta.</li>
<li><strong>k:</strong> Casi irrelevante 1-5; usar k=1-2 ahorra tiempo.</li>
<li><strong>τ:</strong> Sensible pero sin patrón claro; τ=10 dio mejor en baseline, τ=5 peor; necesita más seeds.</li>
<li><strong>HV converge?</strong> Sí, plateau a gen10.</li>
<li><strong>Mejor HV/coste:</strong> Random suele tener mejor ratio HV/eval (ej. synthetic pop20 Random 4.5e-4 vs Full 1.2e-4). Dynamic similar a Random.</li>
<li><strong>Divergencia útil?</strong> Correlación positiva con TV real, especialmente en alta heterogeneidad, pero contribución a HV no demostrada con estos seeds.</li>
<li><strong>Estabilidad seeds?</strong> Alta variabilidad (std 0.03-0.1 sobre media 0.1-0.3). 1 seed no es concluyente; 3 seeds muestran solapamiento intervals.</li>
</ul>
<p class="small">No se favorece hipótesis: reportamos Random≈Dynamic y a veces Random>Dynamic y Full solo ligeramente superior en FMNIST. Ahorro real, pérdida HV moderada.</p>
</div>

<div class="card">
<h2>9. Limitaciones</h2>
<ul>
<li>Solo 3 seeds en 2 configs; resto 1 seed → alta incertidumbre.</li>
<li>Synthetic principalmente (32 runs) vs solo 1 FMNIST (1 run, 1 seed) → generalización limitada.</li>
<li>Espacio NAS MLP pequeño (l_max 3, 4 neuron choices, 2 activaciones) — no CNN/NAS-Bench.</li>
<li>Población/generaciones modestas (10-40, 5-20) por presupuesto CPU (20 min total). No se probó 50+ generaciones.</li>
<li>α=0.01 extremo no probado; clientes 20 no probado (solo 5/6/10).</li>
<li>Tau y k con 1 seed cada; necesidad de más replicación.</li>
<li>BatchNorm + batch 1 bug inicial afectó 1 config (solucionado).</li>
<li>JSON int64 bug en alta heterogeneidad (solucionado con default=str).</li>
<li>HV referencia fijo [0.1,0.1] en espacio minimización; comparación válida solo tras validación común.</li>
</ul>
</div>

<div class="card">
<h2>10. Artefactos y reproducibilidad</h2>
<ul>
<li><code>experiment_index.csv</code> — {total_exp} filas, mapeo config↔directorio (dataset, α, pop, gen, k, τ, seed, proxy, TV)</li>
<li><code>aggregate_results.csv</code> — {total_runs} filas (4 estrategias ×{total_exp} exps) con HV común/interno, mean/median/min/std F1, evals, saving, proxy, TV, hv_per_eval</li>
<li><code>analysis.json</code> — hallazgos estructurados + sensitivity pivots + multi-seed stats + proxy</li>
<li>Cada experimento: <code>config.json/yaml, metadata.json, heterogeneity.json, divergence_matrix.npy, hypervolume_history.csv, summary.csv, exhaustive_validation.csv, pareto_all_clients_f1.csv, pareto_all_clients_accuracy.csv, global_pareto.csv, hall_of_fame.csv, checkpoints/*.csv, cost.json, plots/*.png</code></li>
<li>Top-level agregados: <code>pareto_all_clients_f1.csv</code>, <code>hall_of_fame.csv</code>, <code>checkpoints/checkpoints_all_exhaustive.csv</code>, <code>global_validation/global_exhaustive.csv</code></li>
<li>Comando: <code>python run_campaign.py</code> (o <code>python run_experiment.py --config configs/small.yaml</code>) + <code>python generate_analysis.py && python generate_dashboard.py</code></li>
<li>Tests: <code>pytest tests -v</code> 31 passed (23 base + 8 exhaustive validation: pareto exhaustive, independent init, no weight contamination, checkpoint isolation, HOF, global pareto, HV, persistence)</li>
<li>Seeds fijados en dataset/char/evolution; determinismo salvo nondeterminismo PyTorch (cudnn deterministic/benchmark off).</li>
</ul>
<p class="small">Total evals {total_evals}, tiempo suma {total_time_min:.1f} min, wall ~26.6 min. Guardado sin sobrescritura en <code>results/experimental_campaign/</code> + <code>dashboard/assets/</code>.</p>
</div>

<div class="card">
<h2>11. Recomendaciones próximo presupuesto</h2>
<ul>
<li>5 seeds en: baseline synthetic, FMNIST, α=0.05 y α=10 para intervalo confianza HV.</li>
<li>FMNIST con pop20 gen10 y 3 seeds (coste 3× 8 min ≈ 24 min) — validar si Full mantiene ventaja.</li>
<li>Sweep τ con 3 seeds (0.5,2,10) bajo α=0.1 y α=0.5.</li>
<li>Clientes 20 con synthetic pop20 gen10 (evals 20*10*20=4000 full vs 400 partial, saving 90% teórico).</li>
<li>k=1 vs 5 con 3 seeds cada para confirmar estabilidad proxy.</li>
<li>Probar distance l2_norm vs cosine; y espacio CNN.</li>
<li>Optimizar plots: cerrar figuras para evitar warning >20 figuras.</li>
</ul>
</div>

</div>
<footer style="text-align:center; padding:1rem; color:#666; font-size:0.85rem">EvoFederated — Ponytail lazy: solución simple, sin React, solo HTML/CSS y plots existentes. Dashboard portable: abrir <code>dashboard/index.html</code> directamente.</footer>
</body>
</html>
"""

with open(DASH / "index.html", "w") as f:
    f.write(html)
print(f"Wrote dashboard to {DASH / 'index.html'} size {len(html)}")
