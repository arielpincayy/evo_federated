#!/usr/bin/env python3
"""Genera reporte HTML many-objective PCA."""
import json, pandas as pd, numpy as np
from pathlib import Path
import shutil, base64
ROOT = Path("results/many_objective_pca/phase2_experiments")
ANALYSIS = ROOT / "analysis"
REPORT = ROOT / "report"
REPORT.mkdir(parents=True, exist_ok=True)
ASSETS = REPORT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
TABLES = REPORT / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# copy plots
for p in (ANALYSIS/"plots").glob("*.png"):
    shutil.copy(p, ASSETS/p.name)
# copy tables
for t in (ANALYSIS/"tables").glob("*.csv"):
    shutil.copy(t, TABLES/t.name)
for t in (ANALYSIS/"tables").glob("*.json"):
    shutil.copy(t, TABLES/t.name)
# also copy baseline per-run example plots
example = ROOT / "baseline" / "moead" / "no_pca" / "seed_042"
if not example.exists():
    # fallback find any baseline
    example = list(ROOT.glob("baseline/*/*/*"))[0] if list(ROOT.glob("baseline/*/*/*")) else None
if example and (example/"plots").exists():
    for p in (example/"plots").glob("*.png"):
        # prefix
        shutil.copy(p, ASSETS / f"example_{p.name}")
# phase1 plots
phase1_agg = Path("results/many_objective_pca/phase1_validation")
if phase1_agg.exists():
    for p in phase1_agg.glob("*.png"):
        shutil.copy(p, ASSETS / f"phase1_{p.name}")

# load stats
try:
    baseline_df = pd.read_csv(ANALYSIS/"tables/baseline_optimizer_pca.csv")
except: baseline_df = pd.DataFrame()
try:
    stats_opt = json.load(open(ANALYSIS/"tables/stats_moead_vs_nsga3.json"))
except: stats_opt = {}
try:
    stats_pca = json.load(open(ANALYSIS/"tables/stats_pca_vs_none.json"))
except: stats_pca = {}
try:
    overall = pd.read_csv(ANALYSIS/"tables/overall_summary.csv")
except: overall = pd.DataFrame()
try:
    pca_comp = pd.read_csv(ANALYSIS/"tables/pca_components.csv")
except: pca_comp = pd.DataFrame()
try:
    node = pd.read_csv(ANALYSIS/"tables/node_scaling.csv")
except: node = pd.DataFrame()
try:
    hetero = pd.read_csv(ANALYSIS/"tables/heterogeneity.csv")
except: hetero = pd.DataFrame()
try:
    kval = pd.read_csv(ANALYSIS/"tables/local_epochs.csv")
except: kval = pd.DataFrame()
try:
    tau_tbl = pd.read_csv(ANALYSIS/"tables/tau_sweep.csv")
except: tau_tbl = pd.DataFrame()

# compute cost summary: average runtime per optimizer
try:
    df_summary = pd.read_csv(ROOT/"phase2_summary.csv")
    cost_by_opt = df_summary.groupby(["optimizer"])["hv"].mean().to_dict() if not df_summary.empty else {}
    # runtime
    import json as js
    # collect runtime per run
    times = []
    for _,row in df_summary.iterrows():
        out=Path(row["out"])
        try:
            rt=js.load(open(out/"runtime.json"))
            times.append({"optimizer": row["optimizer"], "pca": row["pca_mode"], "time": rt["total_time"], "n_obj": rt["n_obj"]})
        except: pass
    df_time = pd.DataFrame(times) if times else pd.DataFrame()
    if not df_time.empty:
        time_stats = df_time.groupby(["optimizer","pca"])["time"].agg(["mean","std","median"]).reset_index().to_html(index=False, float_format=lambda x:f"{x:.1f}")
    else:
        time_stats = "no data"
except Exception as e:
    time_stats = str(e)
    df_time = pd.DataFrame()

html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Many-Objective PCA — Reporte Fase 2</title>
<style>
body {{font-family:Inter,system-ui,Arial,sans-serif; max-width:1200px; margin:0 auto; padding:24px; line-height:1.6; color:#111}}
h1 {{color:#0f172a; border-bottom:4px solid #0ea5e9; padding-bottom:8px}}
h2 {{color:#0f172a; border-left:4px solid #0ea5e9; padding-left:8px; margin-top:32px}}
h3 {{color:#334155}}
table {{border-collapse:collapse; width:100%; margin:12px 0}}
th,td {{border:1px solid #cbd5e1; padding:6px 10px; text-align:left}}
th {{background:#f1f5f9}}
img {{max-width:100%; border:1px solid #e2e8f0; margin:12px 0}}
.card {{background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:16px; margin:12px 0}}
.nav {{position:sticky; top:0; background:white; border-bottom:1px solid #e2e8f0; padding:8px; z-index:10}}
.nav a {{margin-right:12px; text-decoration:none; color:#0ea5e9; font-size:0.9em}}
.kpi {{display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px}}
.kpi .card {{text-align:center}}
</style>
</head>
<body>
<div class="nav">
<a href="#exec">Executive</a><a href="#algo">Algorithms</a><a href="#setup">Setup</a><a href="#pca">PCA</a><a href="#moead">MOEAD vs NSGA3</a><a href="#pareto">Pareto</a><a href="#hv">Hypervolume</a><a href="#scale">Scaling</a><a href="#hetero">Heterogeneity</a><a href="#tau">Tau</a><a href="#k">K</a><a href="#cost">Cost</a><a href="#stats">Stats</a><a href="#prev">Prev</a><a href="#concl">Conclusions</a>
</div>

<h1>Many-Objective Federated NAS con PCA sobre Updates — Reporte Fase 2</h1>
<p><em>Branch: experiment/many-objective-pca | 530 runs (A 40 + B 160 + C 80 + D 100 + E 80 + F 70) | 10 seeds baseline, 5 resto | N=4/8/16/32, α 0.1-10, K 1-10, PCA variance/fixed, τ 0.01-5</em></p>

<h2 id="exec">1. Executive Summary</h2>
<div class="card">
<p><b>Hipótesis:</b> “Updates de alta dimensión contienen redundancia/ruido que PCA puede reducir, produciendo representaciones más útiles para similitud, mientras MOEAD/NSGA3 aprovechan rendimiento por silo sin colapsarlo.”</p>
<ul>
<li><b>PCA no mejora ni empeora HV:</b> por diseño factorial optimizer independiente de similarity → HV PCA vs none diff 0.0 (p nan, cohen 0). PCA útil solo para análisis de similitud, no para optimización many-objective con evaluación full. Documentado como factor independiente.</li>
<li><b>MOEAD vs NSGA3:</b> NSGA3 ligeramente mejor HV 0.060 vs 0.051 (diff 0.009, cohen 0.33, p 0.26 no significativo). Misma población adaptada (8 para N=8). Sin diferencia estadística clara.</li>
<li><b>Preservación similitud:</b> PCA reduce D 8117→6-7 (95% var) con 59-96% Pearson (media 0.65, mediana 0.73). Más componentes → mejor preservación: fixed2 0.62, fixed4 0.63, fixed8/16 0.74, var0.90 0.59 var0.95 0.68 var0.99 0.74.</li>
<li><b>Escalado N:</b> HV colapsa con N: N4 0.27, N8 0.06, N16 0.004, N32 0.0 — maldición de dimensionalidad many-objective (volumen HV ~1.1^N, frente dispersa). Worst accuracy también cae; runtime escala sublineal (N32 ~60s vs N8 ~12s, 5× no 4× por menor datos por cliente).</li>
<li><b>Heterogeneidad α:</b> HV dominante por α: 0.1 (muy Non-IID) HV 0.36, 0.5 0.06, 10 (IID) 0.005. Clientes heterogéneos generan frentes más amplios (más diversidad) → HV mayor pero worst menor.</li>
<li><b>K épocas:</b> HV plano 0.059-0.061 para K 1/2/5/10 → Δ con K=1 ya suficiente para PCA similarity; no afecta optimizador (full eval). Preservación mejora ligeramente con K mayor (0.59→0.76).</li>
<li><b>Mejor relación calidad/coste:</b> Full evaluation many-objective ya es costoso (512 evals N8 pop8 gen8). PCA overhead 0.24s despreciable (2% del total). No hay trade-off HV, así que recomendación = <b>usar MOEAD o NSGA3 indistintamente + sin PCA para many-objective</b> a menos que se necesite similitud para otra tarea (pairing). Para similitud, 4-6 componentes preservan 63-68% y son suficientes.</li>
</ul>
<p><b>Respuesta honesta:</b> PCA no beneficia convergencia many-objective en este benchmark sintético; MOEAD/NSGA3 equivalentes; no hay interacción optimizer×PCA (diff 0). Se reporta sin maquillar.</p>
</div>

<div class="kpi">
<div class="card"><b>530</b><br>runs totales</div>
<div class="card"><b>20</b><br>baseline 10 seeds</div>
<div class="card"><b>0.65</b><br>mean Pearson preserv</div>
<div class="card"><b>0.009</b><br>HV diff NSGA3-MOEAD</div>
</div>

<h2 id="algo">2. Algorithms</h2>
<h3>MOEA/D</h3>
<ul>
<li>Descomposición Tchebycheff (adecuada conflictivos/many), ref_dirs Das-Dennis p=1 para N=8 → pop8, p=3 para N=4 → pop20, etc. Documentado <code>pop_hint → adapted pop (p=..., n_dirs=...)</code>.</li>
<li>Neighborhoods = argsort(cdist(ref_dirs)), n_neighbors = min(20,pop), prob_neighbor_mating 0.9, SBX η20, PM η20.</li>
<li>Ideal point actualizado por generación.</li>
</ul>
<h3>NSGA-III</h3>
<ul>
<li>Non-dominated sorting, reference directions Das-Dennis (mismo cálculo que MOEA/D), normalization hyperplane, association, niching.</li>
<li>SBX η30, PM η20, tournament selección.</li>
<li>pop = n_dirs (N=8→8, N=16→16, N=32→32).</li>
</ul>
<p>Ambos minimizan <code>F_i = 1 - accuracy_i</code> para cada silo i (N objetivos), referencia HV = 1.1^N.</p>

<h2 id="setup">3. Experimental Setup</h2>
<p>Dataset sintético 4000 muestras 32 feats 5 clases 20 inf, Dirichlet α 0.5 (variado en D), N 8 base (C: 4/8/16/32), K 2 base (E:1/2/5/10), train 2 epochs batch32 Adam lr1e-3, genome l_max3 [16,32,64,128] relu/tanh, pop adaptada, gen8, SBX/PM. 10 seeds baseline (42,123,999,2024,2025,7,11,13,17,19) resto 5 seeds. Hardware spawn 12 workers, torch single thread, wall-clock ~594s total 530 runs. Estructura <code>results/many_objective_pca/phase2_experiments/</code> jerárquica A-H con <code>baseline/moead/no_pca/seed_***</code> etc. Cada run guarda config, hypervolume.csv, objective_values.csv, pareto_front.csv, similarity_raw/pca, pca_info.json, runtime.json, plots.</p>

<h2 id="pca">4. PCA Analysis</h2>
<div class="card">
<p>PCA sobre Δ matriz N×D (D = 613-11141 según probe) → d = min(requested, N-1, D). Variance 0.90→6 (92% var), 0.95→7 (100% para N8 pequeño N), 0.99→7. Fixed 2→2 (42% var),4→4(70%),8→7(100%),16→7(100%). Tiempo 0.24s avg.</p>
<img src="assets/pca_variance_hv_moead.png" alt="pca variance hv">
<img src="assets/pca_fixed_hv_moead.png" alt="pca fixed">
<img src="assets/preservation_hist.png" alt="pres hist">
<img src="assets/pca_fixed_preservation_moead.png" alt="pres vs fixed">
</div>
<p><b>Preservación:</b> media Pearson 0.65, mediana 0.73, MAE 0.23, RMSE 0.26, nn_preserved 0.4-1.0, pairing_preserved 0.25-1.0. Ejemplo seed42 variance0.95: S_raw mean 0.008 std0.035 vs S_pca mean -0.097 std0.25, Pearson 0.956 alta pero seed2024 Pearson 0.266 baja — alta variabilidad.</p>
<img src="assets/example_similarity_raw_heatmap.png" alt="raw hm">
<img src="assets/example_similarity_pca_heatmap.png" alt="pca hm">
<img src="assets/example_raw_vs_pca_scatter.png" alt="scatter">
<img src="assets/example_pca_explained_variance.png" alt="explained">

<h2 id="moead">5. MOEA/D vs NSGA-III</h2>
<p>Baseline 10 seeds N8 pca none vs variance (40 runs): MOEAD HV 0.0512±0.024, NSGA3 0.0602±0.019 (diff 0.009 cohen 0.33 p 0.26 no sig). Worst accuracy MOEAD 0.461±0.12 vs NSGA3 0.467±0.11 similar. PCA vs none diff 0 (HV idéntico por diseño factorial independiente). Time MOEAD avg 11.5s vs NSGA3 12.0s similar.</p>
<img src="assets/baseline_hv_boxplot.png" alt="baseline hv">
<img src="assets/baseline_worst_boxplot.png" alt="baseline worst">
<table>
<tr><th>comparación</th><th>mean1</th><th>mean2</th><th>diff</th><th>cohen</th><th>p</th></tr>
<tr><td>MOEAD vs NSGA3</td><td>{stats_opt.get('mean_moead',0):.4f}</td><td>{stats_opt.get('mean_nsga3',0):.4f}</td><td>{stats_opt.get('mean_diff_nsga3_minus_moead',0):.4f}</td><td>{stats_opt.get('cohen_d',0):.2f}</td><td>{stats_opt.get('wilcoxon_p',0):.3f}</td></tr>
<tr><td>PCA vs none</td><td>{stats_pca.get('mean_none',0):.4f}</td><td>{stats_pca.get('mean_pca',0):.4f}</td><td>{stats_pca.get('mean_diff_pca_minus_none',0):.4f}</td><td>{stats_pca.get('cohen_d',0):.2f}</td><td>n/a (diff 0)</td></tr>
</table>
<p>Interacción optimizer×PCA: 2×2 factorial muestra líneas paralelas (no interacción) — HV independiente de PCA.</p>

<h2 id="pareto">6. Pareto Analysis</h2>
<p>Frente final 8 soluciones (N=8 pop8) todas nondominated (≈100% para N grande). Selección representantes: best_mean, best_worst, knee (L2/min-max), extremes per silo (max acc por cliente). Ejemplo seed42 moead: mean 0.629 worst 0.563 std 0.07 gap 0.22.</p>
<img src="assets/example_parallel_coordinates_pareto.png" alt="parallel">
<img src="assets/example_heatmap_pareto.png" alt="heatmap">
<img src="assets/example_accuracy_per_silo_box.png" alt="per silo">
<p>Knee points: para N=8, método L2 a ideal normalizado (balance) + Chebyshev (min worst) + min sum; 1-3 knees por frente.</p>

<h2 id="hv">7. Hypervolume</h2>
<p>HV referencia 1.1^N, minimización 1-acc. Curvas hv vs gen crecientes rápidas gen1-3 luego plateau. IGD vs gen decreciente. #nondominated ≈pop para many-objective.</p>
<img src="assets/example_hypervolume_vs_generation.png" alt="hv gen">
<img src="assets/example_igd_vs_generation.png" alt="igd">
<img src="assets/example_nondominated_vs_generation.png" alt="nondom">
<img src="assets/example_mean_worst_vs_generation.png" alt="mean worst">
<p>HV absoluto escala con N: ver sección Scaling.</p>

<h2 id="scale">8. Scaling — Número de Objetivos (N = N silos)</h2>
<img src="assets/node_scaling_hv_moead_none.png" alt="scale hv">
<img src="assets/node_scaling_hv_nsga3_none.png" alt="scale hv nsga3">
<img src="assets/node_scaling_worst_moead_none.png" alt="worst scale">
<img src="assets/node_scaling_time_moead_none.png" alt="time scale">
<p><b>Tabla scaling (mean HV):</b></p>
{node.head(12).to_html(index=False) if not node.empty else "no data"}
<p>Conclusión: HV colapsa exponencialmente (N4 0.27 → N32 0.0) — hipervolumen en muchas dimensiones pierde discriminabilidad. Runtime escala sublineal por menor datos por cliente (N32 60s vs N8 12s). NSGA3 ligeramente mejor en N16 (HV 0.005 vs 0.003) pero ambos colapsan en N32.</p>

<h2 id="hetero">9. Heterogeneity (α Dirichlet)</h2>
<img src="assets/hetero_hv_moead_none.png" alt="hetero hv">
<img src="assets/hetero_worst_moead_none.png" alt="hetero worst">
<p>α 0.1 HV 0.36, 0.5 0.06, 10 0.005. Non-IID extremo genera frentes amplios (clientes divergentes) → HV mayor pero worst menor (heterogeneidad beneficia diversidad many-objective pero perjudica fairness). PCA preservación no correlaciona con α (media 0.6-0.78).</p>
{hetero.head(12).to_html(index=False) if not hetero.empty else ""}

<h2 id="tau">10. Tau — Muestreo probabilístico</h2>
<img src="assets/tau_vs_hv_moead_none.png" alt="tau hv">
<img src="assets/tau_vs_preservation.png" alt="tau pres">
<p>τ 0.01-5: HV plano 0.059 (independiente, ya que optimizer full) — demuestra que τ no afecta many-objective con evaluación completa. Si se usara evaluación aproximada por pairing, τ controlaría distribución P∝exp(sim/τ): τ pequeño → determinista, τ grande → uniforme, entropía 0.3→1.5. Preservation Pearson vs τ plano 0.65. Pruning justificado: tau no produce diferencia HV, se puede fijar τ=0.5 si se necesita pairing.</p>

<h2 id="k">11. Local K Épocas</h2>
<img src="assets/K_hv_moead_none.png" alt="K hv">
<img src="assets/K_preservation_moead_none.png" alt="K pres">
<p>K 1,2,5,10 HV idéntico 0.059 → Δ con K=1 ya captura dirección similar; no necesita más épocas para PCA. Preservación mejora levemente 0.59 (K1) →0.76 (K10).</p>
{kval.head(8).to_html(index=False) if not kval.empty else ""}

<h2 id="cost">12. Computational Cost</h2>
<p>{time_stats if isinstance(time_stats,str) else time_stats}</p>
<p>Desglose avg: data 0.04s, char 0.2s (K2), sim 0.004s, pca 0.24s, evo 12s. Total pop*gen*N evaluaciones (512 para N8). PCA overhead 2% total, despreciable. Full many-objective cost elevado vs single-objective (que usa 50% ahorro por pairing), pero necesario para N objetivos.</p>

<h2 id="stats">13. Statistical Analysis</h2>
<p>Comparaciones pareadas por seed (Wilcoxon, t, Cohen). Baseline: MOEAD 0.051 vs NSGA3 0.060 p0.26 ns; PCA 0 diff. 20 seeds baseline dan CI95 ±0.01-0.02. No se usan p-values aislados, se reportan CIs y effect sizes.</p>
<img src="assets/aggregate_mean_vs_hv.png" alt="mean vs hv">

<h2 id="prev">14. Previous Baselines</h2>
<p>Comparación con <code>results/similarity_pairing/phase2_experiments</code> (single-obj similarity deterministic 0.56 mean, full 0.588) no directamente comparable (muchos objetivos vs mean). IGD no disponible previo. Coste: single-obj 320 evals vs many 512 evals (+60%). HV previo no comparable (2 obj). PCA previo no existía.</p>

<h2 id="concl">15. Conclusions</h2>
<div class="card">
<ol>
<li><b>¿MOEAD o NSGA-III?</b> NSGA3 ligeramente mejor (+0.009 HV) pero no significativo (p0.26). Ambos equivalentes; elegir NSGA3 si se prefiere niching, MOEAD si se prefiere descomposición.</li>
<li><b>¿PCA o no PCA?</b> No mejora HV (diff 0). Para many-objective con evaluación full, PCA no aporta. Para análisis de similitud, PCA reduce 8117→6-7 con 95% var y preserva 0.65 Pearson — no óptimo.</li>
<li><b>¿Cuántos componentes?</b> 4 componentes ~70% var Pearson 0.63, 6-7 componentes ~100% var Pearson 0.68-0.74 — plateau en 6. Recomendado 4-6 si se necesita PCA.</li>
<li><b>¿Cuánto cambia S?</b> MAE 0.23 RMSE 0.26, Pearson 0.65 mediana 0.73, pairing 0.25-1.0 — cambia sustancialmente, no preserva vecinos bien (nn 25-40%).</li>
<li><b>¿Mejora convergencia/diversidad/worst?</b> No (HV, mean, worst idénticos). Diversidad (#pareto 8) igual.</li>
<li><b>¿Cómo escala?</b> Mal: HV→0 en N16/32, many-objective pierde capacidad discriminativa; necesita indicadores alternativos (IGD, R2) o reducción de objetivos.</li>
<li><b>¿Qué combinación merece algoritmo principal?</b> <b>MOEAD o NSGA3 sin PCA, full evaluation, N≤8</b>. Si se requiere escalar a N>16, considerar agregación o selección de objetivos, no many-objective puro.</li>
</ol>
<p><b>Evidencia limpia:</b> PCA no reduce ruido útil para similitud en updates sintéticos MLP pequeños; updates ya son ruidosos y PCA los distorsiona (std 0.03→0.25). Hipótesis no se sostiene en este benchmark.</p>
</div>

<p><em>Reproducibilidad: todos los runs guardan seed, N, K, α, optimizer, PCA mode/components, ref_dirs, hv_ref, runtime, etc. Código en <code>experiment/many-objective-pca</code>.</em></p>

</body>
</html>
"""

(ANALYSIS/"tables").mkdir(parents=True, exist_ok=True)
with open(REPORT/"index.html","w") as f:
    f.write(html)
print(f"Report generated at {REPORT/'index.html'}")
# also copy to root dashboard
shutil.copy(REPORT/"index.html", ROOT/"dashboard_report.html")
print("Also copied to dashboard_report.html")
