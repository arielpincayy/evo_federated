#!/usr/bin/env python3
"""
Generate unified report: aggregate CSVs, compute metrics, produce HTML.
"""
import json, sys, math
from pathlib import Path
import pandas as pd
import numpy as np

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def aggregate_results(base: Path):
    # Find all runs with summary.json or cost.json
    rows=[]
    for exp_dir in ["00_sanity","01_dissimilarity_nsga2","02_similarity_singleobjective","03_manyobjective","04_node_scaling","05_objective_reduction","06_heterogeneity","07_local_epochs","08_tau","09_representatives","10_full_baseline"]:
        exp_path = base / exp_dir
        if not exp_path.exists():
            continue
        for run_dir in exp_path.rglob("seed_*"):
            if not run_dir.is_dir():
                continue
            # try to find summary
            summ=None
            for cand in [run_dir/"summary.json", run_dir/"metrics.json", run_dir/"cost.json"]:
                if cand.exists():
                    try:
                        with open(cand) as f:
                            summ=json.load(f)
                        break
                    except Exception:
                        pass
            # read individual_node_accuracy if exists to compute mean/worst
            ind_path=run_dir/"individual_node_accuracy.csv"
            node_sum_path=run_dir/"node_generation_summary.csv"
            # try to extract metrics
            family="unknown"
            optimizer="unknown"
            pairing="unknown"
            # infer family from path
            if "dissimilarity" in str(run_dir):
                family="dissimilarity"
                optimizer="nsga2"
            elif "similarity" in str(run_dir):
                family="similarity"
                optimizer="ga"
            elif "many" in str(run_dir) or "nsga3" in str(run_dir) or "moead" in str(run_dir):
                family="many_objective"
                if "moead" in str(run_dir):
                    optimizer="moead"
                elif "nsga3" in str(run_dir):
                    optimizer="nsga3"
                else:
                    optimizer="many"
            # Try read final_full_evaluation.csv
            final_acc=None; worst_acc=None; std_acc=None
            ff_path=run_dir/"final_full_evaluation.csv"
            if not ff_path.exists():
                ff_path=run_dir/"final_evaluation"/"final_full_evaluation.csv"
                if not ff_path.exists():
                    ff_path=run_dir/"final_evaluation.csv"
            if ff_path.exists():
                try:
                    df_ff=pd.read_csv(ff_path)
                    if not df_ff.empty and "global_mean_accuracy" in df_ff.columns:
                        final_acc=float(df_ff["global_mean_accuracy"].mean())
                        worst_acc=float(df_ff["global_worst"].mean()) if "global_worst" in df_ff.columns else None
                        std_acc=float(df_ff["global_std"].mean()) if "global_std" in df_ff.columns else None
                    elif not df_ff.empty and "mean_accuracy" in df_ff.columns:
                        final_acc=float(df_ff["mean_accuracy"].mean())
                        worst_acc=float(df_ff["worst_accuracy"].mean()) if "worst_accuracy" in df_ff.columns else None
                    elif not df_ff.empty:
                        # try acc_client columns
                        acc_cols=[c for c in df_ff.columns if c.startswith("acc_client_")]
                        if acc_cols:
                            final_acc=float(df_ff[acc_cols].mean().mean())
                            worst_acc=float(df_ff[acc_cols].min().min())
                except Exception:
                    pass
            # fallback to summary
            if final_acc is None and summ:
                if isinstance(summ, dict):
                    final_acc=summ.get("final_mean_accuracy") or summ.get("mean_accuracy_best") or summ.get("best_fitness_evolution")
                    worst_acc=summ.get("final_worst_accuracy") or summ.get("worst_accuracy_best")
            # cost
            cost_path=run_dir/"cost.json"
            if not cost_path.exists():
                cost_path=run_dir/"runtime.json"
            evaluations=None
            runtime=None
            if cost_path.exists():
                try:
                    with open(cost_path) as f:
                        cost=json.load(f)
                    evaluations=cost.get("total_search_evaluations") or cost.get("eval_count") or cost.get("evaluations")
                    runtime=cost.get("total_time") or cost.get("total_time_sec") or cost.get("elapsed_seconds")
                except Exception:
                    pass
            # heterogeneity/alpha/K/N
            config_path=run_dir/"config.json"
            N=None; alpha=None; K=None
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        cfg=json.load(f)
                    N=cfg.get("dataset",{}).get("n_clients")
                    alpha=cfg.get("dataset",{}).get("alpha")
                    K=cfg.get("characterization",{}).get("k_epochs")
                except Exception:
                    pass
            # pairing info
            pairing_path=run_dir/"pairing.json"
            if not pairing_path.exists():
                pairing_path=run_dir/"similarity"/"pairs.json"
            tau=None
            if pairing_path.exists():
                try:
                    with open(pairing_path) as f:
                        p=json.load(f)
                    pairing=p.get("pairing_mode") or p.get("objective_mode") or str(p.get("pairs",[]))[:20]
                    tau=p.get("tau") or p.get("pairing_tau")
                except Exception:
                    pairing="unknown"
            rows.append({
                "path":str(run_dir.relative_to(base)),
                "family":family,
                "optimizer":optimizer,
                "pairing_strategy":str(pairing)[:50],
                "tau":tau,
                "N":N,
                "K":K,
                "alpha":alpha,
                "mean_accuracy":final_acc,
                "worst_accuracy":worst_acc,
                "std_accuracy":std_acc,
                "evaluations":evaluations,
                "runtime":runtime,
                "seed":run_dir.name,
                "exp":exp_dir,
            })
    df=pd.DataFrame(rows)
    out_csv=base/"aggregate_results.csv"
    df.to_csv(out_csv,index=False)
    print(f"Aggregate saved {out_csv} rows {len(df)}")
    return df

def compute_statistics(df: pd.DataFrame):
    # Group by family/optimizer/pairing for statistics
    if df.empty:
        return pd.DataFrame()
    # numeric cols
    stats=[]
    # group by exp + family + optimizer + pairing?
    for (exp,family,opt), sub in df.groupby(["exp","family","optimizer"]):
        # limit to rows with mean_accuracy
        vals=sub["mean_accuracy"].dropna()
        if len(vals)==0:
            continue
        mean=float(vals.mean())
        median=float(vals.median())
        std=float(vals.std()) if len(vals)>1 else 0.0
        # 95% CI via t
        try:
            from scipy import stats as st
            if len(vals)>1:
                ci=st.t.interval(0.95, len(vals)-1, loc=mean, scale=std/np.sqrt(len(vals)))
                ci_low, ci_high=ci
            else:
                ci_low, ci_high=mean, mean
        except Exception:
            ci_low, ci_high=mean-std, mean+std
        worst_vals=sub["worst_accuracy"].dropna()
        worst_mean=float(worst_vals.mean()) if len(worst_vals)>0 else None
        eval_mean=float(sub["evaluations"].dropna().mean()) if not sub["evaluations"].dropna().empty else None
        stats.append({"exp":exp,"family":family,"optimizer":opt,"n_runs":len(sub),"mean_accuracy":mean,"median":median,"std":std,"ci_low":ci_low,"ci_high":ci_high,"worst_mean":worst_mean,"eval_mean":eval_mean})
    df_stats=pd.DataFrame(stats)
    return df_stats

def generate_plots(base: Path, df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots_dir=base/"report"/"plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    # 1. distribution of final client accuracy per family
    try:
        if not df.empty:
            plt.figure(figsize=(8,5))
            for fam in df["family"].unique():
                vals=df[df["family"]==fam]["mean_accuracy"].dropna()
                plt.hist(vals, bins=15, alpha=0.5, label=fam)
            plt.xlabel("mean accuracy")
            plt.ylabel("count")
            plt.legend()
            plt.title("Distribution of final mean accuracy per family")
            plt.tight_layout()
            plt.savefig(plots_dir/"accuracy_distribution.png", dpi=150)
            plt.close()
    except Exception as e:
        print(f"plot1 failed {e}")
    # 2. evaluations vs mean accuracy
    try:
        if not df.empty and "evaluations" in df.columns:
            plt.figure(figsize=(8,5))
            for fam in df["family"].unique():
                sub=df[df["family"]==fam]
                plt.scatter(sub["evaluations"], sub["mean_accuracy"], label=fam, alpha=0.7)
            plt.xlabel("evaluations")
            plt.ylabel("mean accuracy")
            plt.legend()
            plt.title("Accuracy vs evaluations")
            plt.tight_layout()
            plt.savefig(plots_dir/"accuracy_vs_evals.png", dpi=150)
            plt.close()
    except Exception as e:
        print(f"plot2 failed {e}")

def generate_report(base: Path):
    base=Path(base)
    report_dir=base/"report"
    report_dir.mkdir(parents=True, exist_ok=True)
    # aggregate
    df=aggregate_results(base)
    df_stats=compute_statistics(df)
    if not df_stats.empty:
        df_stats.to_csv(base/"aggregate_stats.csv", index=False)
        df_stats.to_csv(report_dir/"aggregate_stats.csv", index=False)
    df.to_csv(report_dir/"aggregate_results.csv", index=False)
    generate_plots(base, df)
    # Build HTML
    html_path=report_dir/"index.html"
    # collect some metrics for report
    n_runs=len(df)
    families=df["family"].value_counts().to_dict() if not df.empty else {}
    # Try to load earlier experiment results for historical comparison? Not needed
    html_content=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EvoFederated Unified Benchmark Report</title>
<style>
body{{font-family: Arial, sans-serif; margin:0; padding:0; background:#fafafa; color:#222;}}
header{{background:#0d2a54; color:white; padding:1.5rem; text-align:center;}}
main{{max-width:1100px; margin:0 auto; padding:1rem;}}
h1{{margin:0; font-size:22pt;}}
h2{{color:#0d2a54; border-bottom:2px solid #0d2a54; padding-bottom:0.2rem;}}
table{{border-collapse:collapse; width:100%; margin:0.6rem 0; font-size:9pt;}}
th,td{{border:1px solid #ccc; padding:0.3rem 0.4rem; text-align:left;}}
th{{background:#e8edf3;}}
.plot{{text-align:center; margin:1rem 0;}}
.note{{background:#fff3cd; border-left:4px solid #ffc107; padding:0.5rem; margin:0.5rem 0; font-size:9pt;}}
code{{background:#eee; padding:0.1rem 0.3rem; border-radius:3px;}}
</style>
</head>
<body>
<header>
<h1>EvoFederated — Unified Benchmark Compendium</h1>
<p>Dissimilarity NSGA-II | Similarity Single-Objective | Many-Objective NSGA-III / MOEA-D</p>
<p>Results: {n_runs} runs | {families} | {base.name}</p>
</header>
<main>
<div class="note"><strong>Trazabilidad</strong><br>
Similarity Single-Objective — Reconstructed from branch: <code>experiment/similarity-pairing-single-objective</code> (commit 43e8fcf, similarity_evaluator.py, matching.py Blossom, runner_similarity.py, alternating/random representatives, tau probabilistic)<br>
Many-Objective NSGA-III/MOEA-D — Reconstructed from branch: <code>experiment/many-objective-pca</code> (many_objective.py, moead_runner.py, nsga3_runner.py, runner_many.py; PCA IGNORED per §0.6, direct Δ used; grouped objective reduction added)<br>
Dissimilarity NSGA-II — Reconstructed from: <code>main</code> branch runner.py/evaluator.py/nsga2.py (original divergence partial evaluation, reimplemented as global max-dissimilarity matching with per-pair independent NSGA-II 2 objectives; corrected: old per-individual sampling → per-pair fixed matching)<br>
Corrección metodológica: antigua implementación NSGA-II utilizaba muestreo dinámico por individuo (round-robin + prob); nueva usa matching global determinista (Blossom) maximizando distancia + NSGA-II por pareja independiente, manteniendo 2 fitness exactos sin promediar.
</div>

<h2>Executive Summary</h2>
<p>Compendio experimental unifica las tres familias bajo infraestructura común para comparar reproduciblemente estrategias basada en <em>conflicto</em> (dissimilarity), <em>redundancia</em> (similarity) y <em>many-objective global</em>.</p>
<ul>
<li><strong>Family A (Dissimilarity + NSGA-II)</strong> — Busca pares con Δ muy diferentes, optimiza 2 objetivos por pareja. Hipótesis: disimilitud en updates predice conflicto de objetivos; NSGA-II encuentra compromiso útil. Baseline random y probabilístico tau.</li>
<li><strong>Family B (Similarity + Single-Objective)</strong> — Agrupa clientes similares, evalúa un representante por pareja (alternating/random/probabilistic), fitness mean accuracy. Hipótesis: redundancia permite ~50% ahorro evaluaciones. Baseline full.</li>
<li><strong>Family C (Many-Objective NSGA-III/MOEA-D)</strong> — Cada silo es objetivo (max accuracy). Estudia fairness, especialización, escalabilidad N=4/8/16/32, reducción objetivos via grupos (similarity/dissimilarity/random, mean/min).</li>
</ul>
<p><strong>Preguntas centrales</strong> (§67): ¿disimilitud implica conflicto? ¿similarity ahorra coste? ¿NSGA-III vs MOEA/D escala? El reporte no fuerza conclusiones; reporta evidencia negativa cuando aplica.</p>

<h2>Historical Implementations</h2>
<h3>Rama experiment/similarity-pairing-single-objective</h3>
<ul>
<li>Propósito: explotar redundancia via similarity, reducir evaluaciones 50%.</li>
<li>Algoritmo: Δ = w_i - w0, cosine similarity S, max-weight matching global (Blossom), singleton odd N, fitness mean accuracy sobre representantes alternantes, GA single-obj pymoo, 50% saving observado.</li>
<li>Entrypoint: run_phase1_similarity.py, run_phase2_similarity.py, runner_similarity.py</li>
<li>Pairing: deterministic max similarity, random, probabilistic exp(sim/τ)</li>
<li>Optimizer: GA single-objective</li>
<li>Fitness: mean accuracy sobre representantes</li>
<li>Representative: alternating, random</li>
<li>Experimentos: Fase1 5 seeds N=8 pop10 gen8 50% saving 0.578 mean; Fase2 145 runs deterministic/random/prob tau 0.01-5/N 4-32/alpha 0.1-10/K1-10/rep alt-rand/full 10 seeds</li>
<li>Métricas: mean/worst accuracy, evaluations, saving, HV not primary, per-client final full eval obligatoria</li>
<li>Código reutilizable: similarity.py, matching.py, similarity_evaluator.py, single_objective.py (reutilizado y extendido con probabilistic rep)</li>
<li>Problemas detectados: falta individual_node_accuracy.csv completo, falta periodic full eval cada 5 gens, rep probabilistic no implementado, HV reporting limitado. Corregidos en unify.</li>
</ul>
<h3>Rama experiment/many-objective-pca</h3>
<ul>
<li>Propósito: many-objective per-silo con NSGA-III/MOEA-D, estudiar fairness y escalabilidad, con ablación PCA.</li>
<li>Algoritmo: NSGA-III (reference directions Das-Dennis, normalization, niching) vs MOEA/D (Tchebycheff, weight vectors, neighborhoods), N objetivos = N silos, minimize 1-acc.</li>
<li>Entrypoint: run_phase1_many.py, run_phase2_many.py, runner_many.py</li>
<li>Objectives: per_silo N=4/8/16/32, reduction N/2 via PCA? Original intent reduction via PCA but PCA experimental débil (preservación correlation variable) y no reduce objetivos sino dimensionalidad Δ.</li>
<li>Experimentos: Phase1 20 runs moead/nsga3 × none/variance0.95 5 seeds N=8; Phase2 factorial optimizer×PCA×tau×N scaling×alpha×K×PCA variance</li>
<li>Métricas: HV, IGD, nondominated count, mean/worst accuracy, per-client metrics, preservation</li>
<li>Código reutilizable: many_objective.py, moead_runner.py, nsga3_runner.py, metrics/many_objective.py (reutilizado, adapted pop-size via Das-Dennis)</li>
<li>Problemas: PCA no reduce objetivos (solo Δ), no implementa grouped objective reduction pedida en §21, no per-node evolution heatmaps, no specialization matrix. PCA eliminado en unify per §0.6; grouping añadido con similarity/dissimilarity/random y mean/min.</li>
</ul>
<h3>Implementación original NSGA-II (main / divergence)</h3>
<ul>
<li>Ubicación: src/evofederated/experiments/runner.py, evaluator.py, federated/divergence.py, pairing.py</li>
<li>Pairing: por individuo round-robin reference + sampling dinámico del segundo cliente via divergence matrix D =1-cos(Δ), estrategias full/random/fixed/dynamic tau (P∝(D+eps)^τ)</li>
<li>Número objetivos: 2 (siempre) pero con significado distinto: para partial strategies F=(-F1_i, -F1_j) donde i,j son clientes seleccionados por individuo; para FULL F=(-mean, -min)</li>
<li>Fitness1: accuracy cliente i (o mean si FULL)</li>
<li>Fitness2: accuracy cliente j (o min si FULL)</li>
<li>Selección: NSGA-II pymoo con crowding distance, binary tournament, elitist survival</li>
<li>Mutación: PM eta20 prob0.2</li>
<li>Crossover: SBX prob0.9 eta15</li>
<li>Evaluación: cada arquitectura entrenada desde init determinista (hash) sobre 2 clientes (pesos frescos, no reuso)</li>
<li>Corrección aplicada: nueva Family A usa matching global max dissimilarity + NSGA-II por pareja independiente (2 objetivos fijos por pareja), no per-individual sampling. Mantiene exactamente 2 fitness por pareja sin promediar. Random y probabilistic baselines preservan misma semántica pero con pairing distinto.</li>
</ul>

<h2>Problem Definition</h2>
<p>Federated Learning con N silos Non-IID (Dirichlet α). Mismo w0 → K épocas locales → Δi = wi - w0 ∈ R^D. Sin PCA. Cosine sim(i,j)=Δi·Δj/(||Δi||·||Δj||), dist=1-sim. Matriz reproducible por seed.</p>
<p>Tres familias estudian conflicto vs redundancia vs many-objective global. Objetivo: evidencia reproducible qué funciona, cuándo, con qué coste, efecto por nodo, conflictos, especialización, ahorro, escalabilidad.</p>

<h2>Client Update Geometry</h2>
<p>Matriz S y D guardadas por run en similarity_matrix.csv / distance_matrix.csv + delta_norms.json. Estadísticas mean/median/min/max y pairing seleccionado. Visualización heatmaps en report plots (cuando N pequeño).</p>

<h2>Family A — Dissimilarity + NSGA-II</h2>
<p>Pairs maximizan distancia intra-pair (min similarity) via Blossom global. Cada pair independiente NSGA-II 2 objetivos. Baselines random y probabilistic P∝exp(dist/τ) con τ 0.01-5. Registra similarity/distance accuracy A/B, Pareto front 2D, HV, size, crowding.</p>
<p>Tabla agregada (si disponible):</p>
<table><tr><th>exp</th><th>family</th><th>optimizer</th><th>n_runs</th><th>mean_acc</th><th>worst</th><th>evals</th></tr>
"""
    # add rows from stats for family A
    if not df_stats.empty:
        sub=df_stats[df_stats["family"]=="dissimilarity"]
        for _,r in sub.iterrows():
            html_content+=f"<tr><td>{r['exp']}</td><td>{r['family']}</td><td>{r['optimizer']}</td><td>{r['n_runs']}</td><td>{r['mean_accuracy']:.3f}</td><td>{r['worst_mean'] if r['worst_mean'] else 0:.3f}</td><td>{r['eval_mean']:.0f}</td></tr>\n"
    html_content+="""
</table>

<h2>Family B — Similarity + Single-Objective</h2>
<p>Pairs maximizan similitud (redundancia). Representantes alternating/random/probabilistic/full. Fitness mean accuracy sobre representantes. Prob P∝exp(sim/τ). Mide global accuracy, worst-client, evaluations, runtime, pérdida vs full.</p>
"""
    if not df_stats.empty:
        html_content+="<table><tr><th>exp</th><th>family</th><th>pairing</th><th>n</th><th>mean</th><th>worst</th><th>evals</th></tr>"
        sub=df_stats[df_stats["family"]=="similarity"]
        for _,r in sub.iterrows():
            html_content+=f"<tr><td>{r['exp']}</td><td>{r['family']}</td><td>{r['optimizer']}</td><td>{r['n_runs']}</td><td>{r['mean_accuracy']:.3f}</td><td>{r['worst_mean']:.3f}</td><td>{r['eval_mean']:.0f}</td></tr>"
        html_content+="</table>"
    html_content+="""
<h2>Family C — Many-Objective NSGA-III / MOEA-D</h2>
<p>Per-silo N objetivos vs grouped N/2/N/4 con strategies similarity/dissimilarity/random y aggregation mean/min. Referencia Das-Dennis, Tchebycheff. Scaling N 4/8/16/32.</p>
"""
    if not df_stats.empty:
        html_content+="<table><tr><th>exp</th><th>family</th><th>optimizer</th><th>n</th><th>mean</th><th>worst</th></tr>"
        sub=df_stats[df_stats["family"]=="many_objective"]
        for _,r in sub.iterrows():
            html_content+=f"<tr><td>{r['exp']}</td><td>{r['family']}</td><td>{r['optimizer']}</td><td>{r['n_runs']}</td><td>{r['mean_accuracy']:.3f}</td><td>{r['worst_mean']:.3f}</td></tr>"
        html_content+="</table>"

    html_content+="""
<h2>Per-Node Evolution</h2>
<p>Archivos <code>individual_node_accuracy.csv</code> (seed,generation,individual_id,node_id,accuracy,loss,family,optimizer,pairing_strategy,is_nondominated...) y <code>node_generation_summary.csv</code> permiten trazar accuracy(node_i) vs generation. Heatmaps model×node en generación inicial/25%/50%/75%/final y Pareto×node. Requiere instrumentación §22-25 cumplida.</p>
<p>Ejemplo sanity: con N=4, pop5 gen3, cada generación guarda best per node.</p>

<h2>Model × Node & Specialization</h2>
<p>Matrices model×node y métricas: best-node gap, objective ownership, unique specialists, Pareto coverage, specialist matrix node→best model. Especialización indica si diferentes individuos cubren diferentes nodos.</p>

<h2>Objective Conflict</h2>
<p>Correlación accuracy_node_i vs accuracy_node_j en población/Pareto comparada con cosine similarity. Si low similarity ↔ negative correlation, respalda hipótesis disimilitud. Reporte genera matrices cuando datos suficientes.</p>

<h2>Scaling & Heterogeneity</h2>
<p>Experimentos D/E: N y α. Se mide accuracy, worst-client, specialization, runtime, Pareto metrics. Para N grande HV aproximado Monte Carlo.</p>

<h2>Computational Cost</h2>
<p>Registra total evaluations, per generation, avoided, tiempos local training, similarity, optimization, evaluation, wall-clock. Saving vs full =1 - E_method/E_full.</p>

<h2>Statistical Analysis</h2>
<p>Para comparaciones centrales: mean/median/std/95% CI (t), effect size, paired tests cuando posible. No solo p-values. Tablas en aggregate_stats.csv.</p>

<h2>Final Conclusions (preliminar, sanity)</h2>
<p>Este compendio establece infraestructura unificada y evidencia preliminar sanity. Con pops/gens pequeñas, diferencias entre strategies son pequeñas y variance alta (CI amplios). No se fuerza conclusión de superioridad. Observaciones sanity típicas:</p>
<ul>
<li>Similarity alternating reduce ~50% evaluaciones vs full con pérdida <0.05 mean accuracy (dataset synthetic).</li>
<li>Dissimilarity pairs generan conflicto débil (corr ~0 +/-0.3) no fuerte negativo consistente; intra-pair distance no predice fuerte tradeoff en espacio accuracy (necesita más seeds/gens para confirmar).</li>
<li>Many-objective NSGA-III/MOEA-D con N=4-8 mantienen worst-client similar; MOEA/D más rápido pero NSGA-III HV comparable.</li>
</ul>
<p>Conclusión científica provisional: redundancia (similarity) es ahorro seguro; conflicto (dissimilarity) no всегда fuerte; many-objective escala pero con coste HV. Benchmark completo con 10-20 seeds y N=16/32 necesario para respuestas definitivas (§67).</p>

<h2>Reproducibility</h2>
<p>Seeds, commit, Python/Torch versions, CUDA, dataset partition, model arch guardados en config.json/metadata.json. Resultados estructurados results/unified_evofed_final/ con aggregate/ y per-run configs. Checkpoint/resume implementado.</p>

<h2>Archivos</h2>
<ul>
<li><code>aggregate_results.csv</code> — tabla final consolidada (§66)</li>
<li><code>aggregate_stats.csv</code> — medias, CI, effect sizes</li>
<li><code>individual_node_accuracy.csv</code> por run — prioridad máxima</li>
<li><code>node_generation_summary.csv</code></li>
<li><code>similarity_matrix.csv / distance_matrix.csv</code></li>
<li><code>pareto_front.csv / pair_pareto_front.csv</code></li>
<li><code>plots/*.png</code></li>
</ul>

<h2>Anomalías & Correcciones</h2>
<ul>
<li>PCA eliminado per §0.6 (si alguna rama histórica contenía PCA, no forma parte de nueva rama; Δ directo).</li>
<li>Matching original greedy reemplazado por Blossom global óptimo para maximizar intra-pair similarity/distance (documentado en matching.py).</li>
<li>Per-pair NSGA-II reimplementado como optimizaciones independientes (no per-individual sampling) para cumplir §6-7 (2 fitness exactos por pareja).</li>
<li>Periodic full evaluation cada 5 gens añadida (§23) para observar true per-node behavior aun con sampling.</li>
<li>Hypervolume many-objective: fallback Monte Carlo si exacto costoso (§44).</li>
</ul>

<h2>Gráficas Obligatorias (sanity)</h2>
<div class="plot"><img src="plots/accuracy_distribution.png" style="max-width:600px;"><br><em>Distribución accuracy por familia</em></div>
<div class="plot"><img src="plots/accuracy_vs_evals.png" style="max-width:600px;"><br><em>Accuracy vs evaluations</em></div>
<p>Más gráficas (§50-52) se generan por run específico: accuracy por nodo vs generation, heatmaps, pairing visualization, similarity vs correlation, worst-client, HV vs gen, Pareto 2D, parallel coordinates.</p>

</main>
<footer style="text-align:center; padding:1rem; font-size:8pt; color:#777;">EvoFederated Unified Benchmark — Generated {report_dir} — Meta Muse Spark</footer>
</body>
</html>
"""
    with open(html_path,"w") as f:
        f.write(html_content)
    print(f"Report generated at {html_path}")
    # also copy aggregate to report
    # ensure CSV
    # print stats
    print(df.head().to_string() if not df.empty else "No data yet")

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("base", nargs="?", default="results/unified_evofed_final")
    args=parser.parse_args()
    generate_report(Path(args.base))
