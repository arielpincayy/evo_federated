# EvoFederated — NAS evolutivo en FL Non-IID

Framework experimental para estudiar NAS evolutiva en aprendizaje federado Non-IID, evaluando arquitecturas solo sobre subconjuntos de clientes seleccionados por divergencia de dinámicas locales (vectores Δ).

Hipótesis: `Quality_dynamic ≈ Quality_full` mientras `Cost_dynamic << Cost_full`. Comparación de hipervolumen vs evaluaciones.

## Stack
Python, PyTorch, torchvision, pymoo (NSGA-II), NumPy/Pandas, Matplotlib, scikit-learn, scipy, pyyaml

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# o
pip install -r requirements.txt
```

## Estructura
```
src/evofederated/
  data/          # datasets, Dirichlet partition
  models/        # genoma MLP, decoder
  clients/       # FederatedClient / Server (sin fuga de datos)
  federated/     # caracterización Δ, matriz D, pareamiento
  evolution/     # Evaluator + NSGA-II + hypervolume
  metrics/       # HV, coste, saving
  experiments/   # runner, pilot, plots
  utils/         # seed, config
configs/
  main.yaml      # Fashion-MNIST completo (10 clients, 10 gens)
  small.yaml     # synthetic smoke (6 clients, 3 gens)
  ablation.yaml  # ejes de ablation
run_experiment.py
run_pilot.py
run_ablation.py
results/
```

## Genoma
`g = [L, n1..nLmax, act, dropout, bn]` longitud fija. `Genome.from_vector` repara mutaciones. Decoder → MLP con BatchNorm opcional. Todas las arquitecturas válidas.

## Uso

### Piloto (sanity check de dificultad)
```bash
python run_pilot.py --config configs/small.yaml --n-archs 8
# o
python run_pilot.py --config configs/main.yaml
```
Genera `results/pilot/` con matriz arquitectura×cliente, heatmaps, `pilot_summary.json`. Verifica variabilidad y ranking changes. Si `is_trivial=true`, ajustar dataset/α/espacio.

### Experimento completo
```bash
python run_experiment.py --config configs/main.yaml
python run_experiment.py --config configs/small.yaml
python run_experiment.py --config configs/main.yaml --output results/my_run
```

Estrategias: `full` (exhaustivo R·G·N), `random`, `fixed` (most-divergent fijo), `dynamic` (probabilístico τ). Todas con mismo dataset/partición/seed/espacio/población/generaciones para comparación justa.

Mergea ahorro: `Saving = 1 - E_method/E_full`.

### Ablation
```bash
python run_ablation.py --config configs/ablation.yaml --dry-run
python run_ablation.py --config configs/ablation.yaml --axes alpha tau
```

### Datasets
- `fashion-mnist` (recomendado: balance dificultad/coste). CIFAR-10 disponible pero más costoso.
- `cifar10` (cambiar `dataset.name`)
- `synthetic` (make_classification, parametrizable) para smoke/pilot rápido.

Partición Dirichlet α∈{10,1,0.5,0.1,0.05}. Gráficas: samples/cliente, dist clase, heatmap, heterogeneidad (CV, TV).

## Pipeline conceptual
```
Dataset → Partición Dirichlet → Clientes C1..CN
→ Modelo sonda θ0 (misma arch + pesos) → Train k épocas → Δi = θik-θ0
→ Matriz D coseno/L2 → Selección cliente divergente (random/fixed/dynamic τ)
→ Población genomas → Eval parcial (Fi,Fj) multiobjetivo (-F1i,-F1j)
→ NSGA-II → HV_t por generación → Validación exhaustiva Pareto → HV común + coste
```

## Métricas
- Rendimiento: mean/median/min/std F1 & accuracy (macro-F1 prioritario).
- Robustez: min F1, std.
- NAS: Frente Pareto + Hipervolumen (pymoo). `hv_ref_min` fijo (ej. [0,0] → [0.1,0.1] interno). HV interno vs HV validación común (mean/min exhaustivo).
- Coste: evals totales, tiempo, saving.
- Proxy: correlación Spearman/Kendall/Pearson entre D y TV verdadero (offline).

## Resultados persistentes `results/experiment_YYYYMMDD_HHMMSS/`
```
config.json/.yaml
metadata.json (seed, device, versiones, hv_ref)
dataset/{class_distribution.csv, heterogeneity.json}
characterization/{divergence_matrix.npy/.csv, divergence_stats.json, delta_norms.json, proxy_validation.json, true_TV_matrix.npy}
generations/{hypervolume_*.csv, pair_history_*.csv, hypervolume_history.csv, population_history.csv}
architectures/{pareto_*.csv, final_pareto.csv}
strategy_*/{hypervolume_history.csv, pair_history.csv, population_history.csv, final_pareto.csv, exhaustive_validation.csv, hv_common.json, cost.json}
hypervolume_history.csv (gen, strategy/method, seed, hv/HV, hv_type, ref_point, eval_count/cumulative_evals, elapsed/cumulative_time)
population_history.csv
pair_history.csv
evaluations.csv (alias)
final_pareto.csv
exhaustive_validation.csv
summary.csv (hv + saving)
plots/*.png/.pdf (≥21 gráficas)
```

## Hipervolumen
- **Interno/evolutivo**: sobre (F1_i,F1_j) o (mean,min) para FULL. Distintos espacios → no comparar ingenuamente.
- **Validación común**: recalculado sobre (mean_f1, min_f1) exhaustivo con mismo ref para comparación científica.
- Gráficas: HV vs generación, HV vs evals, HV vs tiempo. Referencia documentada y fija.

## Reproducibilidad
Seeds para Python/NumPy/PyTorch/Dirichlet/pymoo/model init. Config y metadata guardadas. Ejecutar con misma config → mismo resultados (deterministic cudnn).

## Tests
```bash
pytest tests -v
```
Cubre: partición Dirichlet, genoma válido, divergencia (simetría, Dii=0), pairing (nunca i=j), Δ dims, HV orientación, no contaminación de pesos, no fuga datos.

## Sanity checks piloto
Diferencias `Acc(A1,Ci) >> Acc(A2,Ci)` y cambios de ranking `Acc(A1,Ci)>Acc(A2,Ci)` pero `Acc(A1,Cj)<Acc(A2,Cj)`. Si dataset trivial, aumentar dificultad o cambiar a Fashion-MNIST.

## Notas de diseño
- MLP `Lmax` controlado; hiperparámetros de entrenamiento fijos para atribuir diferencias a arquitectura.
- Evaluación en 2 clientes contamina? No: pesos frescos + misma init determinista por genoma, registrados.
- Validaciones exhaustivas no intervienen en fitness.
- FULL optimiza (mean,min) para mantener 2 objetivos; documentado para HV comparability.

## Limitaciones & futuro
- Espacio MLP limitado; extender a CNN/NAS-Bench.
- NSGA-II principal; MOEA/D/SPEA2 modularizable.
- Synthetic vs FMNIST: documentar elección por coste.
- Muchas seeds para HV CI no incluidas en smoke por presupuesto.

## Licencia
MIT.
