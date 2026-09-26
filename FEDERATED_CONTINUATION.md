# Continuación: NAS evolutivo federado con FedAvg

## 1. Objetivo del estudio

Encontrar arquitecturas de red que, entrenadas como **un único modelo global con FedAvg**,
funcionen bien en promedio y sean robustas en el peor cliente, en FL no-IID.

- Cada individuo de la búsqueda = una arquitectura (genoma MLP reparado).
- Cada evaluación = entrenar un modelo global con rondas FedAvg y medirlo en validación.
- La búsqueda optimiza **media y mínimo entre clientes** y devuelve un frente Pareto,
  no un "óptimo" único.
- Las propuestas existentes deben re-probarse dentro de este escenario:
  - **A**: disimilitud de updates + NSGA-II.
  - **B**: similitud + GA / representantes.
  - **C**: many-objective NSGA-III / MOEA/D + reducción por grupos.
- Los 220 runs agregados y el paper actual **no son evidencia válida** para esta tesis:
  entrenan un modelo independiente por cliente y la búsqueda consulta `test`.

## 2. Estado actual (lo ya implementado)

### Núcleo FedAvg

- `src/evofederated/clients/server.py`
  - `fedavg_aggregate()`: promedio ponderado por `n_train`.
- `src/evofederated/clients/client.py`
  - `train_from_params()`: recibe pesos globales, entrena local, devuelve params.
  - Requiere `import numpy as np` (ya añadido).
- `src/evofederated/federated/fedavg.py` (nuevo)
  - `fedavg_mean()`, `run_fedavg()`, `evaluate_global_on_split()`.
- `src/evofederated/evolution/federated_evaluator.py` (nuevo)
  - `FederatedEvaluator`: genoma -> FedAvg -> `(-media_val, -min_val)`.
  - Contadores separados: `eval_count`, `n_local_trainings`, `n_rounds_total`.
  - `search_split="val"` por defecto.

### Separación train/val/test

- Búsqueda usa `split="val"` en:
  - `src/evofederated/evolution/evaluator.py`
  - `src/evofederated/evolution/dissimilarity_evaluator.py`
  - `src/evofederated/evolution/similarity_evaluator.py`
  - `src/evofederated/evolution/many_objective.py`
  - `src/evofederated/evolution/grouped_many_evaluator.py`
- `src/evofederated/evolution/validation.py`
  - `exhaustive_evaluate_genome(..., split="test")`
  - `exhaustive_evaluate_pareto(..., split="test")`
  - Final usa `test`; no usar para seleccionar.
- `src/evofederated/data/federated_split.py` (nuevo)
  - `build_federated_clients(cfg)`.
  - Train oficial -> Dirichlet -> train/val por cliente.
  - Test oficial separado -> test por cliente (semilla + 7777).
  - Sintético: train y test generados con semillas distintas.
  - **Aviso git**: la ruta está bajo el ignore `data/` del repo.
    Hay que versionarla con `git add -f src/evofederated/data/federated_split.py`.

### Configuración

- `src/evofederated/utils/config.py`
  - Nueva `FederatedConfig`: `rounds`, `clients_per_round`, `local_epochs`,
    `aggregation="fedavg"`, `search_split="val"`, `final_split="test"`.
  - `ExperimentConfig` incluye `federated`.
  - `load_config()` acepta yamls viejos sin sección `federated`.
- `configs/small.yaml` ya tiene sección `federated`.

### Runner federado unificado

- `src/evofederated/experiments/runner_federated.py` (nuevo)
  - `build_participant_fn()`: `full`, `random-k`, `similarity`, `dissimilarity`,
    `prob-similarity`, `prob-dissimilarity`.
  - `similarity` = pares max-similitud + representantes alternantes por ronda.
  - `dissimilarity` = pares max-disimilitud + representantes alternantes.
  - `prob-*` usan pareo probabilístico con `tau` y lo persisten en `participation.json`.
  - `FederatedManyEvaluator`: N objetivos `1-acc` del mismo modelo global (idea C).
  - `run_federated_nas(cfg, strategy, optimizer, tau, output_dir)`:
    - Optimizadores: `nsga2`, `random`, `nsga3`, `moead`.
    - Selección top-3 en validation por media y mínimo.
    - Final: re-entrena con participación completa y evalúa en `test`.
    - Guarda `participation.json`, `population_history.csv`, `hypervolume.csv`,
      `final_evaluation/final_test.csv`, `cost.json`.

### Tests y smoke verificados

- `tests/test_fedavg.py` (nuevo): media ponderada, reproducibilidad con seed,
  `search_split=val`, efecto de `tau`.
- Suite completa en verde (35 tests).
- Smoke sintético 4 clientes, pop 4, 2 gens, 1 ronda:
  - `full/nsga2`: 8 evals, 32 trainings.
  - `random-k/similarity/dissimilarity/nsga2`: 8 evals, 16 trainings (ahorro 50%).
  - `prob-*` registran `tau`; `full/random` baseline OK.

## 3. Cómo reproducir

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -c "import evofederated.experiments.runner_federated as r; print('import ok')"
```

Smoke mínimo:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from evofederated.utils.config import load_config
from evofederated.experiments.runner_federated import run_federated_nas
cfg = load_config('configs/small.yaml')
cfg.dataset.n_clients = 4
cfg.dataset.synthetic_n_samples = 600
cfg.evolution.pop_size = 4
cfg.evolution.n_generations = 2
cfg.federated.rounds = 1
cfg.federated.clients_per_round = 2
cfg.federated.local_epochs = 1
for strat in ["full", "random-k", "similarity", "dissimilarity"]:
    res = run_federated_nas(cfg, strategy=strat, optimizer="nsga2",
                            output_dir=f"/tmp/opencode/fed_{strat}")
    print(strat, res["cost"])
PY
```

## 4. Lo que falta

### 4.1 Cierre técnico pendiente

1. Versionar `src/evofederated/data/federated_split.py` con `git add -f`.
2. Verificar agregación de BatchNorm en FedAvg y documentar la regla elegida.
3. Verificar que ningún camino de búsqueda consulta `test`:
   buscar `split="test"` y clasificar cada uso en búsqueda vs final.
4. Verificar propagación de `tau` de config a pareo y a `participation.json`
   para todas las variantes probabilísticas.
5. Añadir test de que `test` no se lee durante `run_federated_nas()` antes del final
   y test de igualdad de presupuesto entre estrategias comparadas.

### 4.2 Etapa 0: smoke metodológico completo

- Sintético pequeño, 4 clientes, varias semillas rápidas.
- Comprobar promedio ponderado esperado, mismo init por ronda, sin reutilización
  de pesos entre arquitecturas, búsqueda solo en validation, test intacto,
  A/B/C completan con artefactos persistidos.

### 4.3 Etapa 1: comparación principal de optimizadores

- Dataset: Fashion-MNIST, 8 clientes, `alpha=0.5`.
- Mismo presupuesto: población, candidatos, rondas FedAvg, epochs locales.
- Métodos: NSGA-II, NSGA-III, MOEA/D, búsqueda aleatoria, MLP fija.
- Misma función objetivo media/mínimo en validation.
- Réplicas emparejadas por semilla/partición.
- Hacer piloto de variabilidad y coste; fijar nº final de semillas con esos datos.

### 4.4 Etapa 2: re-probar A/B/C y participación

Con la configuración principal:

- FedAvg completo.
- Selección aleatoria de `k`.
- Selección por similitud.
- Selección por disimilitud.
- Propuestas A, B y C adaptadas a FedAvg.
- Mismo `k`, mismos local epochs y presupuesto equivalente de entrenamientos.
- Reportar coste real, no solo fórmula teórica.

### 4.5 Etapa 3: heterogeneidad y sensibilidad

- Repetir el subconjunto validado en `alpha=0.1` y, si hay presupuesto, `alpha=10`.
- Después, ablaciones de `K` del probe y `tau`, solo tras comprobar que `tau`
  cambia la selección cuando debería.

## 5. Métricas y análisis a reportar

### Calidad final en test

- Media y mínimo de accuracy.
- Media y mínimo de macro-F1.
- Mediana, desviación entre clientes, brecha mejor-peor.

### Frente y coste

- Frente media-peor cliente para todos los enfoques.
- Hypervolume con referencia común fijada antes del análisis.
- No comparar directamente HV de espacios con distinta dimensión.
- Coste separado: entrenamientos locales, rondas, comunicación,
  caracterización probe, evaluación final, tiempo total.
- Curvas calidad frente a coste acumulado.

### Estadística

- Diferencias emparejadas por semilla/partición con intervalos.
- Una métrica primaria y una hipótesis principal predefinidas.
- Hipótesis secundaria: similitud/disimitud vs aleatorio a igual coste.
- No tratar arquitecturas o generaciones como réplicas independientes.

## 6. Estructura prevista del paper

1. Introducción: arquitecturas globales robustas para FL no-IID.
2. Problema: genoma, FedAvg interior, objetivos entre clientes.
3. Algoritmo: NAS exterior + entrenamiento federado + pseudocódigo.
4. Métodos: baselines y adaptación FedAvg de A/B/C.
5. Diseño: datasets, particiones, semillas, presupuestos, train/val/test.
6. Resultados: Pareto, robustez, coste, heterogeneidad.
7. Discusión: qué aporta la evolución y qué aportan las estrategias de clientes.
8. Limitaciones: simulación, MLP, coste, validez externa.

Figuras: ciclo NAS-FedAvg, distribución de clientes, calidad/HV vs coste,
frentes media-peor en test, resultados por cliente.
Tablas: configuración común, optimizadores, participación controlada, ablaciones.

## 7. Decisiones abiertas

- Métrica primaria: media/mínimo de accuracy como fitness; macro-F1 secundario.
  Fijarlo antes de la campaña.
- Nº final de semillas: decidirlo tras el piloto de variabilidad.
- Regla BatchNorm en agregación FedAvg.
- `clients_per_round`, `rounds` y `local_epochs` por defecto para Fashion-MNIST.
- `Paper/evofederated_paper.tex` describe el estudio viejo; reescribirlo solo
  con números FedAvg nuevos.
