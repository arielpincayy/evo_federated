# EvoFederated — NAS evolutivo federado con muestreo estratificado por clusters (CS-NAS)

Búsqueda evolutiva de arquitecturas (frente Pareto accuracy vs nº de parámetros) en FL non-IID con **muchos nodos**.
Cada arquitectura se evalúa con solo **un representante por cluster**; los clusters se forman una vez por similitud de
las actualizaciones (Δ) de un modelo sonda, y los representantes se re-sortean en cada generación. El coste de una
evaluación depende del número de clusters C, no del número de nodos N.

Decisiones de diseño, métricas y la regla de decisión experimental: [`docs/PROTOCOL.md`](docs/PROTOCOL.md).
La revisión sistemática del área está en `manuscript/main.pdf` y el informe experimental final (resultado mayormente negativo) en `manuscript/informe.pdf`. El código y los resultados de las propuestas anteriores
(disimilitud/similitud con 8 nodos, many-objective) siguen en el tag `legacy-final` de git.

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest tests -q
```

## Uso
```bash
# smoke (Fashion-MNIST, 30 nodos, 3 generaciones)
python run_cs.py --config configs/cs_smoke.yaml
# configuración principal (CIFAR-10, 100 nodos)
python run_cs.py --config configs/cs_cifar10.yaml
# baselines y ablaciones por override
python run_cs.py --config configs/cs_cifar10.yaml --set sampling.sampler=random evolution.search=random
# E1+E2: calidad de clusters y del estimador sobre arquitecturas aleatorias
python run_estimator_study.py --config configs/cs_cifar10.yaml --output runs/estimator --seeds 0 1 2
```
Samplers: `cluster` (Δ), `label_cluster` (oracle con etiquetas, no privado), `random_cluster`, `random`, `full`.

## Salidas de una corrida (`runs/<nombre>/`)
`config.json` (configuración y semillas), `population.csv` (cada evaluación, con acc estimada y acc en todos los
nodos), `generation_metrics.csv` (HV, error del estimador, éxito de hijos, diversidad), `clusters.json`,
`final_front.csv` (frente final reentrenado y medido en test), `cost.json` (entrenamientos del probe, búsqueda,
monitoreo y final por separado).

## Estructura
```
src/evofederated/
  data/        # datasets, partición Dirichlet, test oficial por nodo
  clients/     # FederatedClient / Server (sin datos crudos fuera del nodo)
  federated/   # FedAvg, probe de Δ, clustering
  models/      # genoma MLP (tests) y genoma CNN
  evolution/   # evaluador por representantes y bucle NSGA-II / random search
  experiments/ # runner de una corrida
  metrics/     # hipervolumen, coste
tests/  configs/  docs/  manuscript/
```
