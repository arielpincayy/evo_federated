# Resultados: ¿el ranking de arquitecturas cambia según el nodo? (FEMNIST, modelo global FedAvg) (2026-11-13)

Diseño preregistrado: `docs/ARCH_NODE_DESIGN.md`. Scripts: `run_arch_node_study.py`, `analyze_arch_pick.py`; estadísticos en `src/evofederated/metrics/arch_node.py`. Salidas en `runs/arch_node/` (no versionado).
24 arquitecturas aleatorias del espacio v2 × 3 federaciones (semillas 0–2) × 2 réplicas de entrenamiento = 144 modelos (20 rondas × 2 épocas, Adam lr=1e-3, elegido con un piloto en la semilla 100: 28.2 % frente a 22.1 % de lr=1e-2), medidos por nodo en val y en test.

## Interacción arquitectura × nodo (media de 3 federaciones)
| | accuracy | logit |
|---|---|---|
| SD del efecto de **arquitectura** | 18.1 pp | 1.15 |
| SD del efecto de **nodo** | 7.4 pp | 0.47 |
| SD de la **interacción estable** | **4.7 pp** | 0.17 |
| fracción de la variación residual que se repite | 49 % | 20 % |
| IC95 de V > 0 | 3/3 semillas | 3/3 |
| p por permutación (10 000) | 0.0001 en cada semilla (mínimo posible) | 0.0001 |

- Existe una interacción estable (se repite con otro entrenamiento y otra muestra del mismo nodo), también en logit, pero es pequeña frente al efecto de la arquitectura (su varianza es ≈7 % de la de la arquitectura). Parte del efecto en accuracy es de escala (49 % → 20 % en logit).
- El ranking de arquitecturas es casi el mismo en todos los nodos: Spearman 0.97 entre réplicas de entrenamiento y 0.99 entre federaciones; entre el ranking de un nodo y el de otro, 0.92.

## Consecuencia práctica: elegir por nodo o por cluster no gana a la mejor global (validación cruzada)
| | ganancia media | semillas con IC95 > 0 |
|---|---|---|
| por nodo | −1.25 pp | 0/3 |
| por cluster Δ | −0.52 pp | 0/3 |
Con ~36–54 muestras por nodo, la señal de la interacción es más débil que el ruido de seleccionar.

## ¿Hace falta elegir arquitectura? Sí, pero basta evaluar con pocos nodos
- Dispersión entre arquitecturas (accuracy media sobre los 100 nodos): peor ≈5 %, mediana ≈25 %, mejor ≈58 %. Solo 1 arquitectura queda a menos de 2 pp de la mejor y 2 a menos de 5 pp.
- Elegir la arquitectura con k nodos (val de una réplica) y evaluarla en el resto (test de la otra): con 1 nodo se recupera el 91 % de la mejora posible sobre el azar (pierde 2.8 pp frente a la mejor), con 3 el 96.5 % (−1.1 pp), con 10 el 98.5 % (−0.5 pp) y con 30 el 99.9 % (−0.04 pp).
- Accuracy y parámetros: Spearman 0.51; 8 de las 24 están en el frente de Pareto (de 8 062 params/5.4 % a 69 214 params/57.9 %); la mejor usa 69 mil de los 354 494 del espacio.

## Límites
24 arquitecturas aleatorias del espacio v2, una receta y un dataset; test por nodo muy pequeño (~36 muestras); el IC bootstrap remuestrea nodos, no arquitecturas; no se identificó qué arquitecturas o nodos generan la interacción.
No compara una búsqueda evolutiva con "mejor de N aleatorias" (ver `docs/RESULTS_P3.md` para una comparación con presupuesto igual).
