# Resultados P3: k óptimo + NSGA-III con un objetivo por cluster (accuracy penalizada por tamaño) (2026-11-13)

Diseño preregistrado: `docs/P3_DESIGN.md`. FEMNIST (partición natural), espacio de arquitecturas v2, 3 brazos × 3 semillas de datos (0, 1, 2), 9/9 corridas terminadas.
Scripts: `run_p3.sh`, `analyze_k_search.py`, `analyze_p3.py`. Tablas en `runs/p3/` (`per_run.csv`, `paired_contrasts.csv`, `k_search.csv`; no versionadas, se regeneran con los scripts).

## 1. k óptimo: la silueta elige k=2 en las 20 federaciones
Paso 0 (informativo, con las A de `runs/transfer_femnist`): silueta coseno de k-means esférico sobre Δ y gap de transferencia por k (media de 20 federaciones).
| k | 2 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| silueta | **0.291** | 0.234 | 0.118 | 0.089 | 0.071 | 0.059 | 0.039 |
| gap de transferencia | 0.055 | 0.053 | 0.039 | 0.037 | 0.035 | 0.034 | 0.033 |

La silueta decrece de forma monótona, así que la regla preregistrada da siempre k=2. En la práctica `p3` fue NSGA-III con **2 objetivos** (60 direcciones de referencia), no con hasta 10.
La idea de un objetivo por cluster con muchos clusters no se puso a prueba. Es coherente con DBSCAN, que también encontraba ≈2 clusters (`docs/RESULTS_VALIDACION_CLUSTERS.md`).

## 2. Brazos y coste
`p3` (k=2, NSGA-III, `acc_c − λ·tamaño` con λ=1, 420 evaluaciones), `rs` (mismos clusters y objetivos, búsqueda aleatoria, 420 evaluaciones), `base` (C=10, NSGA-II con (accuracy media, tamaño), 390 evaluaciones).
Receta: 20 rondas × 2 épocas, Adam lr=1e-3; cada modelo del hall of fame se reentrena una vez con los 100 nodos y se mide en test.

## 3. Resultado final (media de 3 semillas; test, modelos reentrenados)
| | `p3` | `base` | `rs` |
|---|---|---|---|
| hipervolumen (−accuracy, tamaño) | 0.515 | 0.518 | 0.499 |
| mejor `acc − λ·tamaño` | 0.415 | 0.423 | 0.398 |
| accuracy del mejor modelo | 0.514 | 0.516 | 0.498 |
| tamaño medio del conjunto (normalizado) | 0.078 | 0.060 | 0.092 |
| modelos en el conjunto final | 12 | 4–7 | 12 |
| entrenamientos locales en la búsqueda | 16 800 | 78 000 | 16 800 |
| tiempo por corrida (h) | 0.98 | 1.28 | 0.95 |

Contrastes pareados por semilla (diferencia media; IC95 bootstrap sobre 3 semillas, descriptivo):
| contraste | métrica | media | IC95 | por semilla |
|---|---|---|---|---|
| `p3 − base` | hipervolumen | −0.0024 | [−0.042, +0.019] | +0.019, −0.042, +0.016 |
| `p3 − base` | mejor `acc − λ·tamaño` | −0.0072 | [−0.016, +0.0003] | −0.016, +0.0003, −0.006 |
| `p3 − base` | accuracy del mejor | −0.0016 | [−0.051, +0.026] | +0.021, −0.051, +0.026 |
| `p3 − rs` | hipervolumen | +0.0162 | [−0.032, +0.049] | +0.049, +0.032, −0.032 |
| `p3 − rs` | mejor `acc − λ·tamaño` | +0.0170 | [−0.005, +0.050] | +0.050, −0.005, +0.006 |
| `p3 − rs` | accuracy del mejor | +0.0158 | [−0.040, +0.050] | +0.050, +0.038, −0.040 |

Todos los IC incluyen 0: por la regla preregistrada no se declara ventaja de ningún brazo. Ni siquiera en `acc − λ·tamaño`, el criterio que optimiza `p3`, supera a `base`.
Lo único claro es el coste: `p3` usa 4.6 veces menos entrenamientos locales (2 representantes por evaluación frente a 10) con resultados que no se distinguen (3 semillas).

## 4. ¿Hubo evolución? Sí en la población, poco en el mejor individuo
Calidad media de los individuos **nuevos** por generación (solo hijos desde la 2.ª generación), `S = accuracy estimada − tamaño`, media de 3 semillas:
| generación | `p3` | `rs` | `base` |
|---|---|---|---|
| 0 | 0.025 | 0.007 | 0.105 |
| 1 | 0.040 | −0.006 | |
| 2 | 0.146 | 0.051 | |
| 3 | 0.127 | 0.010 | 0.306 |
| última | 0.127 | 0.010 | 0.282 (gen. 19) |

- Mejora de la primera a la última generación: `p3` +0.10 (positiva en las 3 semillas: +0.13, +0.08, +0.10), `rs` +0.004, `base` +0.18 (3/3; casi todo en las 3 primeras generaciones, con tamaño medio de 0.12 a 0.06).
- Diferencia `p3 − rs` en individuos nuevos (mismos nodos representantes en ambos brazos en cada generación): sube de ≈0.02 (g0) a ≈0.12 (g3); por semilla en g3: +0.19, +0.06, +0.10.
- El mejor `S` acumulado sube parecido en `p3` (0.314 → 0.414) y en `rs` (0.28 → 0.37): parte de la subida de `rs` es solo tener más muestras aleatorias. Por eso el hipervolumen final no se distingue.
- Con arquitecturas aleatorias malas en promedio (accuracy estimada 0.14 en la generación 0), unos cientos de muestras al azar ya incluyen alguna muy buena: la evolución sube la proporción de buenas, pero ayuda poco a encontrar la mejor.

## Límites
3 semillas (IC de ±0.03–0.05), un dataset, un espacio, una receta, un λ; 4 generaciones en `p3`/`rs`. Evaluaciones casi iguales (420 y 390) pero entrenamientos locales distintos. Los conjuntos finales no tienen el mismo tamaño
(el hall of fame de `base` es su frente no dominado, 4–7 modelos), lo que complica comparar hipervolúmenes. La mejora por generación se mide sobre la accuracy *estimada* (pocos nodos, ruidosa), no sobre test. La penalización común a todos los clusters elimina el frente accuracy-tamaño.
