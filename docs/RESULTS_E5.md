# Resultados E5 — NSGA-III con un objetivo por cluster

Diseño preregistrado: `docs/E5_DESIGN.md`. 6 brazos × 5 semillas pareadas (0–4), Fashion-MNIST α=0.1, N=100, C=10,
20 rondas × 2 épocas, Adam 1e-2. 30/30 corridas terminadas sin errores. Tablas en `runs/e5/` (`per_run.csv`,
`summary_by_arm.csv`, `paired_contrasts.csv`; generadas por `analyze_e5.py`, no versionadas).

## Resumen por brazo (media de 5 semillas; test, reentrenamiento en los 100 nodos)

| | n3_cs | n3_rand | n3_cs_lownoise | rs_cs | n2_cs | n2_rand |
|---|---|---|---|---|---|---|
| acc. media del mejor modelo | 0.778 | 0.795 | 0.789 | 0.771 | 0.771 | 0.785 |
| acc. del peor cluster | 0.702 | 0.727 | 0.722 | 0.701 | 0.705 | 0.706 |
| peor decil de nodos | 0.557 | 0.597 | 0.572 | 0.546 | 0.550 | 0.571 |
| params normalizados del mejor | 0.283 | 0.167 | 0.238 | 0.168 | 0.122 | 0.279 |
| tamaño del archivo no dominado | 115 | 97 | 86 | 80 | 14 | 10 |
| nodos que mejoran (media del conjunto) | 0.788 | 0.820 | 0.920 | 0.900 | 0.898 | 0.984 |

## Contrastes pareados (diferencia media, IC bootstrap 5 semillas)

- `n3_cs − n3_rand`: acc. media −0.017 [−0.045, +0.008]; peor cluster −0.025 [−0.073, +0.018].
- `n3_cs − rs_cs`: acc. media +0.007 [−0.033, +0.052].
- `n3_cs − n2_cs`: acc. media +0.006 [−0.033, +0.046].
- `n3_cs − n2_rand`: acc. media −0.008 [−0.024, +0.009].
- `n3_cs_lownoise − n3_cs`: acc. media +0.011 [−0.012, +0.039].
- Todos los ICs de accuracy incluyen 0: no se declara ventaja (regla preregistrada).
- Deriva de tamaño: `n3_cs` elige modelos mayores que `n3_rand` (+0.116 [0.009, 0.230]) y que `n2_cs` (+0.161 [0.068, 0.246]),
  como se anticipó al no incluir el tamaño como objetivo.

## Cautelas

- "Nodos que mejoran" no es comparable entre familias: las poblaciones iniciales difieren (55 vs 10 individuos; nodos distintos).
  Los contrastes negativos de `n3_cs` en esa métrica no se interpretan. `n3_cs` tuvo una semilla con 39% (semilla 4).
- Con 5 semillas el mínimo p de Wilcoxon es 0.0625; los ICs son descriptivos.
- Varianza entre semillas alta (p. ej. `rs_cs` semilla 2: 0.668; `n3_cs` semilla 4: 0.743).

## Conclusión

El objetivo por cluster con NSGA-III no mejora sobre sus controles: la búsqueda aleatoria con el mismo presupuesto y
NSGA-III con nodos aleatorios rinden igual o mejor, y el objetivo de menor ruido tampoco demuestra mejora. Es coherente con
E3: el ruido del fitness domina y, con 10 objetivos, la selección por dominancia se aproxima a muestreo aleatorio.
