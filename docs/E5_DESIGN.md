# E5 — NSGA-III con un objetivo por cluster (fijado antes de lanzar, 2026-10-22)

Cambio de algoritmo propuesto por el usuario: en lugar de optimizar (accuracy estimada media, nº de parámetros), cada **objetivo es la
accuracy del nodo evaluador de un cluster** (nº de objetivos = C), por lo que la búsqueda pasa a ser many-objective y se usa **NSGA-III**.
El tamaño de la red deja de ser objetivo y solo se reporta.

## Configuración común (`configs/cs_e5.yaml`, `run_e5.sh`)
Fashion-MNIST, α=0.1, N=100, C=10, evaluador retenido, Adam 1e-2, 20 rondas × 2 épocas, espacio CNN de E3, 5 semillas pareadas (0–4; misma
semilla = misma partición y sonda en los 6 brazos). Mismo coste por evaluación (10 nodos × 20 rondas = 200 entrenamientos locales).
Con 10 objetivos las direcciones de Das–Dennis con p=2 son 55 → población 55; con reevaluación de padres, 55 + 3 × 110 = **385
evaluaciones** (≈ 77 000 entrenamientos locales por corrida; E3 usaba 150 evaluaciones).

| Brazo | Muestreo de nodos | Objetivos | Búsqueda | Evaluaciones |
|---|---|---|---|---|
| `n3_cs` (propuesta) | un representante por cluster; otro miembro del cluster evalúa | 10 accuracies (una por cluster) | NSGA-III, pob. 55 | 385 |
| `n3_rand` | 10 nodos al azar entrenan; otros 10 al azar evalúan | 10 accuracies de esos nodos | NSGA-III, pob. 55 | 385 |
| `n3_cs_lownoise` (ablación) | igual que `n3_cs` | 10 accuracies = media de TODOS los miembros del cluster que no entrenaron (solo inferencia) | NSGA-III, pob. 55 | 385 |
| `rs_cs` | clusters | (registro de los 10 objetivos) | búsqueda aleatoria, rejilla 55/110/110/110 | 385 |
| `n2_cs` | clusters | (acc. media ponderada, parámetros): algoritmo de E3 | NSGA-II, pob. 10, 20 gen. | 390 |
| `n2_rand` | nodos aleatorios | ídem | NSGA-II, pob. 10, 20 gen. | 390 |
`n3_cs_lownoise` relaja la premisa "no hace falta evaluar a todos los nodos" (evalúa a todos los miembros no entrenados de cada cluster, solo
inferencia): es una ablación/cota, no el algoritmo propuesto. En todos los brazos se registra `obj_acc` (accuracy por objetivo), `acc_rep` (media ponderada),
parámetros y las métricas de monitoreo en todos los nodos (que no influyen en la búsqueda).

## Hall of fame con C objetivos (decisión semántica, misma regla en los 6 brazos)
Se agrega por genoma (media de sus `obj_acc` y de `acc_rep`); candidatos = capas no dominadas 1, 2, … en los C objetivos hasta tener al menos
K = 12; se incluyen (a) el mejor por media ponderada, (b) el mejor por su peor objetivo (max-min) y se completa hasta K por *farthest-first* en el
espacio de objetivos normalizado (empates: menor índice). Implementación y pruebas: `evolution/refine.py::select_hof`. No se aplica el refinamiento de E4.
Población inicial de referencia: subconjunto aleatorio fijo (semilla de la corrida) de 12 arquitecturas de la generación 0 (10 en los brazos NSGA-II).
Todos se reentrenan con los 100 nodos (20 rondas × 2 épocas, inicialización 1000) y se guarda la accuracy en test de cada nodo (`final_nodes.csv`).

## Métricas (preregistradas)
1. **Primaria:** proporción de nodos que mejoran / quedan igual / empeoran, población inicial → hall of fame (media y mejor del conjunto), en test.
2. Accuracy media del mejor modelo del hall of fame (por media ponderada), accuracy del **peor cluster** (clusters de Δ de la semilla, comunes a todos los
   brazos) y accuracy del **peor decil de nodos**.
3. Reporte: nº de parámetros de los seleccionados (deriva de tamaño al quitar el objetivo de tamaño), tamaño del archivo no dominado, error del estimador,
   diversidad por generación.
Contrastes pareados por semilla (IC bootstrap sobre 5 semillas, descriptivo): `n3_cs − n3_rand`, `n3_cs − rs_cs`, `n3_cs − n2_cs`, `n3_cs − n2_rand`,
`n3_cs_lownoise − n3_cs`. No se declara ventaja si el IC incluye 0; se informa la diferencia mínima detectable (MDE).

## Riesgos asumidos (declarados antes de ver resultados)
1. Con 10 objetivos casi todo individuo es no dominado (resistencia a la dominancia) y cada objetivo es la accuracy de un solo nodo (≈ ±0.04 de error
   estándar, más el ruido de entrenamiento ±0.06–0.10): la selección puede ser casi aleatoria; los controles (`rs_cs`, `n3_rand`, `n3_cs_lownoise`) lo detectarían.
2. Solo ≈ 4 generaciones de NSGA-III (E3 tenía 8).
3. Sin objetivo de tamaño, la búsqueda puede derivar hacia redes grandes.
4. Coste ≈ 3.9 × el de E3 completo (6 brazos × 5 semillas × ≈ 77 000 entrenamientos locales).
5. Limitaciones heredadas: un dataset, un α, C=10, 5 semillas, una inicialización por modelo en el test final.
