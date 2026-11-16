# Resultados MOEA: NSGA-III con un objetivo por cluster y el tamaño como objetivo propio, frente a búsqueda aleatoria (2026-11-15)

Diseños preregistrados: `docs/MOEA_DESIGN.md` (v1, variantes A y B, auditoría de ruido, v2 y réplica). Scripts: `run_moea.sh`, `analyze_moea.py`, `run_noise_audit.py`, `analyze_k_search.py`. FEMNIST (partición natural), espacio de arquitecturas v2,
k=2 clusters por silueta (3 objetivos: accuracy de cada cluster + tamaño), 20 rondas × 2 épocas, Adam lr=1e-3. Tablas en `runs/moea/`, `runs/moea_b/`, `runs/moea2/`, `runs/noise_audit/` (no versionadas, se regeneran con los scripts).

## Método
- **Evaluación de una arquitectura:** cada representante (uno por cluster) entrena el modelo global durante 20 rondas × 2 épocas; el servidor agrega con FedAvg ponderado por **el tamaño del cluster que representa cada nodo** (no por muestras). Con ese único modelo global se calcula el objetivo de cada cluster
  (v1 A: accuracy en otro miembro retenido; v1 B: en el propio representante; v2: media sobre todos los miembros que no entrenaron). El tamaño es el tercer objetivo (`params/param_ref`, `param_ref = 354 494`).
- **Métrica primaria:** hipervolumen, en el plano (accuracy en test, tamaño normalizado) con `HV_REF = (0, 1.05)`, de los 12 modelos del hall of fame reentrenados con los 100 nodos (pesos uniformes) y medidos en test. Convergencia: hipervolumen del archivo acumulado con las accuracies *estimadas*. Población: hipervolumen de los individuos nuevos por generación.
- **Contrastes:** pareados por semilla `nsga3 − rs`, Wilcoxon unilateral e IC95 bootstrap (descriptivo). "El MOEA optimiza mejor" solo si la primaria **y** la convergencia tienen p < 0.05. Con 5 semillas el p mínimo es 0.031 (5/5) y con una en contra 0.0625.

## 1. v1: NSGA-III no supera a la aleatoria (5 semillas, 1 988 evaluaciones)
| | A: nodo retenido | B: en los nodos que entrenan |
|---|---|---|
| hipervolumen reentrenado `nsga3` / `rs` | 0.539 / 0.558 | 0.554 / 0.573 |
| `nsga3 − rs` (IC95), semillas a favor | −0.020 [−0.053, +0.024], 1/5 | −0.019 [−0.032, −0.007], 1/5 |
| convergencia del archivo | +0.027 (5/5, p 0.031) | +0.035 (5/5, p 0.031) |
| mejor accuracy en test `nsga3` / `rs` | 0.529 / 0.562 | 0.545 / 0.584 |
| rango de tamaños del frente reentrenado | 0.12 / 0.24 | 0.14 / 0.36 |
Decisión preregistrada en ambas: no se detecta ventaja. La convergencia no es fiable: el archivo de `nsga3` incluye las reevaluaciones de los padres (maldición del ganador). Evaluar en los nodos que entrenan (B) infla las accuracies estimadas (hipervolumen del archivo 0.73 frente a 0.57), pero el hipervolumen reentrenado de ambos brazos cambió igual (+0.015), sin efecto diferencial.

**Causas medidas.** (a) Con el mismo número de evaluaciones, `nsga3` evaluó ≈690 arquitecturas distintas frente a 1 988 de la aleatoria: 65 % de repeticiones (≈49 puntos son reevaluaciones de padres y ≈16 hijos que decodifican a una arquitectura ya vista), y la selección solo usaba la última medición.
(b) Ruido: la desviación de `acc_rep` al reevaluar la misma arquitectura es 0.061, frente a 0.087 de dispersión entre individuos de una generación.

## 2. Auditoría de ruido del objetivo por cluster (24 arquitecturas × 10 sorteos de representantes × 3 federaciones)
| esquema de evaluación | ruido (SD entre sorteos) | señal (SD entre arquitecturas) | fiabilidad de un sorteo | Kendall τ con la accuracy real (100 nodos) |
|---|---|---|---|---|
| E1: un nodo retenido (v1 A) | 0.068 | 0.115 | 0.74 | 0.71 |
| E2: el propio representante (v1 B) | 0.071 | 0.201 | 0.89 | 0.79 |
| **E3: todos los miembros que no entrenaron (v2)** | **0.036** | 0.110 | **0.90** | 0.78 |
| E4: 5 miembros al azar | 0.046 | 0.111 | 0.85 | 0.76 |
El ruido por elegir el nodo de evaluación es ≈0.057 y el de cambiar los representantes que entrenan ≈0.036 (≈74 % de la varianza de E1 era de evaluación). E3 cumple el umbral preregistrado de fiabilidad (0.902 ≥ 0.90), por poco. E2 amplía la señal entre arquitecturas (sesgo dependiente de la capacidad) aunque su ranking sigue alineado.
Promediar sorteos apenas mejora el ranking (τ 0.78 → 0.82): el límite no es solo ruido, entrenar con 2 representantes ordena distinto que con 100 nodos.

## 3. v2: score en todos los miembros, sin reevaluar y sin duplicados
Objetivos `[−m_1, −m_2, tamaño]` con `m_c` = media del cluster; solo se evalúan los hijos, caché por arquitectura decodificada, duplicados eliminados a nivel de arquitectura, parada a 1 988 evaluaciones reales (≈70 generaciones; caché: ≈160 entrenamientos ahorrados por corrida). `nsga3` evalúa ahora ≈1 998 arquitecturas distintas, como la aleatoria.

`nsga3 − rs` (diferencia media; IC95 bootstrap; semillas a favor):
| métrica | semillas 0–4 | réplica 5–9 | 10 semillas (0–9) |
|---|---|---|---|
| **hipervolumen reentrenado (primaria)** | +0.038 [−0.003, +0.065], 4/5, p 0.0625 | −0.004 [−0.031, +0.019], 3/5, p 0.59 | **+0.017 [−0.009, +0.042], 7/10, p 0.12** |
| convergencia del archivo | +0.023 [+0.016, +0.030], 5/5, p 0.031 | +0.032 [+0.011, +0.050], 4/5, p 0.0625 | +0.027 [+0.016, +0.038], 9/10, p 0.002 |
| pendiente de la población | +0.0003, 4/5, p 0.0625 | +0.0004, 5/5, p 0.031 | +0.0003 [+0.0002, +0.0005], 9/10, p 0.002 |
| mejora de la población (últimas 5 gen. − g0) | +0.039, 4/5, p 0.0625 | +0.063 [+0.039, +0.081], 5/5, p 0.031 | +0.051 [+0.029, +0.072], 9/10, p 0.002 |
Por semilla (primaria): 0–4: +0.068, +0.041, +0.068, +0.053, −0.041; 5–9: +0.010, +0.013, +0.029, −0.022, −0.049. Medias por brazo (10 semillas): hipervolumen reentrenado 0.574 (`nsga3`) y 0.557 (`rs`); mejor accuracy en test 0.570 y 0.558; tamaño del frente 10.2 y 9.0; tiempo 2.4 h por corrida.
Frente al v1 A: `nsga3` +0.040 en las semillas 0–4 (0.539 → 0.579) y `rs` −0.017 (dentro del ruido entre semillas).

**Decisión preregistrada (10 semillas): mejora la población, sin ventaja demostrada en el frente final.** La réplica por separado no cumple su criterio (≥ 4 de 5 semillas a favor con media > 0 en la primaria: 3/5 y −0.004).

## 4. Diagnóstico del cuello de botella (pasos A, B y C)
El v2 optimiza el proxy pero el hipervolumen reentrenado no se distingue de la aleatoria. Preregistrado en `docs/MOEA_DESIGN.md`: A y B reusan datos (`analyze_proxy_gap.py`, `analyze_fidelity.py`, `run_noise_audit.py --reps_per_cluster`); C no hace nueva búsqueda (`run_rerank.py`).

**A: calidad de selección del proxy de 1 representante** (24 arquitecturas, 3 federaciones; accuracy real: mejor 0.56–0.59, mediana 0.24–0.26, peor ≈0.05; el azar daría recall@k ≈ k/24):
| | E1: un nodo retenido (v1) | E3: todos los miembros (v2) |
|---|---|---|
| τ | 0.734 | 0.788 |
| **τ en la mitad superior real** | 0.361 | **0.496** |
| recall@3 / @6 / @8 | 0.42 / 0.66 / 0.80 | 0.53 / 0.71 / 0.83 |
| arrepentimiento si eliges la mejor por proxy | 6.9 pp | 3.4 pp |
| arrepentimiento con las 3 / 6 mejores por proxy reentrenadas | 2.4 / 0.5 pp | 0.14 / 0.0 pp |
El proxy ordena bien en conjunto pero mal entre las buenas. Sesgo (proxy − real): −8.2 pp de media (E3), más negativo con BatchNorm (Spearman −0.82), más bloques (−0.67), ELU (−0.52) y más parámetros (−0.56); dropout +0.47. Es decir, penaliza los rasgos de las arquitecturas que ganan de verdad (la mejor, la 18: BatchNorm, ELU, 3 bloques, 69 mil parámetros). Solo 24 arquitecturas con rasgos correlacionados: no se separa cuál causa el sesgo.

**B: fidelidad frente a representantes por cluster** (r representantes al azar por cluster, peso `|G_c|/(r·N)` cada uno; 6 sorteos por (federación, r), 10 para r=1):
| r | coste | ruido | τ | τ mitad superior | recall@3 | arrepentimiento @1 | sesgo medio | ρ(sesgo, BatchNorm) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1× | 0.036 | 0.788 | 0.50 | 0.53 | 3.4 pp | −8.2 pp | −0.82 |
| 2 | 2× | 0.031 | 0.853 | 0.75 | 0.76 | 0.4 pp | −4.4 pp | −0.69 |
| 4 | 4× | 0.023 | 0.892 | 0.87 | 0.85 | 0.9 pp | −2.0 pp | −0.66 |
| 8 | 8× | 0.016 | 0.913 | 0.91 | 0.96 | 0.2 pp | −0.9 pp | −0.46 |
τ de un sorteo frente a r=1 (IC95 bootstrap pareado): r=2 **+0.064 [+0.009, +0.137]**, r=4 **+0.104 [+0.032, +0.210]**, r=8 +0.124 [+0.048, +0.238]: la regla preregistrada (≥ +0.05 con IC > 0 para r ≤ 4) se cumple. El nivel del sesgo baja con r, pero su relación con BatchNorm y profundidad solo se debilita de verdad con r=4–8.

**C: ¿el archivo del MOEA contiene mejores candidatos?** De cada uno de los 20 archivos del v2 se reentrenan 24 candidatos (`select_hof`, K=24; los 12 ya reentrenados se reutilizan) con los 100 nodos:
| | `nsga3` | `rs` |
|---|---|---|
| hipervolumen con los 12 originales | 0.574 | 0.557 |
| hipervolumen con los 24 reentrenados | 0.577 | 0.568 |
| puntos del frente de la unión de ambos brazos (media; por semilla de 7 a 15 frente a 1 a 9) | 11.7 | 4.5 |
`nsga3 − rs` con 24: +0.009 [−0.013, +0.031], p unilateral 0.25, 7/10 semillas (+0.068, +0.042, +0.020, +0.039, −0.041, +0.002, +0.010, +0.027, −0.023, −0.050). Reentrenar 24 en lugar de 12 ayuda más a `rs` (+0.011) que a `nsga3` (+0.003). El hipervolumen de la unión (media 0.591) supera a `nsga3` en ≈0.014 y a `rs` en ≈0.023: el margen disponible es pequeño. El porcentaje del frente de la unión es descriptivo, no preregistrado.

**Lectura preregistrada:** C no distingue los brazos y A/B muestran un proxy sesgado y poco fiable entre las buenas que mejora con más representantes → el límite es la fidelidad del proxy; se prueba el MOEA con r > 1 a igual coste (v3, `docs/MOEA_DESIGN.md`, `configs/cs_moea3.yaml`).

## 5. v3: 2 representantes por cluster a igual coste de entrenamiento
Diseño en `docs/MOEA_DESIGN.md` (sección "MOEA v3") y `configs/cs_moea3.yaml`: como el v2 pero con `reps_per_cluster=2` (4 nodos entrenan por evaluación, peso `|G_c|/(r·N)` cada uno), **994 evaluaciones** (los mismos ≈79 500 entrenamientos locales que el v2) y población 15 (≈66 generaciones); `nsga3` y `rs`, semillas 0–9, 20 corridas.

`nsga3 − rs` (10 semillas; diferencia media, IC95 bootstrap, semillas a favor, p unilateral de Wilcoxon):
| métrica | v3 (r=2, ≈1 000 evaluaciones) |
|---|---|
| **hipervolumen reentrenado (primaria)** | +0.005 [−0.010, +0.020], 5/10, p 0.28 |
| convergencia del archivo | +0.022 [+0.010, +0.034], 9/10, p 0.005 |
| pendiente de la población | +0.0004 [+0.0002, +0.0006], 9/10, p 0.003 |
| mejora de la población (últimas 5 gen. − g0) | +0.083 [+0.053, +0.117], **10/10**, p 0.001 |
Hipervolumen reentrenado medio: 0.582 (`nsga3`) y 0.577 (`rs`); mejor accuracy en test 0.582 y 0.585; tamaño del frente 8.9 en ambos; ≈70 aciertos de caché por corrida de `nsga3`. Decisión preregistrada: **mejora la población, sin ventaja demostrada en el frente final.**

**Fidelidad frente a cantidad (v3 − v2 a igual coste de entrenamiento; descriptivo, Wilcoxon bilateral):**
| | v2 (r=1, 1 988 evaluaciones) | v3 (r=2, ≈1 000) | cambio (IC95), semillas a favor |
|---|---|---|---|
| `nsga3`, hipervolumen reentrenado | 0.574 | 0.582 | +0.009 [−0.006, +0.021], p 0.16, 8/10 |
| `rs`, hipervolumen reentrenado | 0.557 | 0.577 | +0.021 [+0.002, +0.040], p 0.13, 7/10 |
| `nsga3`, Kendall τ estimado-test en el hall of fame | 0.794 | 0.788 | −0.006 [−0.070, +0.064], p 0.82 |
| `rs`, Kendall τ estimado-test en el hall of fame | 0.730 | 0.794 | +0.064 [+0.006, +0.140], p 0.08 |
Mejor evaluar bien que evaluar mucho, pero poco: ambos brazos suben, más la aleatoria, y ningún contraste bilateral llega a 0.05. La ventaja relativa del MOEA en el frente final no aumenta con el proxy más fiable (+0.005 frente a +0.017 del v2).

## 6. Paso D: curva de eficiencia con los archivos del v2
Preregistrado en `docs/MOEA_DESIGN.md` ("Paso D"); `python run_rerank.py --budgets 250 500 1000` → `runs/rerank_prefix/` (410 reentrenamientos nuevos con los 100 nodos). Para cada corrida del v2 (semillas 0–9, ambos brazos) se eligen 12 candidatos (`select_hof`) con solo las primeras B evaluaciones reales del archivo y se mide su hipervolumen reentrenado.
Hipótesis: si el espacio está saturado, la ventaja de un MOEA debería verse con pocas evaluaciones.

| B (evaluaciones) | `nsga3` | `rs` | `nsga3 − rs` (IC95) | semillas a favor | p unilateral (p con Holm) |
|---|---|---|---|---|---|
| 250 | 0.550 | 0.546 | +0.005 [−0.015, +0.023] | 6/10 | 0.28 (0.35) |
| 500 | 0.560 | 0.549 | +0.011 [−0.011, +0.030] | 6/10 | 0.12 (0.35) |
| 1 000 | 0.572 | 0.553 | +0.020 [−0.001, +0.041] | 6/10 | 0.12 (0.35) |
| 1 988 (completo) | 0.574 | 0.557 | +0.017 [−0.009, +0.041] | 7/10 | 0.12 |
**Lectura preregistrada: ningún presupuesto parcial cumple la regla** (p con Holm < 0.05, ≥ 7/10 semillas): no se demuestra eficiencia de muestras. La diferencia `nsga3 − rs` **crece con el presupuesto** (+0.005, +0.011, +0.020), lo contrario de lo esperado si el MOEA ganara sobre todo con pocas evaluaciones: necesita generaciones para acumular ventaja, y esa ventaja sigue por debajo de la dispersión entre semillas (SD ≈ 0.03–0.04 con 10 semillas).
El hipervolumen reentrenado de `nsga3` se aplana hacia 1 000 evaluaciones (0.572 → 0.574 con el presupuesto completo), mientras que el de `rs` sigue subiendo despacio (0.546 → 0.557).

**Post hoc (no preregistrado; bilateral):** `nsga3` con 500 evaluaciones iguala a `rs` con 1 988 (+0.003 [−0.025, +0.029], 6/10, p 1.0), es decir, una calidad parecida con ≈4 veces menos evaluaciones, pero el IC (±0.027) no distingue esto de una ventaja nula ni de una mayor. `nsga3` con 1 000 evaluaciones frente a `rs` con 1 988: +0.016 [−0.011, +0.040], 7/10, p 0.32.

## 7. v4: espacio de arquitecturas más difícil
Diseño y piloto en `docs/MOEA_DESIGN.md` ("MOEA v4") y `configs/cs_moea4.yaml`. Espacio: 5 bloques, canales hasta 128, kernels {3, 5, 7}, activaciones {relu, elu, gelu, tanh}, hasta 3 capas FC de hasta 512; 12 genes (≈3·10⁷ arquitecturas discretas, ×1 600). Piloto (60 arquitecturas aleatorias): tiempo por evaluación mediana 10.5 s, el 78 % aprende, accuracy del proxy con SD 0.150 (0.087 en el espacio anterior) y solo el 3–4 % a ≤ 0.05 de la mejor (10 % antes).
Diseño de la búsqueda como el v3 (k=2, 3 objetivos, 2 representantes por cluster, sin reevaluar, caché, duplicados por arquitectura): `eval_budget` 504 (510 evaluaciones reales) y población 15; `nsga3` y `rs`, semillas 0–9, 20 corridas (2.8 h y 3.4 h por corrida).

`nsga3 − rs` (10 semillas; diferencia media, IC95 bootstrap, p unilateral, semillas a favor):
| métrica | v4 |
|---|---|
| **hipervolumen reentrenado (primaria)** | +0.012 [−0.017, +0.039], p 0.25, 7/10 (+0.040, −0.044, +0.053, +0.073, +0.002, −0.051, +0.036, +0.018, +0.052, −0.055) |
| convergencia del archivo | +0.024 [+0.008, +0.040], p 0.014, 9/10 |
| pendiente de la población | +0.0022 [+0.0015, +0.0028], p 0.002, 9/10 |
| mejora de la población (últimas 5 gen. − g0) | +0.103 [+0.063, +0.140], p 0.002, 9/10 |
Hipervolumen reentrenado medio 0.696 (`nsga3`) y 0.684 (`rs`) (no comparable con v2/v3: otro espacio y otra normalización del tamaño); mejor accuracy en test 0.669 y 0.659 (≈0.58 en el espacio anterior: el espacio mayor contiene modelos mejores); tamaño del frente 8.1 y 7.9; Kendall τ estimado-test en el hall of fame 0.69 y 0.71 (0.79 en el v2). Decisión preregistrada: **mejora la población, sin ventaja demostrada en el frente final.**

**Diagnóstico de saturación.** Ganancia tardía del archivo de `rs` (hipervolumen estimado al 100 % − al 50 % de las evaluaciones): +0.0134 en v4 y +0.0134 en v2, es decir, **igual**: por esta medida el espacio mayor no es menos saturado para la aleatoria. En el mismo rango absoluto de evaluaciones (252 → 504) la ganancia de `rs` es +0.0134 (v4) frente a +0.0157 (v2), y la de `nsga3` +0.0224 frente a +0.0191. (El hipervolumen del archivo usa accuracies estimadas y otra normalización del tamaño.)

**Los tres diseños juntos.** Primaria `nsga3 − rs`: v2 +0.017, v3 +0.005, v4 +0.012 (7, 5 y 7 de 10 semillas a favor). Promediada por federación (las tres versiones comparten las mismas 10 federaciones; descriptivo y no independiente): +0.0115 [−0.0076, +0.0296], 6/10 federaciones a favor, Wilcoxon unilateral p = 0.14; la media por federación va de −0.046 a +0.048.

## Figuras
`python make_report_figures.py` regenera desde `runs/` las figuras del informe LaTeX en `report/figures/` (PDF y PNG; ignoradas por git): frentes de Pareto reentrenados (curva de alcance mediana entre semillas) de v2, v3 y v4, diferencias de hipervolumen por diseño (forest plot), convergencia del archivo y de la población, curva de eficiencia del Paso D y, en el anexo del informe, los frentes por semilla de v2 y v4.

## 8. Coste de la búsqueda y tiempo
**Nodos.** Cada evaluación entrena con 2 nodos (uno por cluster; v1 y v2) o 4 (dos por cluster; v3 y v4), 20 rondas × 2 épocas. v1: se mide en 2 nodos retenidos; desde v2, en todos los miembros del cluster que no entrenaron (≈96–98 nodos), solo inferencia. Los representantes se re-sortean por generación: llegan a entrenar 50–93 de los 100 nodos. El coste de entrenamiento de la búsqueda es igual en ambos brazos por construcción y muy inferior a entrenar cada candidato con los 100 nodos (2 000 entrenamientos locales): ≈79 600 frente a 3.98 M en v2 (50×) y ≈40 300 frente a 1.0 M en v4 (25×). Costes fijos iguales: sonda (100 entrenamientos de 1 época) y etapa final (12 × 2 000 = 24 000).

| diseño | brazo | entrenan / eval. | evalúan / eval. | entren. locales | generaciones | nodos distintos | tamaño medio evaluado | búsqueda (h) | final (h) | total (h) |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 A | `nsga3` | 2 | 2 | 79 520 | 36.0 | 50 | 0.043 | 1.36 | 0.37 | 1.73 |
| v1 A | `rs` | 2 | 2 | 79 520 | 36.0 | 50 | 0.130 | 1.77 | 0.41 | 2.18 |
| v2 | `nsga3` | 2 | ≈98 | 79 936 | 77.1 | 78 | 0.079 | 1.97 | 0.39 | 2.36 |
| v2 | `rs` | 2 | ≈98 | 79 520 | 71.0 | 76 | 0.131 | 2.03 | 0.43 | 2.46 |
| v3 | `nsga3` | 4 | ≈96 | 80 048 | 71.4 | 93 | 0.110 | 2.42 | 0.48 | 2.90 |
| v3 | `rs` | 4 | ≈96 | 80 400 | 67.0 | 92 | 0.130 | 2.16 | 0.56 | 2.72 |
| v4 | `nsga3` | 4 | ≈96 | 40 800 | 34.8 | 75 | 0.033 | 2.01 | 0.80 | 2.81 |
| v4 | `rs` | 4 | ≈96 | 40 800 | 34.0 | 74 | 0.062 | 2.51 | 0.86 | 3.37 |
(`búsqueda` = tiempo de entrenamiento y evaluación del checkpoint de la búsqueda; `final` = total − búsqueda; horas con 20 procesos compartiendo la máquina, ruido ≈10 %.)

**Caché y generaciones.** Los hijos de `nsga3` que repiten una arquitectura se sirven de la caché y no cuentan como evaluación real (≈160 por corrida en v2, ≈70 en v3, 12 en v4): `nsga3` hace ≈6 generaciones más que `rs` en v2 (77.1 frente a 71.0), 4 en v3 y <1 en v4 para el mismo número de evaluaciones reales. La caché no reduce el coste: el presupuesto compra arquitecturas distintas (≈1 998 en v2, como la aleatoria).
**Robustez frente al criterio de parada.** Se paró al llegar al mismo número de evaluaciones reales (igual cómputo), lo que da a `nsga3` ≈8 % más generaciones que a `rs`; con igual número de generaciones tendría ≈7 % menos entrenamiento (1 850 frente a 1 988 evaluaciones en v2). Se cortó el archivo de `nsga3` en el mismo número de generaciones que `rs` y se recalculó el hipervolumen del archivo (accuracies estimadas, 10 semillas):
| diseño | `nsga3`: evaluaciones (todas / cortado) | HV del archivo (todas / cortado) | `rs` | ventaja (todas / cortado) | aporte de las extra |
|---|---|---|---|---|---|
| v2 | 1 998 / 1 850 | 0.4745 / 0.4713 | 0.4439 | +0.0306 / +0.0274 | +0.0031 |
| v3 | 1 001 / 941 | 0.5281 / 0.5276 | 0.5085 | +0.0195 / +0.0191 | +0.0004 |
En v4 la diferencia es de menos de una generación y no se recalculó. La ventaja se mantiene a igual número de generaciones; las extra aportan ≈10 % (v2) y ≈2 % (v3), porque llegan al final de la búsqueda. Estimación indirecta sobre el frente reentrenado (Paso D): `nsga3` pasa de 0.5724 a 0.5737 entre 1 000 y 1 988 evaluaciones, así que ≈160 evaluaciones valen ≈ +0.0002. La elección del criterio no altera las conclusiones.
**Tiempo.** `nsga3` es más rápido que `rs` en v1 (−21 %) y v4 (−17 %), casi igual en v2 (−4 %) y más lento en v3 (+7 %); la etapa final es algo más rápida en los cuatro. `nsga3` evalúa arquitecturas más pequeñas (tamaño 0.043 frente a 0.130 en v1, 0.079 frente a 0.131 en v2, 0.110 frente a 0.130 en v3, 0.033 frente a 0.062 en v4), lo que explica en parte v1 y v4, pero el tiempo no sigue al tamaño (v2: −40 % de tamaño, −3 % de tiempo de búsqueda; v3: −15 % de tamaño, +12 %): los parámetros son un mal indicador del coste de cómputo. Pendiente: comparar a igual tiempo de cómputo (tiempo por evaluación en una máquina sin carga o FLOPs).

## Conclusiones
1. El MOEA v2 **optimiza lo que se le pide**: el archivo y la población mejoran frente a la aleatoria en 9 de 10 semillas (p = 0.002), y se replica en ambas mitades.
2. **No se demuestra una ventaja en el frente final reentrenado**: +0.038 en las 5 primeras semillas se redujo a −0.004 en la réplica (regresión a la media); sobre las 10 semillas es +0.017 con IC que incluye el 0.
3. Las correcciones del v2 (no reevaluar, sin duplicados, score en todos los miembros) cambiaron el resultado del v1 de "peor que la aleatoria" a "no distinguible o ligeramente mejor", y explicaron por qué el v1 perdía (exploración y ruido); no se separa cuál aportó cuánto.
4. v3 (r=2 a igual coste): mejorar el proxy sube el hipervolumen reentrenado de ambos brazos (+0.009 `nsga3`, +0.021 `rs`) pero no abre una ventaja del MOEA (+0.005 frente a la aleatoria; 5/10 semillas), mientras la población sigue mejorando (10/10, p = 0.001).
5. Hipótesis (no probada) principal: el espacio de 9 genes está **saturado** con ~1 000–2 000 evaluaciones: ambos brazos llegan a ≈0.58 de hipervolumen y a ≈0.58 de accuracy máxima, y la unión de los archivos de ambos solo supera a cada brazo en ≈0.014–0.023 (Paso C).
6. Paso D (curva de eficiencia): no se demuestra ventaja a presupuestos bajos (Holm p = 0.35 en B = 250, 500 y 1 000); la diferencia crece con el presupuesto hasta ≈ +0.02, sin significancia con 10 semillas. Con la dispersión observada (SD de la diferencia ≈ 0.03–0.04 por semilla), confirmar un efecto de ≈ +0.02 exigiría ≈ 20–35 semillas.
7. v4 (espacio ×1 600 mayor): mismo patrón. El MOEA mejora la población y el archivo (9/10) y la primaria es +0.012 (IC95 [−0.017, +0.039], 7/10); el diagnóstico de saturación no muestra que el espacio mayor sea menos saturado para la aleatoria (ganancia tardía igual a la del espacio anterior). En los tres diseños el efecto en el frente final es de ≈ +0.01 (+0.0115 por federación, p = 0.14), por debajo de la dispersión entre semillas.
8. Hipótesis (no probada) de por qué se diluye: el MOEA optimiza un **proxy** (2 representantes por cluster) que ordena las arquitecturas con Kendall ≈ 0.78 respecto del entrenamiento con los 100 nodos; la aleatoria también elige su hall of fame con ese proxy, y el hipervolumen reentrenado se mide con 12 modelos y una inicialización (SD entre semillas ≈ 0.04).

## Límites
Un dataset (FEMNIST), un espacio, una receta y un solo MOEA (NSGA-III); 10 semillas en total; la réplica 5–9 es una ampliación posterior al primer análisis (el p combinado es algo optimista). Las tres correcciones del v2 se aplicaron juntas, sin ablaciones. El hipervolumen reentrenado usa una sola inicialización por modelo.
Evaluar en todos los miembros relaja la premisa de no tocar a todos los nodos (solo inferencia, ≈ +50 % de tiempo por evaluación). Los umbrales de la auditoría de ruido (fiabilidad ≥ 0.90) los fijé yo.

## Siguientes pasos
Ni el Paso D ni el espacio más difícil (v4) abren una ventaja del MOEA sobre la aleatoria en el frente final reentrenado. Opciones: (a) cerrar con lo medido y actualizar el informe LaTeX; (b) más federaciones (≈ 40) para confirmar o descartar un efecto de ≈ +0.01; (c) atacar la brecha entre el proxy y el entrenamiento completo (en v4 el τ estimado-test baja a 0.69), por ejemplo reordenando con reentrenamiento los mejores candidatos del archivo antes de elegir el hall of fame (Paso C aplicado a los archivos del v4).
