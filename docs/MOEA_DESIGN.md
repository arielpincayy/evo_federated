# MOEA con un objetivo por cluster y el tamaño como objetivo propio (NSGA-III) frente a búsqueda aleatoria (preregistro, escrito antes de correr)

El producto es un **frente** de arquitecturas y el algoritmo es un **MOEA**. P3 (`docs/P3_DESIGN.md`) restó el tamaño dentro de cada objetivo de cluster, lo que ocultaba el frente accuracy-tamaño (lo fijaba λ),
y con 4 generaciones la evolución apenas actuó (mejoró la población, sin ventaja en el frente final frente al azar: `docs/RESULTS_P3.md`). Aquí el tamaño vuelve a ser un objetivo y el presupuesto sube, porque con k=2 cada
evaluación cuesta ≈5× menos entrenamiento que con 10 nodos. Solo NSGA-III, contra la búsqueda aleatoria al mismo presupuesto (decisión del usuario).

## Preguntas
1. ¿NSGA-III con este diseño optimiza mejor que la búsqueda aleatoria con el mismo presupuesto (calidad del frente reentrenado y velocidad de convergencia)?
2. ¿Qué frente accuracy-tamaño entrega y cuánto difieren los clusters entre sí?

## Diseño
- **Objetivos (minimización):** `[−acc_1, …, −acc_k, params/param_ref]`, con `acc_c` la accuracy del nodo evaluador del cluster c (evaluador retenido; representante re-sorteado en cada generación, `assignment(generation)`).
  `k` = k óptimo por silueta coseno (2..10), como en P3 (esperable k=2 → 3 objetivos). Modo `objectives: per_cluster_size`: `obj_acc` = accuracies por cluster + `−tamaño`, `obj_raw` = accuracies sin tamaño.
- **Direcciones de referencia y presupuesto:** `ref_partitions: 0` con `pop_size: 30` → para 3 objetivos p=6, 28 direcciones (población 28); `eval_budget: 2000` → 36 generaciones (1988 evaluaciones).
  La búsqueda aleatoria usa el mismo nº de individuos por generación y **los mismos representantes en cada generación** (misma semilla): comparación pareada.
- **Brazos:** `nsga3` y `rs`; 5 semillas de datos (0–4). Reentrenamiento final de K=12 modelos del hall of fame (`select_hof`; la regla max-min solo mira las accuracies) con los 100 nodos, medido en test.
- **Receta:** FEMNIST (`configs/cs_moea.yaml`), espacio v2, 20 rondas × 2 épocas, Adam lr=1e-3, como P3.

## Métricas (todas sobre el plano (accuracy, tamaño normalizado), `HV_REF = (0, 1.05)`)
1. **Primaria:** hipervolumen de los modelos reentrenados del hof (accuracy en test media de los 100 nodos).
2. **Convergencia (barata, sobre estimaciones):** hipervolumen del archivo acumulado (accuracy estimada, tamaño) frente al nº de evaluaciones; se resume con el área bajo la curva.
3. **Población:** hipervolumen de los individuos nuevos de cada generación (g0 completa; después solo hijos) y su pendiente por generación.
4. **Descriptivas:** tamaño del frente reentrenado, rango de tamaños que cubre, diferencia entre la mejor y la peor accuracy por cluster (estimada) en el hof.
- **Decisión:** contraste pareado por semilla `nsga3 − rs`, Wilcoxon **unilateral** (con 5 semillas el p mínimo es 0.031) e IC bootstrap (descriptivo). "El MOEA optimiza mejor" solo si la métrica primaria **y** la de convergencia van a favor con p < 0.05;
  si solo mejora la población (métrica 3), se informa como mejora de la población sin ventaja demostrada en el frente final.

## Riesgos asumidos
1. Las accuracies por cluster casi coinciden (Spearman 0.92 entre rankings de nodos): el tercer objetivo puede aportar poco y sí ruido.
2. La accuracy estimada del archivo es optimista (maldición del ganador) en ambos brazos; por eso la métrica primaria usa modelos reentrenados.
3. 5 semillas (intervalos anchos), un dataset, un espacio, una receta, un solo MOEA.

## Variante B (propuesta del usuario): evaluar el modelo ponderado en los propios nodos que entrenan (escrito antes de correr)
Cada objetivo de cluster pasa a ser la accuracy, en el split de **validación del representante** de ese cluster, del modelo global FedAvg ponderado entrenado con los representantes
(`sampling.eval_holdout=False`); en la variante A el evaluador era otro miembro del cluster que no entrenó. Más el tamaño como objetivo propio. Todo lo demás igual (mismas semillas 0–4, brazos, presupuesto, métricas y regla de decisión). Salidas en `runs/moea_b/`.
- **Por qué:** en E2 (`experiment/cluster-sampling-nas`, CIFAR-10, α=0.5, 10 representantes) evaluar en los nodos entrenados daba sesgo optimista (+0.056, RMSE 0.060) pero la componente de error no debida al sesgo era menor
  (≈ √(0.060² − 0.056²) ≈ 0.022 frente a 0.031 con evaluador retenido) y la correlación de rangos igual o mayor (Kendall τ 0.87 frente a 0.84). Una búsqueda necesita ordenar, no acertar el valor absoluto.
- **Riesgo:** el sesgo podría correlacionar con la capacidad del modelo (los grandes ajustan mejor la distribución del nodo que entrenó) y empujar hacia tamaños mayores; el objetivo de tamaño lo contrarresta parcialmente. E2 es otro dataset y otro espacio.
- **Comparación secundaria (descriptiva):** además de `nsga3 − rs` dentro de la variante, se compara el hipervolumen reentrenado de `nsga3` entre variantes B y A.

## Auditoría de ruido del fitness por cluster (escrita antes de correr)
Pregunta (propuesta del usuario): si cada arquitectura se evalúa en n nodos de su cluster y se promedia, ¿el score de un cluster es estable entre generaciones aunque los representantes que entrenan cambien? Si lo es, no hace falta reevaluar a los padres.
Diseño (`run_noise_audit.py`): FEMNIST, espacio v2, receta de `cs_moea.yaml` (k por silueta), federaciones 0–2. Para cada una, 10 sorteos de representantes (el mismo mecanismo `assignment(draw)` de la búsqueda); en cada sorteo se entrenan
las 24 arquitecturas del estudio arquitectura × nodo con FedAvg ponderado y se guarda la accuracy en val de los 100 nodos. Esquemas de evaluación del objetivo de cada cluster (todos derivados de las mismas accuracies):
(E1) un nodo retenido al azar del cluster (variante A), (E2) el propio representante que entrenó (variante B), (E3) la media de todos los miembros del cluster que no entrenaron (n = todos), (E4) la media de 5 miembros al azar que no entrenaron.
- **Ruido:** SD entre sorteos de la misma arquitectura (media sobre arquitecturas, clusters y federaciones). **Señal:** SD entre arquitecturas de la media entre sorteos. **Fiabilidad de un sorteo:** `var_verdadera / (var_verdadera + var_ruido)`, con `var_verdadera = var_entre_arquitecturas − var_ruido/10`.
- **Concordancia:** Kendall τ entre el score de un sorteo y la accuracy real de la arquitectura entrenada con los 100 nodos (media de val y test, estudio arquitectura × nodo), por cluster.
- **Descomposición:** como E3 elimina casi todo el ruido de evaluación, su ruido ≈ ruido por cambiar los representantes que entrenan; el de evaluación ≈ √(ruido(E1)² − ruido(E3)²).
- **Decisión:** la propuesta (re-sortear representantes y no reevaluar) se sostiene si la fiabilidad de un sorteo con E3 es ≥ 0.90. Si es menor, el ruido por cambiar los representantes domina y habría que fijarlos (o promediar sorteos).

## MOEA v2: score por cluster en todos sus miembros, sin reevaluar y sin duplicados (escrito antes de correr)
Causas medidas del resultado del v1 (NSGA-III no superó a la aleatoria en el hipervolumen reentrenado; variantes A y B): (1) el 65 % de sus evaluaciones eran repetidas (≈690 arquitecturas distintas frente a 1 988 de la aleatoria; 49 puntos son reevaluaciones de padres y ≈16 hijos que decodifican a una
arquitectura ya vista) y la selección solo usaba la última medición; (2) el score por cluster era ruidoso: `run_noise_audit.py` dio SD 0.068 con un nodo retenido y 0.036 al evaluar en todos los miembros que no entrenaron (fiabilidad de un sorteo 0.902, justo sobre el umbral de 0.90).
- **Objetivos (minimización):** `[−m_1, …, −m_k, params/param_ref]`, con `m_c` = accuracy media en val de todos los miembros del cluster c que no entrenaron en esa evaluación; k por silueta (k=2 → 3 objetivos). Modo `objectives: per_cluster_mean_size`.
- **Sin reevaluar (`evolution.reeval_parents: False`):** cada generación se evalúan solo los hijos; los padres conservan el fitness con el que entraron. Caché por arquitectura decodificada (`json.dumps(genome.to_dict())`, incluye el dropout): una arquitectura ya evaluada reutiliza su fitness y no entrena.
  Los representantes se siguen re-sorteando en cada generación, de modo que el fitness de un padre viene de otra generación (ruido ≈0.036 frente a una dispersión entre arquitecturas de 0.110).
- **Duplicados:** eliminación a nivel de arquitectura decodificada frente a la población vigente (`genome_duplicates`, pymoo `ElementwiseDuplicateElimination`); la caché cubre los ya expulsados.
- **Presupuesto en evaluaciones reales:** `eval_budget: 1988` (las del v1); la búsqueda para al alcanzarlo (los aciertos de caché no cuentan), con `n_generations: 200` como tope. La aleatoria sortea `pop` individuos nuevos por generación, con los mismos representantes por generación que `nsga3`.
- **Brazos, semillas, receta, métricas y regla de decisión:** los del v1 (`nsga3` y `rs`, semillas 0–4, FEMNIST, espacio v2, 20 rondas × 2 épocas, lr 1e-3, K=12 reentrenados con los 100 nodos; primaria = hipervolumen reentrenado; "optimiza mejor" solo si primaria y convergencia p < 0.05 unilateral).
  Descriptivas nuevas: arquitecturas distintas evaluadas y aciertos de caché. Comparación secundaria (descriptiva): hipervolumen reentrenado v2 frente a v1 para cada brazo (`analyze_moea.py --compare runs/moea`). Salidas en `runs/moea2/`.
- **Riesgos:** (1) el ruido por cambiar los representantes sigue ahí; (2) los scores por cluster casi coinciden entre arquitecturas (Spearman 0.92): el tercer objetivo aporta poco; (3) evaluar en todos los miembros rompe la premisa "no hace falta tocar a todos los nodos" (solo inferencia, ≈ +50 % de tiempo por evaluación);
  (4) las tres correcciones (score en todos los miembros, sin reevaluar, sin duplicados) no se separan: si el v2 mejora no se sabrá cuál aportó qué; (5) 5 semillas, un dataset, un espacio, una receta.

## Réplica confirmatoria del v2: semillas 5–9 (escrita antes de correr)
Con las semillas 0–4, la primaria dio `nsga3 − rs = +0.038` (4/5 semillas a favor, IC95 [−0.003, +0.065], p unilateral 0.0625: con 5 semillas y una en contra ese es el mínimo alcanzable) y la convergencia p = 0.031 (5/5); la regla de decisión del v2 no se cumplió. Se repite el v2 sin cambios
(`CONFIG=configs/cs_moea2.yaml TAG=2 ./run_moea.sh 5 6 7 8 9`; mismas condiciones, misma regla) con federaciones nuevas.
- **Réplica (semillas 5–9) por separado:** mismos contrastes y mismos tests. Se considera que el resultado se replica si la primaria va a favor de `nsga3` en ≥ 4 de 5 semillas con media > 0.
- **Decisión sobre las 10 semillas (0–9):** la misma regla (Wilcoxon unilateral p < 0.05 en primaria **y** convergencia). Es una ampliación posterior al primer análisis (las 0–4 ya se habían visto), así que el p combinado es algo optimista; por eso se informa también la réplica por separado.
- Se reportan además las diferencias por semilla y el IC bootstrap. Análisis: `analyze_moea.py --root runs/moea2 --seeds 5 6 7 8 9 --suffix _rep` y `--seeds 0 1 2 3 4 5 6 7 8 9 --suffix _all`.
