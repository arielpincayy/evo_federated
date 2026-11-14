# Protocolo experimental: CS-NAS (muestreo estratificado por clusters de similitud)

Documento de decisiones que fijan la semántica del experimento. **Se fija antes de correr las campañas** (E2–E6);
cualquier cambio posterior se registra aquí con fecha y motivo.

## Algoritmo
1. **Probe (una vez):** todos los nodos entrenan el mismo modelo sonda desde la misma inicialización `k_epochs`
   épocas; solo se envía el delta Δ (nunca datos).
2. **Clusters:** k-means esférico sobre Δ normalizado (`sampling.method=kmeans`; alternativa `average`).
   C = `sampling.n_clusters` = nº de representantes por evaluación.
3. **Por generación:** se sortea 1 representante uniforme por cluster (RNG derivado de `seed` y la generación).
   Los mismos representantes evalúan a **todos** los individuos de esa generación (números aleatorios comunes).
4. **Fitness (minimización):** `(-acc_est, n_params / n_params_max)`, con
   `acc_est = Σ_c (|c|/N) · acc_val(eval_c)` (estimador estratificado de la accuracy media por nodo).
   Los representantes entrenan un único modelo FedAvg por arquitectura; la agregación pesa por tamaño de cluster
   (`weight_by_cluster`).
   **Nodo evaluador (`eval_holdout`, decisión 2026-10-18):** `eval_c` es *otro* miembro del mismo cluster que no
   entrenó (solo inferencia, sin coste de entrenamiento). Evaluar en el propio representante daba un sesgo optimista
   de +0.025 a +0.045 (piloto, N=100) porque el modelo acaba de entrenar con esos datos. Si el cluster tiene un solo
   nodo válido, el evaluador es el propio representante (se registra `n_eval_fallback`). Con `sampler=random` los
   evaluadores son C nodos distintos de los que entrenan; con `sampler=full` no hay nodos retenidos.
5. **NSGA-II con reevaluación:** padres e hijos se evalúan con los mismos representantes antes de la supervivencia.
6. **Monitoreo (solo reporte):** el mismo modelo se evalúa en `val` de **todos** los nodos (inferencia). Nunca entra al
   fitness ni a la selección (test unitario).

## Decisiones que cambian la semántica
- Los pesos del 2º objetivo se normalizan por los parámetros de la arquitectura más grande del espacio.
- HV con punto de referencia fijo `(-acc=0, params_norm=1.05)`.
- El frente "de salida" de un algoritmo en la generación t es el frente no dominado, por accuracy estimada, de **todas**
  sus evaluaciones hasta t (misma regla para todos los métodos). Su HV se reporta con la accuracy estimada
  (`hv_rep`) y con la accuracy de todos los nodos (`hv_monitor`).
- Random search recibe el mismo nº de evaluaciones por generación que NSGA-II (`pop` en la gen. 0, `2·pop` después)
  para igualar el coste en entrenamientos locales.
- CIFAR-10 se normaliza por canal (media/desv. estándar fijas); Fashion-MNIST solo se escala a [0,1].
- Test: partición Dirichlet independiente del test oficial; **solo** se lee en la reevaluación final
  (`final_front.csv`), con reentrenamiento desde cero y semillas de inicialización independientes (1000+).
- La accuracy se mide en el `val` del nodo evaluador; con `val_ratio=0.3` y C pequeño el error estándar del estimador
  es del orden de 2–3 pp (se mide en E2, no se ignora).

## Experimentos y métricas
| ID | Pregunta | Métrica principal | Comparadores |
|----|----------|-------------------|--------------|
| E0 | ¿Correcto? | tests; C=N equivale a participación completa | — |
| E1 | ¿Δ captura la heterogeneidad? | `eta2_label_dist`, ARI vs clusters por etiquetas | oracle de etiquetas, clusters aleatorios |
| E2 | ¿Mejor estimador que muestreo aleatorio al mismo coste? | sesgo, RMSE, Kendall τ de `acc_rep` vs `mon_acc_mean` | `random`, `label_cluster`, `random_cluster` |
| E3 | ¿Mejor frente? | HV del frente final en test (reentrenado) y `hv_monitor` vs evaluaciones/coste | `random` + NSGA-II, `label_cluster`, búsqueda aleatoria, `full` (N=100) |
| E4 | ¿Escala con N? | calidad vs N, coste de búsqueda constante, probe amortizado | `random` |
| E5 | Sensibilidad | α, C, rondas, método de clustering, épocas del probe | — |
| E6 | ¿La evolución evoluciona? | curva HV vs random search, éxito de hijos, diversidad | `search=random` |

## Regla de decisión (fijada antes de correr)
CS-NAS se considera útil si, a igual coste de entrenamiento y con **≥ 10 semillas pareadas**:
1. E2: reduce el RMSE (y mejora el τ) del estimador frente a `random`, con IC bootstrap al 95 % de la diferencia
   pareada que excluya 0; **y**
2. E3: el HV del frente final reevaluado en test supera al de `random` + NSGA-II con Wilcoxon pareado p < 0.05 y una
   diferencia mayor que el ruido de reentrenamiento (SD del HV al repetir las semillas de init).
Si no se cumple, el resultado es "no se detecta ventaja" y se reporta tal cual. Las semillas son la unidad de
replicación; individuos y generaciones no son réplicas independientes.

## Pendiente (no implementado)
Linaje padre-hijo, FEMNIST, estudio de fidelidad con FedAvg completo en N=100 (entrenar con todos vs con reps),
análisis estadístico agregado entre corridas.


## Adenda E5 (2026-10-22): un objetivo por cluster con NSGA-III
`evolution.objectives` = `per_node` (una accuracy por cluster, del nodo evaluador) o `per_cluster_mean` (media de los miembros no entrenados del cluster, ablación) con
`evolution.search = nsga3` (direcciones de Das–Dennis, población = nº de direcciones). El tamaño de la red deja de ser objetivo. Hall of fame con C objetivos:
`final.hof = select_hof` (ver `docs/E5_DESIGN.md`). Diseño, métricas y riesgos en `docs/E5_DESIGN.md`.


## FEMNIST (partición natural)
Cliente = escritor, sin Dirichlet. Cada escritor tiene split local train/val/test; no hay test oficial particionado como en CIFAR-10/Fashion-MNIST. Ver `docs/TRANSFER_DESIGN.md`.

## Espacio de arquitecturas v2 (genoma CNN con slots y capas FC)
Cambio de semántica del experimento, documentado antes de implementarlo.
- **Vector (todo en [0,1]):** `[conv_1..conv_B, fc_1..fc_F, kernel, activation, dropout, use_bn]`, `n_var = B + F + 4` (B = `max_blocks`, F = `max_fc`).
  Desaparece el gen `n_blocks`: la profundidad sale de los slots `conv_i`.
- **Slots:** `conv_1` siempre activo. `conv_i` (i >= 2) y `fc_j` están apagados si `v < off_threshold` (0.25); si no, `pick((v - t)/(1 - t), n)` sobre
  `channels` / `fc_widths`. Los slots apagados se saltan (la lista se compacta). Kernel y activación son compartidos por todos los bloques.
- **Red:** `[Conv -> BN? -> act -> MaxPool] x bloques -> AdaptiveAvgPool2d(pool) -> Flatten -> [Dropout -> Linear(w) -> act] x fc -> Dropout -> Linear(clases)`.
  Con `fc = []` y `pool = 1` es idéntica a la red anterior (la sonda no cambia: 1610 parámetros, mismos Δ y clusters).
- **Configuración:** `space.max_fc` (0 = sin FC), `space.fc_widths`, `space.pool` (1 = pooling global; 2 = entrada FC de 4·ch), `space.off_threshold`.
  `off_threshold = 0.25` y `pool = 2` son elecciones propias, no derivadas de un experimento.
- **`max_n_params`** incluye las FC: cambia la normalización del 2º objetivo (parámetros). Los resultados con este espacio **no son comparables** con E3/E5
  (que usaron el vector v1; ese código sigue en la rama `experiment/cluster-sampling-nas`). `cs_e3`, `cs_e5`, `cs_fmnist` y `cs_e5_smoke` ya no reproducen E3/E5 exactamente.
- **Redundancia aceptada:** `(16, off, 32)` y `(16, 32, off)` dan la misma red con genotipos distintos.
