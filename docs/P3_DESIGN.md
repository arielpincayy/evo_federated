# P3: k óptimo (≤ 10) + NSGA-III con un objetivo por cluster = accuracy penalizada por tamaño (preregistro, escrito antes de correr)

Propuesta del usuario. Es E5 (`n3_cs`, ver `docs/E5_DESIGN.md`) más selección de k y tamaño dentro del objetivo: E5 quitó el tamaño y las redes derivaron a tamaños mayores.
"Ronda" = generación: en cada generación se re-sortea un representante por cluster (`assignment(generation)`, ya existente) y cada individuo se mide en todos los clusters.

## Decisiones
- **k:** el k ∈ {2..`sampling.k_max`=10} que maximiza la silueta coseno de k-means esférico sobre Δ (`choose_k`; no supervisado, no mira A, etiquetas ni accuracy). Se guarda la tabla completa en `clusters.json`.
  Si elige un k pequeño, NSGA-III opera con pocos objetivos: es una consecuencia, no un fallo.
- **Objetivos:** `obj_c = acc(eval_c) − λ·params/param_ref`, `λ = 1` fijado a priori (`evolution.size_penalty`), `param_ref = 354 494` (máximo del espacio v2 en FEMNIST). `obj_acc` guarda el vector penalizado
  (el que optimiza la búsqueda y usan `output_front`/`select_hof`); `obj_raw` guarda las accuracies sin penalizar; `acc_rep` (media ponderada) no se penaliza. El tamaño ya no es un objetivo aparte (nº de objetivos = k).
  Con las 24 arquitecturas del estudio arquitectura × nodo: λ ≤ 1 elige la mejor (57.9 %, 69 mil parámetros), λ ≈ 1.3 empata con la de 41.9 % (27 mil) y λ = 2 elige esta última.
- **Direcciones de referencia:** `ref_partitions: 0` = automático: mayor p con C(k+p−1, p) ≤ `pop_size` (60). k=2 → 60 direcciones (p=59), 3 → 55 (9), 4 → 56 (5), 5 → 35 (3), 6 → 56 (3), 7 → 28 (2), 8 → 36 (2), 9 → 45 (2), 10 → 55 (2). La población de NSGA-III es el nº de direcciones; la búsqueda aleatoria usa el mismo nº.
- **Presupuesto:** `evolution.eval_budget: 390` fija las generaciones de cada brazo (`G = round(1 + (390 − pop)/(2·pop))`; NSGA-II base: pop 10 × 20 gen = 390). Las evaluaciones son iguales; los entrenamientos locales no (k representantes frente a 10): se reportan.
- **Brazos** (semillas de datos 0, 1, 2; misma federación y sonda en todos): `p3` (k óptimo, NSGA-III, λ=1, hof `select_hof` K=12); `base` (C=10, NSGA-II, (accuracy media, tamaño), hof `front_top` con 12);
  `rs` (mismos clusters y objetivos que `p3`, búsqueda aleatoria con las mismas evaluaciones, hof `select_hof`).
- **Receta:** FEMNIST (`configs/cs_p3.yaml`, espacio v2), Adam `lr=1e-3` (piloto del estudio arquitectura × nodo), 20 rondas × 2 épocas, evaluador retenido; cada modelo del hof se reentrena una vez (init 1000) con los 100 nodos y se mide en test.

## Métricas y decisión
- **Primaria:** hipervolumen de los modelos reentrenados del hof en (−accuracy test media, tamaño normalizado), con `HV_REF = (0, 1.05)`.
- **Secundarias:** mejor `acc − λ·tamaño` del conjunto (es el criterio de `p3`: favorece al brazo penalizado y se reporta como tal), accuracy y tamaño del mejor por accuracy, tamaño medio del conjunto, k elegido, coste (entrenamientos locales).
- **Contrastes pareados por semilla:** `p3 − base`, `p3 − rs`; IC bootstrap sobre 3 semillas, **descriptivo**. No se declara ventaja si el IC incluye 0.

## Riesgos asumidos
1. Los k objetivos serán muy correlacionados (Spearman 0.92 entre rankings de nodos) y ruidosos; NSGA-III puede parecerse a una selección por la media (riesgo ya anotado en E5).
2. La misma penalización en todos los clusters elimina el frente accuracy-tamaño: λ fija el tamaño preferido. La comparación en el plano (accuracy, tamaño) es la prueba más dura para `p3`.
3. Solo 3 semillas, un dataset, un espacio, una receta y un λ: primera señal, no conclusión.
