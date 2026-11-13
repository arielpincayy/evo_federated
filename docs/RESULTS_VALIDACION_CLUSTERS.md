# Validación de los clusters por Δ: PCA, transferencia entre nodos, FEMNIST y DBSCAN (2026-11-13)

Tres estudios sobre los clusters que forma `cluster_delta` (k-means esférico sobre Δ de un modelo sonda, C=10, N=100 nodos).
Ninguno entrena arquitecturas del NAS. Scripts: `run_pca_study.py`, `run_transfer_study.py`, `analyze_hsf.py`, `run_dbscan_study.py`. Diseño preregistrado de los
estudios de transferencia: `docs/TRANSFER_DESIGN.md`. Salidas en `runs/` (no versionado, se regeneran con los scripts).

## Resumen
1. **PCA antes del clustering no ayuda**: es igual o peor que Δ crudo. La pérdida viene de centrar, no de reducir dimensiones.
2. **Los clusters Δ agrupan nodos funcionalmente similares** (CIFAR-10 + Dirichlet): un modelo entrenado en un nodo rinde 10–35 puntos de accuracy
   más en nodos de su cluster que en otros. Δ recupera entre 76 % y 89 % de la ganancia del oráculo con etiquetas.
3. **En FEMNIST (partición natural) el efecto es pequeño pero real** (3,3 puntos) y Δ iguala al oráculo. Controlando por la distribución de
   etiquetas, Δ conserva una señal de ~1 punto (exploratorio).
4. **Agrupar por `hsf_id` transfiere casi tan bien como Δ**, pero son agrupaciones distintas y Δ rinde algo mejor (exploratorio).

## 1. PCA vs Δ crudo (CIFAR-10, 30 seeds por celda, solo calidad de clusters)
Mismo Δ por (α, seed); k-means esférico C=10 sobre cada representación. η² = fracción de la varianza de la distribución de clases explicada por
los clusters; ARI = acuerdo con el oráculo (k-means sobre la distribución de clases).

| α | métrica | crudo | centrado sin reducir | PCA 95 % var. | PCA 80 % var. | PCA k=5 |
|---|---|---|---|---|---|---|
| 0.1 | η² | **0.696** | 0.510 | 0.520 | 0.518 | 0.395 |
| 0.1 | ARI oráculo | **0.609** | 0.307 | 0.316 | 0.320 | 0.229 |
| 0.5 | η² | 0.468 | 0.447 | 0.450 | 0.452 | 0.410 |
| 0.5 | ARI oráculo | 0.311 | 0.287 | 0.292 | 0.289 | 0.231 |
| 1.0 | η² | 0.424 | 0.404 | 0.413 | 0.418 | 0.381 |
| 1.0 | ARI oráculo | 0.252 | 0.232 | 0.240 | 0.248 | 0.212 |

- Con heterogeneidad fuerte (α=0.1) el crudo gana por mucho (brecha de 0.19 en η²). Con α=0.5 y 1.0 la diferencia es de 0.01–0.03 (sin test pareado).
- "Centrado sin reducir" rinde igual que el PCA: la pérdida viene del centrado, no de la reducción. Reducir demasiado (k=5) empeora siempre.
- Por qué el centrado perjudica no está comprobado.

## 2. Transferencia entre nodos (CIFAR-10 + Dirichlet, 10 seeds por α, 5 épocas)
`A[s,j]` = accuracy en el val del nodo j de un modelo entrenado solo en el nodo s (arquitectura distinta de la sonda).
`gap` = accuracy media dentro del cluster − fuera. Aleatorio = 0 por construcción.

| α | acc. media | gap Δ (IC95 bootstrap) | gap oráculo | Δ / oráculo (mediana) | seeds con p<0.05 | Holm-p |
|---|---|---|---|---|---|---|
| 0.1 | 0.134 | **0.348** (0.335–0.362) | 0.394 | 0.89 | 10/10 | 0.003 |
| 0.5 | 0.170 | **0.166** (0.152–0.179) | 0.210 | 0.81 | 10/10 | 0.003 |
| 1.0 | 0.182 | **0.100** (0.092–0.106) | 0.133 | 0.76 | 10/10 | 0.003 |

H1 se apoya en los tres α. Holm-p = 0.003 es el mínimo alcanzable con 10 seeds (todas en la misma dirección). El gap mide sobre todo solapamiento de
clases entre nodos. Exploratorio: el coeficiente de `mismo_cluster` controlando por la distancia de Hellinger entre etiquetas es 0.077 / 0.075 / 0.048.

## 3. Réplica en FEMNIST (partición natural por escritor, 20 seeds, 50 épocas)
100 escritores por seed, sin Dirichlet. Mediana de 38 de 62 clases por nodo (clase dominante 9 %) y 92 muestras de entrenamiento. Se fijaron 50 épocas
con una seed fuera del estudio (100) según la accuracy en el propio nodo (5 ép.: 6.6 %, 20: 7.9 %, 50: 17.3 %; azar 1.6 %), no según el gap.

| | gap Δ (IC95) | gap oráculo | gap aleatorio | seeds con p<0.05 | Wilcoxon Δ vs azar |
|---|---|---|---|---|---|
| FEMNIST | **0.033** (0.030–0.035) | 0.030 | 0.0002 | 20/20 | 9.5e-7 |

- Accuracy media 0.097. Δ supera al oráculo en 16 de 20 seeds (diferencia media +0.0026).
- Exploratorio: coeficiente de `mismo_cluster` controlando por Hellinger = **0.0101** (IC95 0.0046–0.0152; positivo en 16/20; Wilcoxon p=0.0018).

## 4. `hsf_id` vs Δ en FEMNIST (exploratorio, post hoc)
`hsf_id` = serie de formularios NIST; cada escritor pertenece a una de 7. Mismas matrices A, sin reentrenar.

| agrupación | gap (IC95) | seeds con p<0.05 |
|---|---|---|
| `hsf_id` (7 grupos) | 0.0307 (0.0288–0.0325) | 20/20 |
| Δ, C=10 | 0.0327 (0.0301–0.0354) | 20/20 |
| Δ, C=7 | **0.0348** (0.0319–0.0375) | 20/20 |

- Con C=7, Δ > hsf en 16/20 seeds (Wilcoxon bilateral p=0.0002); con C=10, 11/20 (p=0.076).
- ARI entre Δ y hsf: 0.14 (C=10) / 0.17 (C=7): son agrupaciones distintas.
- Regresión conjunta `A ~ mismo_Δ + mismo_hsf + Hellinger`: coeficiente Δ 0.0100 (0.0045–0.0150), hsf 0.0059 (0.0051–0.0068).

## 5. DBSCAN (con/sin PCA) vs k-means sobre Δ (exploratorio)
Mismas matrices A, sin reentrenar (CIFAR-10: 3 α × 10 seeds; FEMNIST: 20 seeds). Δ crudo, PCA 95 % y PCA 80 %; DBSCAN con coseno, `min_samples=5`,
eps = cuantil q ∈ {0.3, 0.5, 0.7, 0.9} de la distancia al 5.º vecino. Variante principal = mayor silueta (no supervisado). El ruido se asigna al
centroide más cercano. La comparación justa es con k-means del **mismo k** y misma representación (el gap depende de k). La reconstrucción de Δ
reproduce los clusters guardados (ARI = 1). Diseño en `docs/TRANSFER_DESIGN.md`.

| dataset | α | representación | k medio | ruido | gap DBSCAN | gap k-means (mismo k) | Holm-p |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | 0.1 | crudo | 5.7 | 25 % | 0.217 | 0.237 | 0.54 |
| CIFAR-10 | 0.1 | PCA 95 % | 2.5 | 40 % | 0.054 | 0.033 | 0.041 |
| CIFAR-10 | 0.5 | PCA 80 % | 2.8 | 20 % | 0.049 | 0.076 | 0.016 |
| CIFAR-10 | 1.0 | PCA 80 % | 2.6 | 17 % | 0.025 | 0.042 | 0.023 |
| FEMNIST | natural | crudo | 2.05 | 34 % | 0.0555 | 0.0551 | 0.54 |

(Selección de celdas; la tabla completa está en `runs/dbscan_study/summary.csv`, regenerable con el script.)

- DBSCAN encuentra pocos clusters (k medio 2.3–5.7 en CIFAR-10, 2.0 en FEMNIST) con 17–43 % de ruido. Con q alto todo se fusiona en un cluster:
  48–90 % de las corridas (seed × q) de CIFAR-10 son degeneradas; con Δ crudo en α=0.5 y 1.0 solo hay variante válida en 5 y 4 de 10 seeds.
- Frente a k-means del mismo k, DBSCAN empata (CIFAR α=0.1 crudo, FEMNIST) o pierde (CIFAR α=0.5 y 1.0 con PCA, Holm-p 0.016–0.023). La única
  celda a su favor es CIFAR α=0.1 con PCA 95 % (p=0.041), con gaps muy bajos.
- Frente a k-means C=10 los contrastes salen significativos, pero **no son interpretables** porque difiere k (en CIFAR-10 DBSCAN pierde, en FEMNIST con k=2
  gana: 0.055 frente a 0.033).
- Para el NAS (≈10 representantes) DBSCAN da 2–6 clusters grandes, lo que implica mayor varianza del estimador. No se integra como `sampling.method`.

## Límites
- Los gaps son de accuracy absoluta con modelos débiles (nodos de 90–350 muestras); en FEMNIST el efecto es de pocos puntos.
- Una sola arquitectura de medición; solo CIFAR-10 y FEMNIST.
- Con Dirichlet la similitud funcional depende casi solo de las etiquetas. En FEMNIST el coeficiente de ~1 punto sobre Hellinger es exploratorio
  (modelo lineal sobre una relación que no lo es; puede quedar confusión residual) y no identifica qué captura Δ.
- Tamaños de cluster desiguales en Δ frente a los 7 grupos de tamaño parecido de hsf no se controlaron.
- DBSCAN con un solo `min_samples` (5). No se midió el efecto de PCA, DBSCAN ni de la elección de clusters sobre el NAS ni sobre la accuracy final.
