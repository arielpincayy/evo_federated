# Transferencia entre nodos: ¿los clusters por Δ agrupan nodos funcionalmente similares? (preregistro)

Escrito antes de correr. Script: `run_transfer_study.py`; métrica: `src/evofederated/metrics/transfer.py`.

## Hipótesis
- **H1 (principal):** un modelo entrenado en el nodo s rinde más en nodos de su mismo cluster Δ que en nodos de otros clusters.
- **H0:** pertenecer al mismo cluster Δ no se asocia con la accuracy de transferencia más allá de una agrupación al azar del mismo tamaño.
- Secundaria: fracción de la ganancia del oráculo (clusters por etiquetas) que logran los clusters Δ.

## Medición
Por (α, semilla): CIFAR-10, N=100, C=10. Un modelo (`CNNGenome(2,[16,32],3,relu,0,bn)`, distinta de la sonda, misma init para todas las
fuentes, init con seed+1000) se entrena K épocas en el train de cada nodo s y se evalúa en el **val** de cada nodo j≠s → `A[s,j]`.
Los clusters Δ usan la sonda de siempre; el modelo de medición es otra arquitectura (evita circularidad).

## Estadístico
`gap = media_s( media_{j∈c(s), j≠s} A[s,j] − media_{j∉c(s)} A[s,j] )`, en accuracy (0–1), para Δ, oráculo (etiquetas) y aleatorio.
- Por semilla: p por permutación de las etiquetas de cluster (10 000, conserva tamaños), unilateral. Los pares no son independientes, por eso no se usa un test sobre pares.
- Entre semillas (por α): media de `gap_Δ` con IC95 bootstrap; Wilcoxon pareado unilateral de `gap_Δ − gap_aleatorio`; Holm sobre los 3 α; `frac_oráculo = gap_Δ / gap_oráculo`.
- Decisión: H1 se apoya en un α si Holm-p < 0.05 **y** el IC95 de `gap_Δ` excluye 0.

## Exploratorio (no confirmatorio)
Regresión por semilla `A[s,j] ~ mismo_cluster + Hellinger(etiquetas_s, etiquetas_j)`: si `mismo_cluster` cae a ~0, Δ no aporta más que las etiquetas.

## Límites
Con Dirichlet la similitud funcional depende casi solo de las etiquetas: se valida que Δ la recupera, no que capture algo extra.
Una sola arquitectura de medición; modelos entrenados en ~350 muestras son débiles (se reporta también la accuracy media).

## Réplica en FEMNIST (partición natural; preregistro, escrito antes de correr)
Objetivo: en CIFAR-10+Dirichlet la similitud funcional la determinan casi solo las etiquetas. FEMNIST tiene clientes reales
(escritores), con *feature skew* y label skew suave (en una seed de prueba: mediana de 38 de 62 clases por nodo, clase dominante 9 %).
- **Datos:** `flwrlabs/femnist` (HF, LEAF), 62 clases, 28×28. Cada seed elige 100 escritores al azar con >= 100 muestras; split local
  train/val/test por escritor (test no se usa aquí). No hay Dirichlet: `alpha` se ignora (`--alphas natural`).
- **Medición:** igual que arriba (misma arquitectura distinta de la sonda, misma init, `A[s,j]` en val). Cambia solo `epochs`: con ~92
  muestras de train por nodo, 5 épocas dejan los modelos casi en azar. Se fijó **epochs=50** con una seed fuera del estudio (100) mirando
  la accuracy en el val del propio nodo, no el gap: 5 ép. 6.6 %, 20 ép. 7.9 %, 50 ép. 17.3 % (azar 1.6 %). Regla: menor nº de épocas del
  conjunto {5, 20, 50} con accuracy propia > 10 %.
- **Seeds:** 20 (cada una con otros 100 escritores y otra sonda). Un solo grupo, así que Holm = p crudo.
- **Estadístico y decisión:** los mismos (`gap_Δ` vs aleatorio, Wilcoxon pareado unilateral, IC95 bootstrap que excluya 0; `frac_oráculo`).
- **Pregunta central, exploratoria:** coeficiente de `mismo_cluster(Δ)` en `A ~ mismo_cluster + Hellinger`. Si es > 0 con IC que excluya 0,
  Δ captura estructura que las etiquetas no explican. Se reporta con IC bootstrap entre seeds. El oráculo por etiquetas puede dejar de ser un techo.
- **Límites:** nodos de ~92 muestras con 62 clases dan modelos débiles; una sola arquitectura; cambia la semántica del test respecto a
  CIFAR-10 (aquí no hay test oficial; solo se usa val).

## DBSCAN (con/sin PCA) vs k-means sobre Δ (exploratorio, escrito antes de correr el estudio completo)
Pregunta: ¿una agrupación por densidad (que decide sola el nº de clusters) transfiere mejor que k-means C=10? Se evalúa con las A ya calculadas
(CIFAR-10: `runs/transfer_study`, 3 α × 10 seeds; FEMNIST: `runs/transfer_femnist`, 20 seeds), sin reentrenar. Script: `run_dbscan_study.py`.
- **Representaciones:** Δ normalizado (raw), PCA 95 % y PCA 80 % de varianza. Como el PCA centró Δ y eso empeoró a k-means, `raw` separa el efecto de DBSCAN del del PCA.
- **DBSCAN:** distancia coseno, `min_samples=5`, eps = cuantil q de la distancia al 5.º vecino, q ∈ {0.3, 0.5, 0.7, 0.9} (sensibilidad). **Variante principal** por
  (representación, seed) = la q con mayor silueta coseno entre las no degeneradas (criterio no supervisado, no mira A ni etiquetas).
- **Ruido:** cada nodo de ruido se asigna al cluster de centroide más cercano (coseno). Se reporta la fracción de ruido. Degenerado = k<2: se cuenta y no entra en gap.
- **Comparaciones sobre la misma A:** k-means con el **mismo k** que DBSCAN y misma representación (comparación justa, el gap depende de k), k-means C=10 actual, oráculo.
  Comprobación: la reconstrucción de Δ reproduce las etiquetas k-means guardadas (ARI = 1).
- **Inferencia (variante principal):** Wilcoxon pareado bilateral de `gap_DBSCAN − gap_kmeans(mismo k)` y `− gap_kmeans10` por (dataset, α), con Holm sobre las 4 celdas
  (3 α CIFAR + FEMNIST) dentro de cada representación; p por permutación (2 000) por seed. Todo lo demás es sensibilidad descriptiva.
- No se integra al NAS salvo que DBSCAN supere a k-means(mismo k) y a k-means C=10.
