# Cohesión y alternativas de agrupación (2026-10-21)

Scripts: `analyze_clusters.py` (métricas de cohesión y comparación de agrupaciones), `analyze_cluster_stability.py` (estabilidad),
`analyze_allocation.py` (reparto de representantes). Reconstruyen Δ con la misma sonda y semillas de E3 (Fashion-MNIST, N=100,
5 semillas; α=0.1 y 0.5). Salidas en `runs/clusters/` (no versionado). Sin entrenar arquitecturas.

## 1. Los clusters ya no son de tamaño uniforme
k-means esférico sobre Δ con C=10 y α=0.1: tamaños entre 5 y 17 nodos (mínimo medio 6.2, mediana 9.9, máximo medio 14.6; Gini 0.15;
0 clusters de un solo nodo). Ejemplos de E3: semilla 0 [5, 7, 8, 8, 10, 10, 12, 12, 13, 15]; semilla 4 [7, 7, 7, 8, 8, 9, 11, 12, 14, 17].

## 2. Cohesión (α=0.1, C=10; media de 5 semillas)
| agrupación | tamaño min/mediana/máx | η² etiquetas | distancia Hellinger dentro / entre | pureza clase dominante | silueta en Δ | var. relativa del estimador (proxy) |
|---|---|---|---|---|---|---|
| k-means Δ (usada) | 6.2 / 9.9 / 14.6 | 0.68 | 0.54 / 0.86 | 0.78 | 0.21 | 0.40 |
| oráculo con etiquetas | 6.4 / 9.6 / 14.6 | 0.80 | 0.50 / 0.86 | 0.96 | 0.19 | 0.24 |
| enlace promedio sobre Δ (tamaño libre) | 1.8 / 9.2 / 20.4 (0.6 singletons) | 0.65 | 0.57 / 0.86 | 0.79 | 0.21 | 0.57 |
| aleatoria | 10 / 10 / 10 | 0.10 | 0.83 / 0.83 | 0.28 | −0.09 | 1.00 |
Con α=0.5 la cohesión baja (k-means Δ: η² 0.46, proxy 0.65; oráculo: 0.62, 0.46).
Proxy (sin entrenar): varianza exacta de estimar la media de las distribuciones de etiquetas por nodo con un representante por cluster
(peso |cluster|/N) relativa a C nodos al azar; 1 = igual que el azar.

## 3. Lecturas
- Los nodos de un cluster por Δ sí están relacionados (distancia de etiquetas 0.54 dentro frente a 0.83 al azar; el oráculo logra 0.50),
  pero no están claramente separados en el espacio Δ (silueta ≈ 0.21).
- Los clusters grandes son menos cohesivos: correlación 0.38 entre tamaño y dispersión de etiquetas dentro del cluster.
- **Estabilidad moderada**: con otra inicialización de la sonda el ARI entre agrupaciones es 0.54 y con otra semilla de k-means sobre los mismos
  Δ, 0.68. Con el mismo número de hilos la reconstrucción es idéntica (ARI 1.0); con otro número de hilos puede cambiar por no determinismo
  numérico de la sonda.
- **Dar libertad total de tamaño no ayuda**: el enlace promedio genera singletons y clusters de hasta 20 nodos, con peor proxy (0.57 frente a 0.40).
- **Repartir más representantes a los clusters grandes/heterogéneos (asignación de Neyman) con presupuesto fijo tampoco ayuda**: con 10
  representantes, 5 clusters con Neyman dan 0.67 y 10 clusters con uno cada uno 0.40; con 20 representantes, 10 clusters con Neyman (0.35)
  equivalen a 20 clusters con uno cada uno (0.36).
- Subir C reduce el proxy (C=20: 0.33; C=40: 0.27) a costa de singletons (≈13 de 40 con C=40).

## 4. Límites
El proxy usa la distribución de etiquetas como sustituto de lo que determina la accuracy de un nodo; no incluye el ruido de entrenamiento, que fue
el limitante en E3. Solo Fashion-MNIST, N=100 y 5 semillas.
