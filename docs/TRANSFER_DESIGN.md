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
