# ¿El ranking de arquitecturas cambia según el nodo? (preregistro, escrito antes de correr)

Script: `run_arch_node_study.py`; estadísticos: `src/evofederated/metrics/arch_node.py`. Salidas en `runs/arch_node/` (no versionado).

## Pregunta
¿Hay arquitecturas que funcionan mejor en unos nodos que en otros (interacción arquitectura × nodo), o rinden igual de bien/mal en todos?
Régimen: **modelo global FedAvg** de cada arquitectura (100 nodos, participación completa), medido en cada nodo. Solo FEMNIST (partición natural).
La pregunta importa para el NAS: sin interacción, el ranking global basta y los objetivos por cluster/nodo no aportan.

## Diseño
- **Arquitecturas:** 24 genomas distintos (se deduplica por estructura, sin contar el dropout) de `CNNGenome.from_vector(RandomState(0).rand(.., n_var), space)`
  con el `space` de `configs/cs_femnist_natural.yaml` (espacio v2: slots conv con "apagado", hasta 2 FC, `pool=2`). Las mismas 24 en todas las federaciones.
- **Federaciones:** semillas de datos 0, 1, 2 (otros 100 escritores cada una). Clusters Δ (C=10) reutilizados de `runs/transfer_femnist/A_alphanatural_seed{s}.npz`.
- **Réplicas de entrenamiento:** 2 (`init_seed` 1000 y 1001: cambian init y orden de datos).
- **Receta:** FedAvg, 20 rondas × 2 épocas, pesos uniformes, Adam. El `lr` se elige con un piloto en la semilla 100 (fuera del estudio), 3 arquitecturas, regla fija:
  mayor accuracy media en val entre {1e-3, 1e-2}; empate (diferencia < 0.5 pp) → 1e-3. No se mira nada de la interacción.
- **Medición:** accuracy y n por nodo en val y en test (splits locales disjuntos).

## Estadísticos (por semilla de datos; la unidad de replicación es la federación)
Doble centrado `R = Y − media por arquitectura − media por nodo + media global`, por (réplica, split).
1. **Interacción estable:** `V = media_celdas(R[r, val] · R[r', test])`, r ≠ r', promediando ambas direcciones. Estimador sin sesgo de la varianza de la interacción
   verdadera: el ruido de entrenamiento y el de evaluación son independientes entre los dos factores. Se reporta `SD_int = sqrt(max(V, 0))` (pp), junto a la SD del
   efecto de arquitectura y la del efecto de nodo, y `V / Var(R)`.
2. **Prueba:** permutación de nodos (columnas) de `R[r', test]` (10 000), p unilateral (V > 0); IC95 bootstrap de nodos (2 000). Entre semillas: combinación de Fisher.
3. **Consecuencia práctica (validación cruzada):** elegir con val de la réplica r, evaluar en test de la réplica r'. Ganancia = accuracy eligiendo **por nodo** (y **por cluster Δ**)
   − accuracy de la arquitectura global elegida con todos los nodos (pp). Con interacción nula sale negativa (se elige ruido); > 0 solo con estructura real. IC95 bootstrap de nodos.
4. **Control de escala:** repetir en logit, `logit((k + 0.5)/(n + 1))`. Si la interacción desaparece, era un efecto de suelo/techo de la escala de probabilidad.
5. **Contexto:** Spearman del ranking de arquitecturas entre réplicas y entre semillas (si todas rindieran igual, la pregunta no tendría sentido).
- **Decisión:** "existe interacción" si Fisher p < 0.05 y el IC95 de V es > 0 en las 3 semillas (en la escala de accuracy); su tamaño se compara con el efecto de arquitectura.
  Si la ganancia por cluster/nodo es ≤ 0, se informa que, aunque exista interacción, no es aprovechable con estos tamaños de muestra.

## Límites
24 arquitecturas aleatorias del espacio v2 y una receta; el test por nodo es pequeño (~36 muestras: estimador insesgado pero ruidoso); solo FEMNIST y un `lr`.
No responde a "qué arquitectura se ajusta mejor a los datos locales de un nodo" (entrenamiento local).
