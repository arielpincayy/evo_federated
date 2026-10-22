# Proporción de nodos que mejoran (2026-10-22)

`run_node_level.py` reentrena los modelos y mide la accuracy **por nodo** (ningún experimento previo la guardó); `analyze_node_level.py`
calcula, para dos modelos o conjuntos A y B, la proporción de nodos que **mejoran** (acc_B > acc_A), **quedan igual** y **empeoran**.
Salidas en `runs/node_level/` (no versionado). Control de integridad: la accuracy media recalculada coincide con la guardada en
`final_front.csv` (diferencia 0.00000 en las 20 corridas).

- **E3**: población inicial (generación 0, 10 arquitecturas) frente al **hall of fame** final (frente de Pareto: no dominadas por accuracy
  estimada y parámetros entre todo lo evaluado; 6–15 por corrida) y su versión refinada (E4). Modelos entrenados con los 100 nodos
  (20 rondas × 2 épocas), medidos en test. Para conjuntos: accuracy media del nodo sobre el conjunto ("media") o la mejor ("mejor").
- **Entrenamiento y ruido**: una arquitectura por comparación, validación de los 100 nodos.

## E3 (media ± DE entre 5 semillas por método; filas = comparaciones agrupadas sobre 20 corridas)
| comparación | agregación | mejoró | igual | empeoró | cambio medio (pp) |
|---|---|---|---|---|---|
| población inicial → hall of fame | media | 60 % | 0 % | 39 % | +2.5 |
| población inicial → hall of fame | mejor | 48 % | 29 % | 22 % | +2.5 |
| por método (media): CS-NSGA / NSGA-II nodos aleat. / búsq. aleat. clusters / búsq. aleat. nodos aleat. | media | 59 / 59 / 57 / 66 % | ≈0 | 41 / 41 / 43 / 33 % | +1.8 / +3.0 / +1.8 / +3.5 |
| hall of fame → hall of fame refinado | media | 38 % | 1 % | 61 % | −1.1 |
| CS-NSGA → NSGA-II nodos aleat. | mejor | 47 % | 26 % | 27 % | +2.5 |
| NSGA-II nodos aleat. → búsq. aleat. nodos aleat. | mejor | 32 % | 24 % | 44 % | −1.6 |
| CS-NSGA → búsq. aleat. clusters | mejor | 42 % | 20 % | 38 % | +0.7 |
| búsq. aleat. clusters → búsq. aleat. nodos aleat. | mejor | 39 % | 25 % | 37 % | +0.2 |
Con la media del conjunto, el 52 % de los nodos mejora más de 2 pp y el 31 % empeora más de 2 pp. Entre semillas la proporción varía mucho
(DE 0.24–0.43). En el cuartil de nodos más débil mejora el 77 % (regresión a la media a favor).

## Entrenamiento y número de nodos
| comparación | mejoró | igual | empeoró | cambio medio (pp) |
|---|---|---|---|---|
| α=0.1: Adam 1e-3/5 r. → Adam 1e-2/5 ép./20 r. | 86 % | 4 % | 10 % | +27.3 |
| α=0.5: ídem | 100 % | 0 % | 0 % | +51.9 |
| α=0.1: 10 nodos → todos (misma configuración) | 66 % | 1 % | 33 % | +17.8 |
| α=0.5: 10 nodos → todos | 75 % | 4 % | 21 % | +3.1 |
| ruido, C=10: nodos aleatorios → clusters (media sobre 10 arq. × 2 sorteos) | 44 % | 5 % | 51 % | −1.8 |
| nodos aleatorios C=10 → C=20 / C=40 / todos | 55 / 57 / 65 % | 8 / 4 / 4 % | 37 / 39 / 30 % | +6.6 / +10.2 / +13.8 |
| clusters C=10 → C=40 / todos | 58 / 66 % | 5 / 5 % | 37 / 29 % | +10.3 / +15.6 |

## Lectura
1. Del conjunto inicial al hall of fame mejora algo más de la mitad de los nodos y empeora casi el 40 % (con "mejor": 48 / 29 / 22 %).
   La búsqueda evolutiva no mejora a más nodos que la aleatoria (media: 59 % CS-NSGA y NSGA-II con nodos aleatorios frente a 57 % y 66 % de las aleatorias).
2. Entre métodos el reparto es casi parejo; el único contraste inclinado es NSGA-II con nodos aleatorios frente a la búsqueda aleatoria con los
   mismos nodos (32 % de nodos mejoran, 44 % empeoran al pasar a la aleatoria), coherente con ser el método con mejores cifras.
3. El refinamiento no mejora a la mayoría de los nodos (38 % mejoran, 61 % empeoran).
4. Aun con participación completa, un tercio de los nodos queda peor que con 10 nodos cuando α=0.1, aunque el promedio mejora +17.8 pp.
5. Con el mismo C=10, entrenar con representantes por clusters mejora al 44 % de los nodos y empeora al 51 % frente a nodos aleatorios.

## Cautelas
Cada nodo tiene pocas muestras de prueba (decenas a cientos): cambios pequeños son ruido (de ahí las columnas de >2 pp). "Media" sobre un frente
mezcla modelos pequeños y grandes; "mejor" es más justa entre frentes. Sin intervalos más allá de la DE entre semillas/arquitecturas; el diagnóstico de
entrenamiento usa una sola arquitectura y partición. Estimador, paisaje y clusters no tienen versión por nodo.

## ¿Hay arquitecturas del hall of fame que mejoren a todos los nodos? (2026-10-22)
`analyze_node_level.py` (salidas `hof_per_model.csv`, `hof_best_vs_best.csv`). 205 arquitecturas del hall of fame en 20 corridas (20 ya estaban en la población inicial).
| referencia (población inicial) | mejoran a todos los nodos | ningún nodo peor (≥) | mediana de nodos que mejoran | máx | con ≥90 % |
|---|---|---|---|---|---|
| accuracy media de la población inicial (por nodo) | 2 | 4 | 70 % | 100 % | 86 |
| mejor modelo inicial (por accuracy media) | 0 | 6 | 8 % | 90 % | 1 |
| mejor modelo inicial de cada nodo | 0 | 0 | 2 % | 67 % | 0 |
- Los 2 modelos que mejoran a todos los nodos frente a la media inicial pertenecen a la búsqueda aleatoria con nodos aleatorios (uno era arquitectura nueva).
  Los 6 casos de "ningún nodo peor" frente al mejor modelo inicial son ese mismo modelo.
- Mejor modelo del hall of fame frente al mejor inicial: es el mismo en 5 de 20 corridas; en las 15 restantes mejora de media al 51 % de los nodos
  (21–90 %) y su accuracy media es mayor en 8 corridas y menor en 7. Ninguna corrida mejora a todos los nodos.
- Corrección: una respuesta anterior en el chat decía que la media subía en 10 de las 15; el cálculo correcto es 8 (y baja en 7).
