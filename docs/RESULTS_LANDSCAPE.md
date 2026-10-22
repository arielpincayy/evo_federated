# Diagnóstico del paisaje de fitness con los datos de E3 (2026-10-20)

`analyze_landscape.py` sobre `runs/e3/*/seed*/population.csv` (Fashion-MNIST, α=0.1, N=100, C=10; 5 semillas). La accuracy
usada es la del modelo entrenado por los representantes evaluado en **todos** los nodos (`mon_acc_mean`).

## 1. El paisaje NO es plano
- 150 arquitecturas aleatorias por semilla (`cs_rand`): acc media 0.384, SD entre arquitecturas 0.157, rango 0.064–0.756.
- Solo el 1.1 % está a ≤ 0.02 de la mejor y el 3.1 % a ≤ 0.05: las buenas son raras.
- Los genes explican el 65 % de la varianza de la accuracy (random forest, CV 5): `n_blocks` 0.55, `dropout` 0.20,
  `ch_sum` 0.10 de la importancia. Spearman(acc, parámetros) = 0.74.
Hay estructura que una búsqueda dirigida podría explotar.

## 2. El ruido de fitness es grande para distinguir entre las mejores
Misma arquitectura reevaluada en generaciones distintas (NSGA-II reevalúa a los padres con otros representantes):

| método | SD dentro de la misma arquitectura (acc. todos los nodos) | SD entre arquitecturas | acc. estimada por reps |
|---|---|---|---|
| `cs_nsga` (clusters Δ) | 0.058 | 0.181 | 0.071 |
| `rand_nsga` (nodos aleatorios) | 0.059 | 0.187 | 0.080 |

- Misma arquitectura, otros 10 nodos de entrenamiento ⇒ su accuracy global cambia ±0.058 (relación ruido/señal ≈ 0.32).
- El ruido es **igual** con clusters de Δ que con nodos aleatorios: la estratificación mejoró el error de *evaluación* (E2) pero no
  el ruido de *entrenamiento* (qué nodos entrenan), que domina.
- Las mejores arquitecturas están en una banda de ≈ 0.05; con ruido ±0.058, ordenarlas entre sí es casi al azar.

## 3. La evolución no supera a random search
HV con todos los nodos, media de 5 semillas, gen 0 → gen 7: `cs_nsga` 0.617 → 0.735; `cs_rand` 0.554 → 0.739;
`rand_nsga` 0.657 → 0.753. El archivo de random search crece igual de rápido (cumulativo).

## 4. Frentes finales (reevaluados en test, participación completa)
Acc. test del mejor del frente ≈ 0.80–0.81 en los tres métodos; los frentes están dominados por arquitecturas de ~3 bloques.

## Conclusión
La hipótesis "el espacio es demasiado fácil/plano" **se descarta**. La que mejor encaja es: el fitness de baja fidelidad
(10 nodos, α=0.1) tiene un ruido de ±0.058 por *qué nodos entrenan*, comparable a la diferencia entre las mejores
arquitecturas, y eso limita a cualquier algoritmo de selección. Pendiente de probar (no medido aún): cómo baja ese ruido con C
mayor, con representantes fijos entre generaciones o promediando evaluaciones.
