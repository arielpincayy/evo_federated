# Estudio de ruido del fitness por nodos de entrenamiento (2026-10-21)

`run_noise_study.py` + `analyze_noise.py`. Fashion-MNIST, N=100, α=0.1, Adam 1e-2, 20 rondas × 2 épocas, 1 partición (semilla 0).
20 arquitecturas aleatorias fijas; para cada una y cada variante se entrenó con 4 sorteos independientes de los nodos de
entrenamiento y se midió la accuracy del modelo en todos los nodos. Referencia: participación completa (100 nodos), 1 vez por
arquitectura (acc. completa: media 0.527, SD entre arquitecturas 0.159, rango 0.155–0.803). Crudos: `runs/noise/` (no versionado).
Aviso: solo 20 arquitecturas, 4 sorteos, una partición; las SD y los τ son estimaciones ruidosas (sin IC).

| variante | SD dentro (misma arq., otros nodos) | ruido/señal | τ test-retest | τ vs completa (1 sorteo) | sesgo vs completa | RMSE vs completa |
|---|---|---|---|---|---|---|
| clusters Δ, C=10 | 0.098 | 0.70 | 0.75 | 0.82 | −0.166 | 0.195 |
| clusters Δ, C=20 | 0.043 | 0.28 | 0.80 | 0.83 | −0.083 | 0.101 |
| clusters Δ, C=40 | 0.037 | 0.24 | 0.83 | 0.85 | −0.067 | 0.082 |
| nodos aleatorios, C=10 | 0.085 | 0.65 | 0.75 | 0.81 | −0.113 | 0.145 |
| nodos aleatorios, C=20 | 0.064 | 0.44 | 0.82 | 0.81 | −0.064 | 0.095 |
| nodos aleatorios, C=40 | 0.050 | 0.32 | 0.83 | 0.85 | −0.026 | 0.063 |

Mitad superior por accuracy completa (10 arquitecturas, 0.51–0.80) y 30 pares con |Δ acc. completa| < 0.05:

| variante | τ vs completa, mitad superior | acierto en pares cercanos |
|---|---|---|
| clusters Δ C=10 / 20 / 40 | 0.76 / 0.87 / 0.87 | 0.72 / 0.64 / 0.66 |
| nodos aleatorios C=10 / 20 / 40 | 0.81 / 0.81 / 0.78 | 0.65 / 0.61 / 0.67 |

## Lectura
1. **El ruido por nodos de entrenamiento baja mucho de C=10 a C=20** (clusters: 0.098 → 0.043; aleatorios: 0.085 → 0.064) y poco de
   20 a 40. C=10 es el punto más ruidoso.
2. **Agrupar por Δ reduce ese ruido solo con C ≥ 20** (0.043 vs 0.064 y 0.037 vs 0.050); con C=10 no (0.098 vs 0.085).
3. **Pero el ranking frente a participación completa casi no mejora con C** (τ 0.81–0.85 en todas las variantes) y, entre las
   arquitecturas cercanas, acertar el orden ronda 61–72 % sin tendencia clara con C ni con el método. Con una sola evaluación, las
   diferencias finas entre buenas arquitecturas se resuelven poco mejor que al azar.
4. **Entrenar con pocos nodos subestima la accuracy** (−0.03 a −0.17); más que con nodos aleatorios en el caso de clusters, y se
   reduce con C.
5. El ruido/señal con C=10 aquí (≈0.65–0.70) es mayor que el estimado con los datos de E3 (≈0.32); son conjuntos de arquitecturas
   distintos (aquí incluye muchas malas, más inestables) y estimaciones con pocos datos.

## Implicación
Subir C a 20 reduce a menos de la mitad el ruido de entrenamiento y es el punto de rendimientos decrecientes, pero no resuelve la
discriminación entre las mejores arquitecturas, que es lo que necesitaría la evolución para superar a random search.
