# E4 — refinamiento multifidelidad del frente de salida (fijado antes de lanzar, 2026-10-21)

Motivación: el diagnóstico de ruido (`RESULTS_NOISE.md`) muestra que con C=10 el fitness de una arquitectura varía ±0.06–0.10 según
qué nodos entrenan, comparable a la banda de las mejores. El frente de salida de E3 se elige con una sola estimación ruidosa por
arquitectura (efecto del ganador). Hipótesis: reevaluar solo a los candidatos prometedores con más nodos (C=40, nodos nuevos)
mejora el frente final, sin repetir la búsqueda.

Procedimiento (`run_refine.py`, sobre las 15 corridas de E3 ya hechas; no se repite ninguna búsqueda):
1. Candidatos: capas no dominadas del conjunto de todas las arquitecturas evaluadas (accuracy estimada media por genoma, parámetros),
   hasta 12 arquitecturas.
2. Reevaluación: C=40 nodos aleatorios nuevos (RNG independiente del de la búsqueda), mismo entrenamiento (Adam 1e-2, 20 rondas × 2
   épocas) y evaluador retenido. El mismo muestreo para todos los métodos, para que las diferencias entre métodos vengan de la búsqueda.
3. Frente final = no dominado por (accuracy refinada, parámetros), ≤5 arquitecturas por accuracy, igual criterio que E3; se reentrena
   desde cero con participación completa (100 nodos, semilla de inicialización 1000) y se mide en test.

Coste añadido por corrida (mismo para los tres métodos): ≤12 candidatos × 40 nodos × 20 rondas = 9 600 entrenamientos locales
(32 % del coste de búsqueda de 30 000), más la reevaluación final igual a la de E3.

Contrastes, pareados por semilla y método sobre HV en test (punto de referencia fijo, mismo que E3):
- H1 (valor del refinamiento): HV refinado − HV original de E3, dentro de cada método.
- H2 (¿aparecen diferencias entre métodos con el ruido de selección reducido?): `cs_nsga − rand_nsga` y `cs_nsga − cs_rand` tras el
  refinamiento.
No se declara efecto si el IC bootstrap (n=5) incluye 0. Limitaciones asumidas: una sola reevaluación por candidato, una semilla de
inicialización en la evaluación final, el candidato set depende de la búsqueda original (no se repite), un dataset/α/C.
La diferencia mínima detectable de E3 para la evolución (≈5.7 % del HV con 5 semillas) sigue aplicando.
