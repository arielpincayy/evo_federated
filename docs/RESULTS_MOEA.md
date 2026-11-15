# Resultados MOEA: NSGA-III con un objetivo por cluster y el tamaño como objetivo propio, frente a búsqueda aleatoria (2026-11-15)

Diseños preregistrados: `docs/MOEA_DESIGN.md` (v1, variantes A y B, auditoría de ruido, v2 y réplica). Scripts: `run_moea.sh`, `analyze_moea.py`, `run_noise_audit.py`, `analyze_k_search.py`. FEMNIST (partición natural), espacio de arquitecturas v2,
k=2 clusters por silueta (3 objetivos: accuracy de cada cluster + tamaño), 20 rondas × 2 épocas, Adam lr=1e-3. Tablas en `runs/moea/`, `runs/moea_b/`, `runs/moea2/`, `runs/noise_audit/` (no versionadas, se regeneran con los scripts).

## Método
- **Evaluación de una arquitectura:** cada representante (uno por cluster) entrena el modelo global durante 20 rondas × 2 épocas; el servidor agrega con FedAvg ponderado por **el tamaño del cluster que representa cada nodo** (no por muestras). Con ese único modelo global se calcula el objetivo de cada cluster
  (v1 A: accuracy en otro miembro retenido; v1 B: en el propio representante; v2: media sobre todos los miembros que no entrenaron). El tamaño es el tercer objetivo (`params/param_ref`, `param_ref = 354 494`).
- **Métrica primaria:** hipervolumen, en el plano (accuracy en test, tamaño normalizado) con `HV_REF = (0, 1.05)`, de los 12 modelos del hall of fame reentrenados con los 100 nodos (pesos uniformes) y medidos en test. Convergencia: hipervolumen del archivo acumulado con las accuracies *estimadas*. Población: hipervolumen de los individuos nuevos por generación.
- **Contrastes:** pareados por semilla `nsga3 − rs`, Wilcoxon unilateral e IC95 bootstrap (descriptivo). "El MOEA optimiza mejor" solo si la primaria **y** la convergencia tienen p < 0.05. Con 5 semillas el p mínimo es 0.031 (5/5) y con una en contra 0.0625.

## 1. v1: NSGA-III no supera a la aleatoria (5 semillas, 1 988 evaluaciones)
| | A: nodo retenido | B: en los nodos que entrenan |
|---|---|---|
| hipervolumen reentrenado `nsga3` / `rs` | 0.539 / 0.558 | 0.554 / 0.573 |
| `nsga3 − rs` (IC95), semillas a favor | −0.020 [−0.053, +0.024], 1/5 | −0.019 [−0.032, −0.007], 1/5 |
| convergencia del archivo | +0.027 (5/5, p 0.031) | +0.035 (5/5, p 0.031) |
| mejor accuracy en test `nsga3` / `rs` | 0.529 / 0.562 | 0.545 / 0.584 |
| rango de tamaños del frente reentrenado | 0.12 / 0.24 | 0.14 / 0.36 |
Decisión preregistrada en ambas: no se detecta ventaja. La convergencia no es fiable: el archivo de `nsga3` incluye las reevaluaciones de los padres (maldición del ganador). Evaluar en los nodos que entrenan (B) infla las accuracies estimadas (hipervolumen del archivo 0.73 frente a 0.57), pero el hipervolumen reentrenado de ambos brazos cambió igual (+0.015), sin efecto diferencial.

**Causas medidas.** (a) Con el mismo número de evaluaciones, `nsga3` evaluó ≈690 arquitecturas distintas frente a 1 988 de la aleatoria: 65 % de repeticiones (≈49 puntos son reevaluaciones de padres y ≈16 hijos que decodifican a una arquitectura ya vista), y la selección solo usaba la última medición.
(b) Ruido: la desviación de `acc_rep` al reevaluar la misma arquitectura es 0.061, frente a 0.087 de dispersión entre individuos de una generación.

## 2. Auditoría de ruido del objetivo por cluster (24 arquitecturas × 10 sorteos de representantes × 3 federaciones)
| esquema de evaluación | ruido (SD entre sorteos) | señal (SD entre arquitecturas) | fiabilidad de un sorteo | Kendall τ con la accuracy real (100 nodos) |
|---|---|---|---|---|
| E1: un nodo retenido (v1 A) | 0.068 | 0.115 | 0.74 | 0.71 |
| E2: el propio representante (v1 B) | 0.071 | 0.201 | 0.89 | 0.79 |
| **E3: todos los miembros que no entrenaron (v2)** | **0.036** | 0.110 | **0.90** | 0.78 |
| E4: 5 miembros al azar | 0.046 | 0.111 | 0.85 | 0.76 |
El ruido por elegir el nodo de evaluación es ≈0.057 y el de cambiar los representantes que entrenan ≈0.036 (≈74 % de la varianza de E1 era de evaluación). E3 cumple el umbral preregistrado de fiabilidad (0.902 ≥ 0.90), por poco. E2 amplía la señal entre arquitecturas (sesgo dependiente de la capacidad) aunque su ranking sigue alineado.
Promediar sorteos apenas mejora el ranking (τ 0.78 → 0.82): el límite no es solo ruido, entrenar con 2 representantes ordena distinto que con 100 nodos.

## 3. v2: score en todos los miembros, sin reevaluar y sin duplicados
Objetivos `[−m_1, −m_2, tamaño]` con `m_c` = media del cluster; solo se evalúan los hijos, caché por arquitectura decodificada, duplicados eliminados a nivel de arquitectura, parada a 1 988 evaluaciones reales (≈70 generaciones; caché: ≈160 entrenamientos ahorrados por corrida). `nsga3` evalúa ahora ≈1 998 arquitecturas distintas, como la aleatoria.

`nsga3 − rs` (diferencia media; IC95 bootstrap; semillas a favor):
| métrica | semillas 0–4 | réplica 5–9 | 10 semillas (0–9) |
|---|---|---|---|
| **hipervolumen reentrenado (primaria)** | +0.038 [−0.003, +0.065], 4/5, p 0.0625 | −0.004 [−0.031, +0.019], 3/5, p 0.59 | **+0.017 [−0.009, +0.042], 7/10, p 0.12** |
| convergencia del archivo | +0.023 [+0.016, +0.030], 5/5, p 0.031 | +0.032 [+0.011, +0.050], 4/5, p 0.0625 | +0.027 [+0.016, +0.038], 9/10, p 0.002 |
| pendiente de la población | +0.0003, 4/5, p 0.0625 | +0.0004, 5/5, p 0.031 | +0.0003 [+0.0002, +0.0005], 9/10, p 0.002 |
| mejora de la población (últimas 5 gen. − g0) | +0.039, 4/5, p 0.0625 | +0.063 [+0.039, +0.081], 5/5, p 0.031 | +0.051 [+0.029, +0.072], 9/10, p 0.002 |
Por semilla (primaria): 0–4: +0.068, +0.041, +0.068, +0.053, −0.041; 5–9: +0.010, +0.013, +0.029, −0.022, −0.049. Medias por brazo (10 semillas): hipervolumen reentrenado 0.574 (`nsga3`) y 0.557 (`rs`); mejor accuracy en test 0.570 y 0.558; tamaño del frente 10.2 y 9.0; tiempo 2.4 h por corrida.
Frente al v1 A: `nsga3` +0.040 en las semillas 0–4 (0.539 → 0.579) y `rs` −0.017 (dentro del ruido entre semillas).

**Decisión preregistrada (10 semillas): mejora la población, sin ventaja demostrada en el frente final.** La réplica por separado no cumple su criterio (≥ 4 de 5 semillas a favor con media > 0 en la primaria: 3/5 y −0.004).

## Conclusiones
1. El MOEA v2 **optimiza lo que se le pide**: el archivo y la población mejoran frente a la aleatoria en 9 de 10 semillas (p = 0.002), y se replica en ambas mitades.
2. **No se demuestra una ventaja en el frente final reentrenado**: +0.038 en las 5 primeras semillas se redujo a −0.004 en la réplica (regresión a la media); sobre las 10 semillas es +0.017 con IC que incluye el 0.
3. Las correcciones del v2 (no reevaluar, sin duplicados, score en todos los miembros) cambiaron el resultado del v1 de "peor que la aleatoria" a "no distinguible o ligeramente mejor", y explicaron por qué el v1 perdía (exploración y ruido); no se separa cuál aportó cuánto.
4. Hipótesis (no probada) de por qué se diluye: el MOEA optimiza un **proxy** (2 representantes por cluster) que ordena las arquitecturas con Kendall ≈ 0.78 respecto del entrenamiento con los 100 nodos; la aleatoria también elige su hall of fame con ese proxy, y el hipervolumen reentrenado se mide con 12 modelos y una inicialización (SD entre semillas ≈ 0.04).

## Límites
Un dataset (FEMNIST), un espacio, una receta y un solo MOEA (NSGA-III); 10 semillas en total; la réplica 5–9 es una ampliación posterior al primer análisis (el p combinado es algo optimista). Las tres correcciones del v2 se aplicaron juntas, sin ablaciones. El hipervolumen reentrenado usa una sola inicialización por modelo.
Evaluar en todos los miembros relaja la premisa de no tocar a todos los nodos (solo inferencia, ≈ +50 % de tiempo por evaluación). Los umbrales de la auditoría de ruido (fiabilidad ≥ 0.90) los fijé yo.

## Siguientes pasos posibles
Mejorar la fidelidad del proxy antes de elegir el hall of fame: reordenar los ~30 mejores candidatos del archivo con más representantes o reentrenándolos con más nodos, o usar 3 representantes por cluster (auditar de nuevo el Kendall).
