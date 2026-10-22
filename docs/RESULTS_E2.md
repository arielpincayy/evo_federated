# E2 (estimador de fitness) — resultado, 2026-10-19

Configuración: CIFAR-10, N=100, α=0.5, C=10 reps por evaluación, 8 rondas × 2 épocas, 24 arquitecturas aleatorias
× 3 sorteos de reps × 5 semillas de partición (72 evaluaciones por variante y semilla). Código:
`run_estimator_study.py`, `analyze_estimator.py`; crudos en `runs/estimator_e2/` (no versionado).

| Variante | sesgo | RMSE | Kendall τ |
|---|---|---|---|
| clusters por Δ, evaluador retenido | +0.003 | 0.031 | 0.84 |
| clusters por etiquetas (oracle), retenido | −0.002 | 0.032 | 0.83 |
| clusters aleatorios, retenido | −0.002 | 0.033 | 0.82 |
| nodos aleatorios, retenido | −0.015 | 0.039 | 0.82 |
| clusters por Δ, sin retener | +0.056 | 0.060 | 0.87 |
| nodos aleatorios, sin retener | +0.062 | 0.068 | 0.85 |

Calidad de los clusters (5 semillas): η² de la distribución de etiquetas = 0.49 (Δ), 0.62 (oracle), 0.09 (aleatorio);
ARI Δ vs oracle = 0.31 ± 0.10.

## Conclusiones
1. **Evaluar en un nodo retenido del cluster funciona** y es independiente del criterio de agrupación: RMSE 0.060 → 0.031
   (diferencia pareada +0.029, IC bootstrap 95 % [0.025, 0.033]) y sesgo 0.056 → 0.003. (Wilcoxon p = 0.0625 es el mínimo
   posible con 5 semillas.)
2. **No se detecta ventaja de agrupar por Δ.** Frente a clusters aleatorios o a nodos aleatorios con evaluador retenido, las
   diferencias de RMSE (+0.002 y +0.007) tienen IC que incluyen 0, y el τ es el mismo (≈0.82–0.84). La ventaja aparente del
   piloto de 24 evaluaciones (0.018 vs 0.040) era ruido.
3. Según la regla de decisión de `PROTOCOL.md`, el criterio 1 (E2) **no se cumple** en este régimen.

## Limitaciones del régimen probado
- Accuracy media ≈ 0.21 con 10 clases: los modelos casi no aprenden, así que el fitness refleja velocidad de entrenamiento
  más que heterogeneidad entre nodos.
- α = 0.5 es heterogeneidad moderada; la estratificación debería importar más con α = 0.1.
- El estudio mide solo el error del estimador de *evaluación* (mismo modelo en todos los nodos), no el efecto de entrenar
  solo con los representantes.

---

# E2 en régimen alternativo — Fashion-MNIST, 2026-10-19

Configuración: Fashion-MNIST, N=100, C=10, 5 rondas × 2 épocas, 16 arquitecturas × 2 sorteos (32 evaluaciones por variante y
semilla), **3 semillas** de partición, α ∈ {0.1, 0.5}. Crudos en `runs/estimator_fmnist/` (no versionado).

RMSE medio por variante (evaluador retenido salvo indicación):

| α | clusters Δ | oracle etiquetas | clusters aleatorios | nodos aleatorios | Δ sin retener |
|---|---|---|---|---|---|
| 0.1 | **0.025** | 0.041 | 0.064 | 0.049 | 0.058 |
| 0.5 | **0.031** | 0.032 | 0.039 | 0.060 | 0.056 |

η² de los clusters por Δ: 0.70 (α=0.1) y 0.48 (α=0.5), frente a 0.80 / 0.63 del oracle y 0.08 / 0.07 al azar.

Lectura:
- El evaluador retenido vuelve a reducir el RMSE y el sesgo en ambos α (p. ej. α=0.1: 0.058 → 0.025).
- Agrupar por Δ tiene el menor RMSE entre las variantes de bajo coste en ambos α: supera a nodos aleatorios (3/3 semillas en
  cada α) y a clusters aleatorios (3/3 en α=0.1; 2/3 en α=0.5), y empata con el oracle de etiquetas en α=0.5.
- Kendall τ: Δ 0.83 / 0.90 frente a nodos aleatorios 0.78 / 0.90 y clusters aleatorios 0.72 / 0.84; sin diferencia clara.

**Cautelas:** solo 3 semillas (los IC bootstrap sobre 3 valores y el Wilcoxon no son fiables; es evidencia descriptiva);
y el régimen **no llegó a ser de aprendizaje**: la accuracy media de todos los nodos fue 0.20 (α=0.1) y 0.27 (α=0.5), muy por
debajo de lo que alcanza Fashion-MNIST entrenado a fondo. Con CIFAR-10 (5 semillas, α=0.5, acc ≈ 0.21) no se vio ventaja
de Δ sobre clusters/nodos aleatorios; con Fashion-MNIST sí hay dirección favorable, así que el resultado depende del régimen
y aún no es concluyente.

---

# Diagnóstico de entrenamiento (2026-10-19)

`run_training_diagnostic.py`: arquitectura fija (2 bloques conv 16/32, BN) con FedAvg sobre C=10 nodos aleatorios,
Fashion-MNIST N=100, 1 partición; accuracy media sobre `val` de todos los nodos. Crudos: `runs/training_diagnostic.csv`.

| configuración (C=10 salvo indicación) | α=0.1 | α=0.5 |
|---|---|---|
| Adam 1e-3, 2 épocas, 5 rondas (régimen E2 previo) | 0.18 | 0.26 |
| Adam 1e-3, 2 épocas, 20 rondas | 0.20 | 0.37 |
| Adam 1e-2, 2 épocas, 20 rondas | 0.33 | 0.70 |
| Adam 1e-2, 5 épocas, 20 rondas | 0.45 | 0.78 |
| SGD 0.05, 2 épocas, 20 rondas | 0.53 | 0.65 |
| SGD 0.05, 5 épocas, 20 rondas | 0.47 | 0.78 |
| Adam 1e-2, 2 épocas, 20 rondas, sin BN | 0.32 | 0.63 |
| Adam 1e-2, 2 épocas, 20 rondas, **todos los nodos (C=100)** | 0.51 | 0.73 |

Conclusiones:
- El régimen del E2 (Adam 1e-3, pocas rondas) estaba muy subentrenado; el learning rate era el factor dominante.
- Con lr 1e-2 (o SGD 0.05) y 20 rondas los modelos llegan a 0.7–0.78 con α=0.5. Con α=0.1 el techo con 10 nodos es ≈0.33–0.53.
- C=10 frente a todos los nodos (mismo presupuesto de rondas/épocas): 0.70 vs 0.73 (α=0.5) y 0.33 vs 0.51 (α=0.1): con
  heterogeneidad alta, entrenar solo con 10 representantes pierde bastante.
- BatchNorm ayuda (0.70 vs 0.63 a α=0.5).
- La accuracy del peor nodo es ≈0 con α=0.1 en casi todas las configuraciones.

---

# E2 definitivo — Fashion-MNIST, Adam 1e-2, 20 rondas × 2 épocas, 5 semillas (2026-10-19)

N=100, C=10, 16 arquitecturas aleatorias × 2 sorteos (32 evaluaciones por variante y semilla), α ∈ {0.1, 0.5}. El lr de 1e-2
se aplicó también al probe. Accuracy media de todos los nodos sobre las arquitecturas evaluadas: 0.39 (α=0.1) y 0.59 (α=0.5).
Crudos: `runs/estimator_fmnist_lr/` (no versionado). Análisis: `analyze_estimator.py --root runs/estimator_fmnist_lr/a<α>`.

RMSE medio (menor = mejor) y Kendall τ medio dentro de cada sorteo:

| α | variante | RMSE | τ |
|---|---|---|---|
| 0.1 | clusters Δ (retenido) | **0.047** | 0.89 |
| 0.1 | oracle etiquetas | 0.047 | 0.90 |
| 0.1 | clusters aleatorios | 0.073 | 0.89 |
| 0.1 | nodos aleatorios | 0.094 | 0.85 |
| 0.1 | clusters Δ sin retener | 0.080 | 0.91 |
| 0.5 | clusters Δ (retenido) | **0.028** | 0.96 |
| 0.5 | oracle etiquetas | 0.036 | 0.96 |
| 0.5 | clusters aleatorios | 0.036 | 0.95 |
| 0.5 | nodos aleatorios | 0.049 | 0.95 |
| 0.5 | clusters Δ sin retener | 0.051 | 0.95 |

Diferencias pareadas por semilla frente a clusters Δ (RMSE; >0 = peor que Δ; IC bootstrap sobre 5 semillas, descriptivo):
- vs nodos aleatorios: α=0.1 +0.047 [0.024, 0.076] (Δ mejor en 5/5 semillas); α=0.5 +0.022 [0.003, 0.041] (3/5).
- vs clusters aleatorios: α=0.1 +0.026 [0.002, 0.047] (4/5); α=0.5 +0.008 [−0.004, 0.021] (4/5).
- vs oracle de etiquetas: +0.0001 (α=0.1) y +0.008 (α=0.5), IC incluyen 0.
- evaluador retenido vs no retenido (Δ): −0.033 (α=0.1) y −0.023 (α=0.5).
Wilcoxon con n=5 no baja de p=0.0625; no se usa como criterio.

Calidad de clusters (η² de la distribución de etiquetas / ARI vs oracle): Δ 0.71/0.66 (α=0.1) y 0.48/0.28 (α=0.5);
oracle 0.80/0.62; aleatorio ≈0.09.

## Lectura
1. **El estimador por clusters de Δ con evaluador retenido tiene menor error absoluto que muestrear nodos al azar** a igual
   coste, en ambos α (IC que excluyen 0), y con α=0.1 iguala al oracle que usa etiquetas. La ventaja sobre clusters
   aleatorios es menor e incierta (IC rozan 0 en α=0.5).
2. **No mejora el ranking.** Kendall τ es 0.85–0.96 en todas las variantes y las diferencias no son distinguibles de 0. Para
   *seleccionar* arquitecturas lo que importa es el orden, y cualquier muestreo con evaluador retenido ya lo conserva bien.
3. Criterio E2 del protocolo ("menor RMSE **y** mejor τ frente a `random`"): se cumple la parte de RMSE, **no** la de τ. Por
   tanto no está demostrado que el estimador mejore la búsqueda; eso lo decide E3.
4. Limitaciones: arquitecturas aleatorias (no las que produce una búsqueda), 3 sorteos de reps por arquitectura, 5 semillas,
   un solo dataset en este régimen, solo C=10, y solo el error de *evaluación* (no el efecto de entrenar solo con los reps).
