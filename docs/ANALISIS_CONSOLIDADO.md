# Análisis consolidado de los experimentos de CS-NAS (2026-10-21)

Cubre todo lo ejecutado desde el cambio de rumbo (propuesta anterior archivada en el tag `legacy-final`): estudios de
estimador E2 (CIFAR-10 y Fashion-MNIST), diagnóstico de entrenamiento, E3 (búsqueda), diagnóstico de paisaje y estudio de
ruido. Detalles y tablas en `docs/RESULTS_*.md`; diseño en `docs/PROTOCOL.md` y `docs/E3_DESIGN.md`.

## 1. Qué se ejecutó

| Experimento | Régimen | Réplicas | Pregunta |
|---|---|---|---|
| Smoke (Fashion-MNIST, N=30) | 3 gen., pob. 4 | 1 | ¿Funciona el código? |
| E2 piloto (CIFAR-10, N=100) | Adam 1e-3, 2 rondas | 12 arq. × 2 sorteos, 1 semilla | Primera señal; mide sesgo del evaluador |
| E2 CIFAR-10 | Adam 1e-3, 8 rondas × 2 ép. | 24 arq. × 3 sorteos, 5 semillas | ¿Δ mejora el estimador? |
| E2 Fashion-MNIST (v1) | Adam 1e-3, 5 rondas | 16 × 2, 3 semillas, α ∈ {0.1, 0.5} | Idem en otro régimen |
| Diagnóstico de entrenamiento | 40 configuraciones | 1 partición | ¿Qué configuración hace que aprendan? |
| E2 Fashion-MNIST definitivo | Adam 1e-2, 20 rondas × 2 ép. | 16 × 2, 5 semillas, α ∈ {0.1, 0.5} | ¿Δ mejora el estimador (con modelos que aprenden)? |
| **E3** | misma config., pob. 10 × 8 gen. | 3 métodos × 5 semillas | ¿Mejores frentes a igual coste? ¿Aporta la evolución? |
| Paisaje (datos de E3) | — | 750 evaluaciones/método | ¿Por qué no aporta la evolución? |
| Ruido | C ∈ {10, 20, 40} × {clusters, aleatorio} | 20 arq. × 4 sorteos | ¿Cuánto ruido y cómo baja? |

## 2. Hallazgos, de más a menos sólidos

1. **El régimen de entrenamiento determinaba todo.** Con Adam 1e-3 y pocas rondas la accuracy media era 0.18–0.27 y los
   estudios previos medían ruido de arranque; con Adam 1e-2 y 20 rondas llega a 0.70–0.78 (α=0.5) y ≈0.33–0.53 (α=0.1) con 10
   nodos. Todo lo anterior a este diagnóstico solo es válido como historia del proceso (el piloto con 24 evaluaciones sugería una
   ventaja de Δ de 0.018 vs 0.040 de RMSE que se disipó con más datos).
2. **Evaluar en otro nodo del cluster (evaluador retenido) elimina el sesgo optimista** y reduce el RMSE a casi la mitad,
   en CIFAR-10 y Fashion-MNIST, α=0.1 y 0.5, para cualquier muestreo (p. ej. 0.080 → 0.047 con α=0.1). Es el resultado más
   reproducido (4 estudios) y no depende de cómo se agrupe.
3. **Los clusters por Δ capturan estructura de etiquetas sin ver etiquetas**: η² 0.48–0.71 (oracle 0.62–0.80, azar ≈0.09);
   con α=0.1 igualan al oracle en RMSE de evaluación (0.047 vs 0.047).
4. **El estimador por Δ es más exacto, no más ordenador.** Menor error absoluto de evaluación que nodos aleatorios
   (α=0.1: +0.047 de RMSE en contra de aleatorios, IC [0.024, 0.076], 5/5 semillas), pero Kendall τ idéntico (0.85–0.96).
   En CIFAR-10 con modelos sin aprender no se detecta ninguna ventaja.
5. **La propuesta no mejora la búsqueda (E3).** HV del frente reevaluado en test: `cs_nsga` 0.795, `rand_nsga` 0.804,
   `cs_rand` 0.794. `cs_nsga − rand_nsga` = −0.0087 [−0.0123, −0.0045], pierde en 5/5 semillas. La desviación pareada es
   pequeña (0.005), así que este déficit (~1 % del HV) es consistente, no ruido.
6. **No se detecta que la evolución aporte** (`cs_nsga − cs_rand` = +0.001 [−0.021, +0.021]; curvas de HV casi idénticas),
   **pero el diseño no tiene potencia para efectos pequeños**: la diferencia mínima detectable con 5 semillas es ≈0.046
   (5.7 % del HV). Un efecto de la evolución menor que eso no se habría visto.
7. **Por qué: paisaje con estructura, pero fitness demasiado ruidoso.** Las buenas arquitecturas son raras (3 % a ≤0.05 de la mejor)
   y los genes explican el 65 % de la varianza (n_bloques, dropout): la hipótesis de "espacio plano" se descarta. El ruido
   por *qué nodos entrenan* es ±0.058–0.098 con C=10, comparable a la banda de las mejores (≈0.05). Con C=20 baja a 0.043 (clusters)
   / 0.064 (aleatorios) y apenas mejora hacia C=40; con una sola evaluación se acierta el orden de arquitecturas cercanas solo en
   61–72 % de los pares, sin tendencia clara con C ni con el método.

## 3. Coherencia entre experimentos

- E2 y E3 son consistentes entre sí: el estimador por clusters tiene menos error de evaluación durante la búsqueda (RMSE 0.043 vs
  0.077 en E3, igual que en E2) y, sin embargo, no produce mejores frentes.
- La explicación se apoya en el estudio de ruido: a C=10 los clusters **no** reducen el ruido de entrenamiento (0.098 vs 0.085) y
  se alejan más de la accuracy con participación completa (RMSE 0.195 vs 0.145; sesgo −0.17 vs −0.11). La ventaja de Δ aparece
  en *evaluación* y, con C ≥ 20, en *ruido de entrenamiento*; E3 se corrió con C=10, donde no la tiene. Esto es una hipótesis
  coherente, no una prueba (E3 con C=20 no se ha corrido).
- `hv_rep` (HV con la accuracy estimada por los representantes) favorece a `rand_nsga` (0.837 vs 0.771) y se invierte con la
  accuracy de todos los nodos y con test: optimismo por selección sobre un estimador más ruidoso (efecto del ganador). Conviene no
  usar `hv_rep` como evidencia.

## 4. Discrepancias y cabos sueltos

- Ruido/señal con C=10: ≈0.32 (datos de E3) frente a 0.65–0.70 (estudio de ruido). Conjuntos de arquitecturas distintos (el segundo
  tiene muchas malas, más inestables) y pocas observaciones por celda; no está reconciliado.
- Un error propio en `analyze_noise.py` (τ frente a la referencia sin alinear arquitecturas) se detectó y corrigió antes de
  reportar; los números de `RESULTS_NOISE.md` son los corregidos.
- Los estudios E2 aplicaron lr 1e-2 también al probe (cambia los Δ respecto a lr 1e-3; los clusters por Δ siguieron funcionando).

## 5. Limitaciones que condicionan cualquier conclusión

- 5 semillas por contraste (3 en E2 v1); los IC bootstrap sobre 5 valores son descriptivos y el Wilcoxon no baja de p = 0.0625.
- El frente final de E3 se reevalúa con 1 semilla de inicialización y ≤5 arquitecturas por frente.
- Un dataset por régimen (Fashion-MNIST en E3), un solo α en E3 (0.1), C=10 en E3 y espacio CNN pequeño.
- E2/ruido evalúan arquitecturas aleatorias, no las que genera una búsqueda; E2 mide el error de evaluación, no el efecto de entrenar
  solo con los representantes (que sí se midió después en el estudio de ruido, con 1 partición).
- La rotura de empate de `random_nodes` evaluador/entrenadores y la ponderación por tamaño de cluster no se han ablacionado.

## 6. Qué se puede afirmar hoy

| Afirmación | Respaldo |
|---|---|
| Evaluar en un nodo retenido del cluster reduce sesgo y RMSE | Fuerte (4 estudios) |
| Los clusters por Δ aproximan a los de etiquetas sin etiquetas | Fuerte en α=0.1; moderada en α=0.5 |
| El estimador por Δ tiene menor error absoluto que muestrear al azar | Moderada (Fashion-MNIST, 5 semillas) |
| CS-NAS (C=10) no mejora el frente frente a nodos aleatorios | Moderada-fuerte (5/5 semillas, déficit ~1 %) |
| La evolución aporta sobre random search | No demostrada; no se podría ver por debajo de ~5.7 % de HV |
| Subir C a 20 mejora la búsqueda | No probado (solo reduce el ruido de fitness) |

## 7. Opciones

1. **Repetir E3 con C=20** (el doble de coste, ~60 000 entrenamientos locales por corrida); es la única prueba directa de que el
   ruido era el cuello de botella. Probablemente con más semillas si se quiere potencia para la evolución.
2. **Fitness multifidelidad (successive halving / reevaluar a los mejores con más nodos)**, que ataca la discriminación fina.
3. **Cerrar como resultado negativo**: evaluador retenido + estimador por Δ + diagnóstico de ruido + revisión sistemática, con
   los límites arriba.
