# E3 — diseño (fijado antes de lanzar, 2026-10-19)

Pregunta: a igual coste de entrenamiento, ¿el frente accuracy-vs-parámetros que encuentra CS-NAS es mejor que el de las
alternativas, y la evolución aporta algo sobre muestrear al azar?

Configuración común (`configs/cs_e3.yaml`): Fashion-MNIST, N=100, α=0.1, C=10 representantes por evaluación, evaluador
retenido, Adam lr 1e-2, 20 rondas × 2 épocas, espacio CNN (1–3 bloques; 8–64 canales), población 10, 8 generaciones
(10 evaluaciones en la gen. 0 y 20 en cada una de las 7 siguientes = 150 evaluaciones; 150 × 10 nodos × 20 rondas = 30 000 entrenamientos locales de búsqueda).
Semillas 0–4 (misma semilla para partición, probe y sampler en los tres métodos; **los dos NSGA-II comparten la población inicial, la búsqueda aleatoria no**: parte de otras 10 arquitecturas → comparación pareada por semilla, con esa salvedad).

| Método | Muestreo de nodos | Búsqueda |
|---|---|---|
| `cs_nsga` (propuesta) | un representante por cluster de Δ | NSGA-II |
| `rand_nsga` | C=10 nodos al azar por generación | NSGA-II |
| `cs_rand` | un representante por cluster de Δ | random search (mismo nº de evaluaciones) |

Contrastes: `cs_nsga` vs `rand_nsga` (¿ayuda el muestreo por clusters?) y `cs_nsga` vs `cs_rand` (¿ayuda la evolución?).
El coste de búsqueda es idéntico (mismos entrenamientos locales); el probe de `cs_*` (100 entrenamientos de 1 época) se
reporta aparte.

Métrica primaria: hipervolumen (punto de referencia fijo (−acc=0, params_norm=1.05)) del **frente final reevaluado en
test**: los ≤5 mejores del frente de salida se reentrenan desde cero con participación completa (los 100 nodos) y una
semilla de inicialización independiente, y se mide la accuracy media por nodo en `test`.
Secundarias: HV por generación con la accuracy de todos los nodos (`hv_monitor`) frente a evaluaciones; error del estimador
durante la búsqueda; éxito de hijos; diversidad.

Análisis: diferencias pareadas por semilla del HV de test con IC bootstrap (n=5, descriptivo). No se declara ventaja si el IC
incluye 0. Limitaciones asumidas: 1 semilla de reevaluación por arquitectura, ≤5 arquitecturas por frente, 1 dataset/α,
solo C=10.
