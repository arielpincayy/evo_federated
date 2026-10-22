# E3 — resultado (2026-10-20)

Diseño en `docs/E3_DESIGN.md` (fijado antes de lanzar). Fashion-MNIST, N=100, α=0.1, C=10, pob. 10, 8 generaciones
(150 evaluaciones; 30 000 entrenamientos locales de búsqueda por corrida, idénticos en los tres métodos), 5 semillas pareadas.
Crudos en `runs/e3/` (no versionado); análisis: `analyze_e3.py`.

| Método | HV frente final en test (primaria) | HV con acc. de todos los nodos | RMSE del estimador en la búsqueda | τ |
|---|---|---|---|---|
| `cs_nsga` (clusters Δ + NSGA-II) | 0.795 ± 0.026 | 0.735 ± 0.042 | 0.043 | 0.91 |
| `rand_nsga` (nodos aleatorios + NSGA-II) | **0.804 ± 0.023** | 0.753 ± 0.010 | 0.077 | 0.88 |
| `cs_rand` (clusters Δ + random search) | 0.794 ± 0.009 | 0.739 ± 0.011 | 0.045 | 0.89 |

Contrastes pareados por semilla (media de la diferencia; IC bootstrap sobre 5 semillas; >0 favorece al primero):
- `cs_nsga − rand_nsga`, HV test: **−0.0087 [−0.0123, −0.0045]**, gana 0/5 semillas.
- `cs_nsga − cs_rand`, HV test: +0.0011 [−0.0207, +0.0208], gana 3/5.
- HV con todos los nodos: −0.019 [−0.055, +0.017] (vs `rand_nsga`) y −0.004 [−0.037, +0.029] (vs `cs_rand`).

## Lectura
1. **No se detecta ventaja de la propuesta.** Con clusters de Δ el frente final es ligeramente *peor* que con nodos aleatorios
   (−0.009 de HV, ≈1 %, en las 5 semillas), aunque el estimador de fitness durante la búsqueda es más exacto (RMSE 0.043 vs
   0.077, como en E2). Un estimador más exacto no se tradujo en mejores frentes.
2. **No se detecta que la evolución aporte frente a muestreo aleatorio**: NSGA-II y random search con los mismos clusters
   dan el mismo HV (0.795 vs 0.794) y curvas de HV por generación casi idénticas.
3. El HV estimado por los representantes (`hv_rep`) es mayor para `rand_nsga` (0.837 vs 0.771) pero esa ventaja desaparece con la
   accuracy de todos los nodos y con test: es optimismo por ruido de selección (efecto del ganador) del estimador más ruidoso.
4. Regla de decisión de `PROTOCOL.md`: **no se cumple** (no hay mejora del HV en test frente a `random` + NSGA-II).

## Limitaciones
5 semillas; frente reevaluado con 1 semilla de inicialización y ≤5 arquitecturas; un dataset, α=0.1 y C=10; espacio CNN pequeño
(la plana del paisaje de fitness puede ocultar el efecto de la evolución); presupuesto de 150 evaluaciones. El resultado no
descarta ventajas con presupuestos o espacios mayores.

## Adenda (2026-10-21): curvas por generación y población inicial
- Corrección: los dos NSGA-II comparten la población inicial de cada semilla; `cs_rand` parte de otras 10 arquitecturas (0 coincidencias).
  Por eso su HV en la generación 0 difiere (0.55 vs 0.62–0.66) y la "mejora desde la generación 0" no es comparable entre métodos.
- HV con todos los nodos, gen 0 → 7: `cs_nsga` +0.118 (4/5 semillas mejoran), `rand_nsga` +0.097 (5/5), `cs_rand` +0.184 (5/5).
- `cs_nsga − cs_rand` por generación: de +0.018 [−0.022, +0.063] (gen 1) a −0.004 [−0.037, +0.029] (gen 7); área bajo la curva
  +0.008 [−0.027, +0.038]. `rand_nsga − cs_rand` (área): +0.030 [+0.013, +0.045], 4/5, confundido (cambian algoritmo y muestreo).
- Lectura: la búsqueda mejora con las generaciones en los tres métodos, pero no se puede atribuir a los operadores evolutivos.

## Adenda 2 (2026-10-21): control `rand_rand` (búsqueda aleatoria con nodos aleatorios)
5 corridas más, mismo coste (30 000 entrenamientos locales), mismas semillas; evalúa exactamente las mismas 150 arquitecturas por semilla que `cs_rand`.
HV test 0.795 ± 0.018; HV con todos los nodos 0.743; tras refinamiento 0.794 (Δ −0.001).
- `rand_nsga − rand_rand`: HV test +0.0085 [−0.001, +0.019] (4/5); HV todos +0.0104 [−0.0003, +0.027] (4/5); área bajo la curva **+0.0227 [+0.016, +0.029] (5/5)**.
- `cs_rand − rand_rand` (efecto del muestreo bajo búsqueda aleatoria): HV test −0.001 [−0.016, +0.017]; área −0.007 [−0.018, +0.005].
- `cs_nsga − rand_rand`: HV test −0.0002 [−0.012, +0.015]; área +0.002 [−0.033, +0.035].
- `rand_nsga − cs_rand`: HV todos +0.015 [+0.009, +0.023] (5/5); área +0.030 [+0.012, +0.045] (4/5).
Lectura: con muestreo aleatorio de nodos, NSGA-II supera a la búsqueda aleatoria por un margen pequeño y consistente en el área bajo la curva;
con clusters no se distingue. El muestreo por clusters no cambia el resultado de la búsqueda aleatoria. Explicación posible, no probada: a C=10 los
clusters dan modelos más ruidosos y sesgados (estudio de ruido), lo que degrada la selección de NSGA-II.
