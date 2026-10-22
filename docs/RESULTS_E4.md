# E4 — refinamiento multifidelidad del frente de salida: resultado (2026-10-21)

Diseño en `docs/E4_DESIGN.md`; `run_refine.py` + `analyze_e4.py`; crudos en `runs/e3/<método>/seed*/refine/` (no versionado).
Sobre las 15 corridas de E3: ≤12 candidatos por corrida reevaluados con C=40 nodos aleatorios nuevos; frente final elegido con esa
estimación y reentrenado con participación completa para medir en test. Coste añadido ≈ 9 600 entrenamientos locales por corrida.

HV del frente final en test (media de 5 semillas), antes → después del refinamiento:

| método | HV original (E3) | HV refinado | Δ pareado [IC bootstrap 95 %] | semillas que mejoran |
|---|---|---|---|---|
| `cs_nsga` | 0.795 | 0.794 | −0.001 [−0.010, +0.004] | 4/5 |
| `rand_nsga` | 0.804 | 0.812 | +0.009 [+0.001, +0.017] | 4/5 |
| `cs_rand` | 0.794 | 0.799 | +0.005 [+0.001, +0.009] | 5/5 |

Contrastes entre métodos (HV en test, pareado; >0 favorece al primero):
- `cs_nsga − rand_nsga`: −0.0087 [−0.012, −0.005] (original) → −0.0187 [−0.031, −0.006] (refinado), 1/5 semillas a favor.
- `cs_nsga − cs_rand`: +0.001 [−0.021, +0.021] → −0.005 [−0.031, +0.021].
- `rand_nsga − cs_rand`: +0.010 [−0.011, +0.028] → +0.014 [−0.004, +0.030].

Otras medidas: la mejor accuracy de test del frente apenas cambia (−0.005 a +0.003); la accuracy media de las 5 arquitecturas
seleccionadas **baja** (−0.02 a −0.05) porque el conjunto elegido cambia (solo 1–3 de 5 arquitecturas coinciden con el original) y tiende a
más parámetros más pequeños en los métodos con clusters (params_norm medio 0.098 → 0.067 en `cs_rand`, 0.071 → 0.054 en `cs_nsga`); la
estimación refinada (C=40) correlaciona 0.99 con la accuracy del mismo modelo en todos los nodos, y la estimación de búsqueda (C=10) 0.92.

## Lectura
1. **El refinamiento aporta, como mucho, ganancias pequeñas** (+0.005 a +0.009 de HV, ≈1 %) en dos de tres métodos y nada en `cs_nsga`.
2. **No cambia el orden entre métodos**: `rand_nsga` ≥ `cs_rand` ≈ `cs_nsga`; el déficit de `cs_nsga` frente a `rand_nsga` se amplía
   (−0.019) y la evolución sigue sin distinguirse de random search (IC amplios que incluyen 0).
3. La estimación a más fidelidad es mucho más fiable (r = 0.99 con la accuracy global), pero eso no se traduce en mejores frentes finales
   medidos con entrenamiento completo: el cuello de botella parece estar en el conjunto de candidatos que genera la búsqueda y/o en la
   diferencia entre entrenar con C nodos y con todos (el ranking frente a participación completa se mantuvo en τ ≈ 0.85 en el estudio de ruido).
4. Limitaciones: 5 semillas; una sola reevaluación por candidato y una semilla de inicialización en el test final; el refinamiento es
   posterior a la búsqueda (no guía la evolución); mismo dataset/α/C=10 en la búsqueda.
