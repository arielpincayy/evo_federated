"""Problema pymoo para NAS federado.
Soporta diferentes estrategias de selección de clientes.

NOTA: Debido a que la evaluación requiere entrenamiento real en clientes,
implementamos un problema que delega evaluación a un Evaluator que
mantiene estado (pairing, generación, etc.). Para pymoo, usamos
pymoo.core.problem.Problem con n_var = dim del genoma continuo [0,1].
Repair via Genome.from_vector.

Para FULL: evaluación en todos los clientes -> objetivos pueden agregarse,
pero para comparación justa, FULL también debe optimizar 2 objetivos?
Según spec, FULL evalúa en todos y debería producir frente pero con coste mayor.
Diseño: FULL => objetivos = (mean F1, min F1) or still 2 objetivos sobre aggregated?
Para mantener comparabilidad con 2-cliente methods, usamos para FULL:
  f1 = mean F1 over all clients, f2 = min F1 (fairness) => ambos maximizados.
Alternativamente, usar 2 objetivos como en proposed pero con exhaustiva evaluación
sería: seleccionar par más divergente internamente? No.

Spec dice FULL cada arquitectura se evalúa en todos los clientes durante evolución.
Los objetivos siguen siendo 2: pero ¿cuáles? Spec sec 8 dice objetivos son F1_i, F1_j.
Para FULL, no hay i,j únicos. Propuesta razonable: FULL optimiza (mean F1, -std) o (mean, min).
Pero para HV comparability, guardamos HV_interno per strategy y HV_validación común.

Simplificamos: FULL => objetivos = (mean_F1, min_F1) transformados a -mean, -min.
RANDOM/FIXED/DYNAMIC => objetivos = (F1_i, F1_j).

Esto mantiene 2 objetivos para todos, aunque semántica difiere. Documentamos.
Para HV comparison, usaremos HV de validación común (mean + min o mean + median etc.)
evaluando finalistas exhaustivamente en todos los clientes y recalculando HV sobre
(mediana, min) o (mean, min) consistent.

Decisión documentada.

Otra opción: FULL podría optimizar (F1_global_promedio, F1_global?) no. Usamos mean/min.

Implementamos evaluador que para cada individuo devuelve 2 valores minimización.
"""
import numpy as np
from pymoo.core.problem import Problem


class FederatedNASProblem(Problem):
    def __init__(self, n_var: int, evaluator, **kwargs):
        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=0.0, xu=1.0, **kwargs)
        self.evaluator = evaluator

    def _evaluate(self, X, out, *args, **kwargs):
        # X shape (n_individuals, n_var)
        F = self.evaluator.evaluate_batch(X)
        out["F"] = F
