# AGENTS.md

## Proyecto

Proyecto experimental en Python orientado a aprendizaje federado, búsqueda evolutiva de arquitecturas neuronales y optimización multiobjetivo.

Stack principal:
- Python
- PyTorch
- pymoo
- NumPy / Pandas
- Matplotlib
- pytest

## Directrices de implementación

- Prioriza código simple, legible, modular y fácil de experimentar.
- Evita sobreingeniería y abstracciones prematuras.
- Antes de modificar código, revisa la implementación existente y reutiliza lo que ya funcione.
- Mantén separadas las responsabilidades principales: datos, clientes federados, modelos, evolución, métricas, experimentos y visualización.
- Usa configuración para parámetros experimentales; evita valores importantes hardcodeados.
- Mantén reproducibilidad mediante seeds y registro de la configuración utilizada.
- No mezcles datos de entrenamiento, validación y prueba.
- En la simulación federada, conserva la separación conceptual entre servidor y clientes. Los datos crudos de los clientes no deben centralizarse.
- Mantén independientes las evaluaciones de modelos entre clientes; no reutilices accidentalmente pesos entrenados en otro cliente.
- Guarda resultados experimentales y métricas de forma estructurada para permitir análisis posterior.
- Añade tests para lógica crítica y para bugs corregidos cuando sea razonable.
- Antes de declarar una tarea terminada, ejecuta los tests relevantes y, cuando aplique, un smoke test pequeño end-to-end.
- No hagas refactors grandes ni modifiques archivos no relacionados sin una razón concreta.
- Mantén las dependencias al mínimo.
- No optimices prematuramente; identifica primero los cuellos de botella reales.
- No alteres el diseño experimental para favorecer artificialmente un método o resultado.
- Si una decisión técnica puede cambiar la semántica del experimento, documéntala antes de implementarla.

## Ponytail

Usa la skill **Ponytail** durante la planificación e implementación para favorecer soluciones simples y evitar sobreingeniería.

Aplica especialmente sus principios al:
- diseñar nuevas abstracciones;
- decidir si crear nuevas clases o módulos;
- evaluar refactors;
- añadir dependencias;
- proponer infraestructura adicional.

Si existen varias soluciones correctas, prioriza la alternativa más simple que satisfaga los requisitos actuales y preserve la claridad y reproducibilidad del experimento.

## Git

No incluir en commits:
- datasets descargados;
- entornos virtuales;
- checkpoints o pesos;
- resultados generados;
- archivos temporales;
- secretos o `.env`.

Mantén los cambios pequeños, coherentes y relacionados con la tarea actual.
