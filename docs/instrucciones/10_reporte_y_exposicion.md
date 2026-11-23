# 10 — Reporte, walkthrough y presentación

**Dueños:** ambos, mitad y mitad (cada quien lo de sus partes) · **Carpeta:** `docs/`
**Leer antes:** `docs/Lab04.pdf` §2.1, §4, §5 y la rúbrica (§7)

## Entregables

| Archivo | Contenido |
|---|---|
| `docs/reporte.pdf` | reporte ejecutivo: formulación de Powell, decisiones con alternativas y evidencia, resultados agregados (media ± desviación entre folds), análisis de operaciones, sensibilidad y fallas de producción, respuestas a las 7 preguntas, advertencia de interpretación y limitaciones |
| `docs/walkthrough.pdf` | recorrido del código: estructura, flujo de `main.py`, decisiones clave de implementación y elecciones no evidentes, con la función que implementa cada cosa |
| `docs/presentacion.pdf` | máximo **14 diapositivas** sin contar portada ni cierre, 10 min de exposición + 5 de preguntas; ambos exponen |

Generación (como en el Lab 3): Markdown → HTML → PDF con Chrome sin interfaz (no hay pandoc ni LaTeX en la Mac).

## Checklist del reporte

- [ ] Formulación de Powell completa (parte 08).
- [ ] Tabla de decisiones de diseño: elección, alternativa rechazada y evidencia (datos o experimento).
- [ ] **Pregunta 1:** ≥ 4 regímenes del entrenamiento con volatilidad, tendencia y drawdown.
- [ ] **Pregunta 2:** por qué ese estado de información, con autocorrelación y estacionariedad de los folds de
      entrenamiento, y la alternativa rechazada.
- [ ] **Pregunta 3:** contribución por paso o terminal y el comportamiento indeseado de la otra.
- [ ] **Pregunta 4:** variación entre folds y qué explica el más débil.
- [ ] **Pregunta 5:** contra BTC buy-and-hold y S&P 500 después de costos; si no gana, qué mide el resultado.
- [ ] **Pregunta 6:** sensibilidad a la resolución intrabarra y a una barra de retraso.
- [ ] **Pregunta 7:** aporte de cada familia de variables (ablación) y segundo algoritmo contra el primero.
- [ ] §3.9 completo (3 peores operaciones, mayor drawdown con gráfica, correlación con BTC).
- [ ] §3.10 completo (ejecución, latencia, capacidad, cambio de régimen).
- [ ] **Advertencia de interpretación** explícita: el simulador supone el resto del sistema gratuito, instantáneo y
      perfecto; magnitud estimada con los experimentos de la parte 07.
- [ ] Procedimiento de entrenamiento, búsqueda de hiperparámetros y criterio de selección con referencia a
      `validation_results_fold_{k}.csv`.
- [ ] Toda cifra sale de `artifacts/` (verificar contra los CSV antes de entregar).

## Checklist de la presentación

- [ ] ≤ 14 diapositivas + portada + cierre; reparto de tiempo entre ambos y ensayo cronometrado (10 min).
- [ ] Cada gráfica sustenta un punto concreto; títulos, ejes y leyendas legibles desde el fondo del salón; sin capturas de código.
- [ ] Cierre con conclusiones definidas: ¿ventaja o artefacto de los supuestos?
- [ ] Ambos pueden explicar cualquier parte y señalar dónde ocurre en el código (preguntas probables de cada guía).
