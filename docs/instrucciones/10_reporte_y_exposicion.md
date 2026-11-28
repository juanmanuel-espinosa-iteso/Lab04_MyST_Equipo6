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

- [x] Formulación de Powell completa (parte 08).
- [ ] Tabla de decisiones de diseño: elección, alternativa rechazada y evidencia (datos o experimento).
- [x] **Pregunta 1:** ≥ 4 regímenes del entrenamiento con volatilidad, tendencia y drawdown.
- [x] **Pregunta 2:** por qué ese estado de información, con autocorrelación y estacionariedad de los folds de
      entrenamiento, y la alternativa rechazada.
- [x] **Pregunta 3:** contribución por paso o terminal y el comportamiento indeseado de la otra.
- [x] **Pregunta 4:** variación entre folds y qué explica el más débil.
- [x] **Pregunta 5:** contra BTC buy-and-hold y S&P 500 después de costos; si no gana, qué mide el resultado.
- [x] **Pregunta 6:** sensibilidad a la resolución intrabarra y a una barra de retraso.
- [x] **Pregunta 7:** aporte de cada familia de variables (ablación) y segundo algoritmo contra el primero.
- [x] §3.9 completo (3 peores operaciones, mayor drawdown con gráfica, correlación con BTC).
- [x] §3.10 completo (ejecución, latencia, capacidad, cambio de régimen).
- [x] **Advertencia de interpretación** explícita: el simulador supone el resto del sistema gratuito, instantáneo y
      perfecto; magnitud estimada con los experimentos de la parte 07.
- [x] Procedimiento de entrenamiento, búsqueda de hiperparámetros y criterio de selección con referencia a
      `validation_results_fold_{k}.csv`.
- [ ] Toda cifra sale de `artifacts/` (verificar contra los CSV antes de entregar).

**Estado (2026-11-27, JME):** `docs/reporte.md` → `docs/reporte.pdf` con `python docs/generar_pdf.py reporte`
(rama `feat/jme-reporte`). Hechas las secciones de JME: resumen, 1 (datos), 3 (Powell resumido; Apéndice A =
`powell.md`), 6 (P4 y P5), 7 (§3.9), 8 (P6), 10 (§3.10) y 11 (advertencia y limitaciones). Secciones de GC hechas
(2026-11-27, rama `feat/gc-reporte`): 2 (P1 y P2), 4 (P3), 5 (entrenamiento) y 9 (P7, con la simulación visual); la
sección 9 lee `ablation.csv` y `algorithm_comparison.csv`, que llegan con el #10.

## Checklist de la presentación

- [ ] ≤ 14 diapositivas + portada + cierre; reparto de tiempo entre ambos y ensayo cronometrado (10 min).
- [ ] Cada gráfica sustenta un punto concreto; títulos, ejes y leyendas legibles desde el fondo del salón; sin capturas de código.
- [ ] La simulación visual (`artifacts/agent_simulation.gif`, petición del profesor) se muestra en la presentación para
      explicar cómo decide y cierra posiciones el agente.
- [ ] Cierre con conclusiones definidas: ¿ventaja o artefacto de los supuestos?
- [ ] Ambos pueden explicar cualquier parte y señalar dónde ocurre en el código (preguntas probables de cada guía).
