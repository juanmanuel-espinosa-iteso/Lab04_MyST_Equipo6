# Project Charter — Lab04 Equipo 6

> Máximo una página. Entrega: último commit antes del domingo 29 de noviembre de 2026, 23:59. Exposición: lunes 30.

## Alcance

**Hará:** formular el trading de BTCUSDT horario como problema de decisión secuencial de Powell (estado físico,
de información y de creencia; decisión; información exógena; transición semi-Markov con cierre por horizonte, stop,
take-profit o cierre explícito; contribución con costos; objetivo); features causales con normalizador por fold;
EDA y análisis de features solo en entrenamiento; entorno Gymnasium validado con `check_env` y líneas base; agente
de Stable-Baselines3 con replay buffer semi-Markov, varias semillas y búsqueda de hiperparámetros en validación;
walk-forward de 4 folds con prueba tocada una vez; métricas contra BTC buy-and-hold y S&P 500 con costos; análisis de
operaciones, sensibilidad (intrabarra, retraso), ablación, segundo algoritmo y fallas de producción; 4+ pruebas con
pytest; reporte, walkthrough y presentación en PDF; simulación visual animada del agente operando (petición del
profesor); un notebook de demostración por parte.

**No hará:** operación en vivo ni conexión a exchanges; datos externos de precio como features (el S&P 500 solo
como benchmark); modelado del libro de órdenes, impacto de mercado o financiamiento de cortos más allá de los
experimentos de sensibilidad; dashboard; implementación propia de los algoritmos de RL (se usa SB3).

## Plan de hitos

| Hito | Fecha objetivo | Responsable | Estado |
|---|---|---|---|
| Repo, estructura, guías, notas de clase y charter | 2026-11-23 | JME | hecho |
| Datos, features causales, folds y normalizador (01) | 2026-11-24 | JME | hecho |
| EDA, regímenes y análisis de features (02) | 2026-11-24 | GC | hecho (2026-11-27) |
| Entorno con `check_env` y operación de referencia (03) | 2026-11-24 | JME | hecho |
| Contribución (04) | 2026-11-24 | GC | hecho (2026-11-27) |
| Líneas base y un fold de punta a punta (05) | 2026-11-25 | GC | pendiente |
| 4 folds × semillas, búsqueda de HP y evaluación en prueba (05, 06) | 2026-11-26 | GC / JME | pendiente |
| Sensibilidad, ablación, segundo algoritmo; `main.py` completo (07) | 2026-11-27 | JME / GC | pendiente |
| Reporte, walkthrough y presentación (08, 10) | 2026-11-28 | ambos | pendiente |
| Revisión desde clon limpio (99) y último commit | 2026-11-29 | ambos | pendiente |

## Registro de riesgos

| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|---|
| 1 | El tiempo de entrenamiento (4 folds × semillas × rejilla × ablación × 2 algoritmos) no cabe en la semana ni en `main.py` | Alta | Alto | Rejilla chica, pasos acotados, procesos en paralelo por fold; medir el tiempo de un fold el día 25 y ajustar |
| 2 | Look-ahead en features, normalización o llenado de órdenes | Media | Alto | Prueba de causalidad, llenado en la apertura de t + 1, normalizador ajustado solo con train |
| 3 | Error del simulador (el aleatorio gana dinero, la contabilidad no cuadra) | Media | Alto | Líneas base y prueba de identidad contable antes de entrenar; operación de referencia a mano |
| 4 | El agente no aprende nada mejor que plano o siempre largo | Alta | Medio | Reportarlo con honestidad y explicar qué mide el resultado (pregunta 5); la rúbrica premia el análisis, no la ganancia |
| 5 | Sobreajuste a validación por la búsqueda de hiperparámetros | Media | Medio | Rejilla y criterio fijados antes de ver resultados; prueba tocada una sola vez |
| 6 | Commits desbalanceados o un integrante sin commits | Media | Alto | Ramas por integrante, PR cruzados, `git shortlog` diario |
| 7 | Incompatibilidades de versiones (SB3, Gymnasium, torch) en otra máquina | Baja | Alto | Versiones fijadas, revisión desde clon limpio el día 29 |

## Roles

- **Juan Manuel Espinosa (JME):** datos y features (01), entorno y gestor de posiciones (03), evaluación y benchmarks
  (06), sensibilidad de ejecución (07), Powell del estado y la transición (08), pruebas obligatorias 1–4, charter y sprint.
- **Gonzalo Cano (GC):** EDA y análisis de features (02), contribución (04), entrenamiento y búsqueda de
  hiperparámetros (05), ablación y segundo algoritmo (07), Powell de la contribución y el objetivo (08), prueba de la recompensa.
- **Ambos:** revisión cruzada por PR, reporte, walkthrough, presentación y post-mortem. Meta: commits parejos
  (`git shortlog -sn --no-merges`).
