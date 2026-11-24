# 06 — Evaluación: métricas, benchmarks, bitácora y análisis de operaciones

**Dueño:** Juan Manuel Espinosa (JME) · **Revisor:** Gonzalo Cano (GC) · **Notebook:** `notebooks/06_evaluacion.ipynb` · **Módulo:** `code/evaluate.py` (+ `code/plots.py`) · **Depende de:** 05 · **Alimenta a:** 07, 10
**Leer antes:** `docs/Lab04.pdf` §3.6, §3.7, §3.9 y preguntas 4 y 5

## Objetivo

Correr el modelo congelado de cada fold **una sola vez** sobre su prueba, llevar la bitácora de operaciones y la curva
de capital, compararlas contra BTC buy-and-hold y el S&P 500 con los mismos costos, y agregar métricas entre folds.

## Parámetros de este módulo

Capital inicial, anualización y tasa libre de riesgo al inicio de `code/evaluate.py`.

## Contrato

- **Lee:** modelos de `artifacts/models/`, `TradingEnv` en modo evaluación, `data/sp500_daily.csv`.
- **Escribe:** `artifacts/trade_log_fold_{k}.csv`, `artifacts/equity_curves_test_fold_{k}.png`,
  `artifacts/metrics_summary.csv`, `artifacts/trade_analysis.json`, `artifacts/max_drawdown_fold_{k}.png`,
  `artifacts/agent_simulation.gif`,
  `data/sp500_daily.csv` (snapshot de `^GSPC`, solo si falta).
- **Expone (propuesta):** `run_policy(model, env) -> (equity, trade_log)`, `benchmark_btc(ohlc, period)`,
  `benchmark_sp500(prices, period)`, `compute_metrics(equity, trades) -> dict`, `aggregate_folds(df)`,
  `worst_trades(logs, n=3)`, `max_drawdown_window(equity)`.

## Loop de trabajo

```
REPETIR hasta cumplir el criterio de terminado:
  1. Toma el primer ítem sin [x].
  2. Implementa; verifica cada métrica contra un cálculo a mano sobre una serie de juguete.
  3. Corre la identidad contable.
  4. Marca [x], registra la decisión (anualización, alineación de calendarios), commit.
```

## Checklist

- [ ] Evaluación en prueba con el modelo elegido en validación, política determinista, **una sola corrida por fold** (se
      registra en `run_metadata.json`).
- [ ] `trade_log_fold_{k}.csv`: entrada y salida (marca de tiempo y precio), razón de salida (horizonte, stop,
      take-profit o cierre), tamaño, P&L bruto, costos, P&L neto, duración y estado en la entrada.
- [ ] `test_accounting.py` (prueba obligatoria 4): capital final = capital inicial + Σ P&L neto de la bitácora.
- [ ] Benchmarks con costo de 0.125 % por cada operación que generen: BTC buy-and-hold (compra al inicio, venta al final
      del periodo) y S&P 500 indexado (`^GSPC`, snapshot versionado, solo benchmark).
- [ ] Alineación de calendarios declarada (BTC 24/7 contra S&P 500 en días hábiles) y anualización coherente.
- [ ] Métricas de §3.7 para estrategia y benchmarks, por fold y agregadas (media y desviación entre los 4 folds):
      retorno total, retorno anualizado, volatilidad anualizada, Sharpe, Sortino, MDD, Calmar, hit rate, duración
      promedio de las operaciones, rotación y retorno en exceso contra cada benchmark (alfa) → `metrics_summary.csv`.
- [ ] `equity_curves_test_fold_{k}.png`: estrategia contra ambos benchmarks.
- [ ] **§3.9.1:** las 3 operaciones con mayor pérdida de prueba (todos los folds): entrada, salida, razón, variables de
      estado en la entrada y la interpretación (qué cambiaría en estado, recompensa o gestión de riesgo).
- [ ] **§3.9.2:** periodo del mayor drawdown en prueba: gráfica de acciones y precio, y explicación.
- [ ] **§3.9.3:** correlación entre rendimientos del agente y de BTC en prueba; si es alta, discutir si el agente es un
      largo apalancado o filtrado.
- [ ] **Simulación visual** (petición del profesor en clase, 2026-11-23; dueño **GC**, revisor JME): animación del agente
      operando un tramo de prueba, como el pingüino de FrozenLake de clase: precio barra por barra, decisión del agente,
      entrada y salida con su razón (horizonte, stop, take-profit, cierre), niveles de stop y take-profit, posición y
      capital acumulado contra BTC buy-and-hold → `artifacts/agent_simulation.gif` (`code/plots.py`, con
      `matplotlib.animation` y Pillow). Lee solo la bitácora y la curva de capital; el tramo se elige con un criterio
      fijo (p. ej. el periodo del mayor drawdown de §3.9.2 o las 2 semanas con más operaciones). Se usa en la presentación.
- [ ] **Preguntas 4 y 5:** variación entre folds y qué explica el fold más débil; si supera a ambos benchmarks después de
      costos y, si no, qué mide entonces el resultado.

## Criterio de terminado

Los artefactos de §3.8 de esta parte existen para los 4 folds, la identidad contable pasa y las cifras de §3.9 y de
las preguntas 4–5 salen de los artefactos.

## Trampas

- Volver a evaluar prueba después de cambiar algo (la prueba se toca una vez; cualquier cambio posterior se declara).
- Sharpe anualizado con √252 sobre rendimientos horarios de un activo 24/7.
- Comparar contra benchmarks sin costos o en periodos distintos.

## Preguntas probables

- ¿El agente le gana a comprar y mantener después de costos? ¿Qué explica el fold más débil?
- ¿Cuál fue la peor operación y por qué la tomó el agente? ¿Su agente es un largo disfrazado?
