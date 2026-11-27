# 03 — Entorno Gymnasium semi-Markov y gestor de posiciones

**Dueño:** Juan Manuel Espinosa (JME) · **Revisor:** Gonzalo Cano (GC) · **Notebook:** `notebooks/03_entorno.ipynb` · **Módulo:** `code/env.py` · **Depende de:** 01 · **Alimenta a:** 04, 05, 06, 07
**Leer antes:** `docs/notas_clase/02_rl_q_learning_gymnasium.md` (§5–6) · `docs/notas_clase/03_sb3_dqn_ppo.md` (§2, §3 y §7, plantilla del entorno) · `docs/notas_clase/01_sequential_decision_analytics.md` · `docs/Lab04.pdf` §3.2 y §3.3

## Objetivo

Un `gym.Env` que modela la decisión de abrir posiciones sobre BTCUSDT con un gestor que las cierra en el primero
de: horizonte, stop-loss, take-profit o cierre explícito (si el diseño lo permite). Cada `step` va de una decisión a la
siguiente (semi-Markov): **estado en la apertura, acción, recompensa al cierre, estado al cierre, duración real τ**.

## Parámetros de este módulo

`COST_PER_SIDE = 0.00125`, `INTRABAR = "stop_first"`, acciones, horizonte H, SL, TP, capital inicial, tamaño de
posición y longitud del episodio al inicio de `code/env.py`.

## Contrato

- **Lee:** `FoldData` de la parte 01 (OHLC crudo, features normalizadas y `raw_features` para la volatilidad de las
  barreras) y la función de contribución de `code/reward.py`.
- **Expone:** `TradingEnv(ohlcv, features, barrier_vol, reward_fn, mode="train"|"eval", horizon_hours, k_stop,
  k_take, intrabar, allow_short, episode_bars, obs_lag, cost)`, `make_env(fold_data, split, reward_fn, **params)`,
  `barrier_levels(...)`, `resolve_exit(...) -> Exit(bar, price, reason, ambiguous, at_open)`, `TradeOutcome`,
  constantes `WAIT, LONG, SHORT`. En modo eval, `trade_log()` y `equity_series()` devuelven el último episodio
  terminado (sobreviven al reset automático de un VecEnv; parte 06). `info` del `step` lleva `duration` (τ en horas),
  `decision_time`, `equity`, `outcome` y, si hubo operación, la fila de la bitácora (entrada, salida, razón, precios,
  niveles, cantidad, P&L bruto, costos, neto, `exit_at_open`, `intrabar_ambiguous`).
- **Contrato con la parte 04 (GC):** la contribución es una función `TradeOutcome -> float`. `TradeOutcome` trae
  `side, duration_hours, gross_return, cost_return, net_return, net_log_return, entry_vol, exit_reason` (rendimientos
  sobre el capital antes de decidir; en una espera todo vale 0 salvo `duration_hours` y `entry_vol`).

## Loop de trabajo

```
REPETIR hasta cumplir el criterio de terminado:
  1. Toma el primer ítem sin [x].
  2. Implementa en code/env.py.
  3. Verifica con una trayectoria de juguete calculada a mano (5–10 barras) y check_env.
  4. Marca [x], registra la decisión (elección, alternativa, evidencia), commit.
```

## Checklist

- [x] **Espacio de acciones** y tratamiento de acciones inválidas (p. ej. {esperar, largo, corto}; decidir si hay corto
      y si hay cierre explícito, con evidencia). Registrar la alternativa descartada.
- [x] **Estado** = física (posición, precio de entrada, tiempo en posición, efectivo) + información (features de 01)
      + creencia (estimado con incertidumbre). Observación `Box` float32 con la posición incluida (sin ella no es Markov).
- [x] Temporalidad: decisión con la barra t cerrada; llenado en la **apertura de t + 1**; el costo se cobra por lado
      (entrada y salida) sobre el nocional.
- [x] **Gestor de posiciones:** cierre en el primero de horizonte, stop-loss, take-profit o cierre explícito, recorriendo
      la trayectoria OHLC barra por barra.
- [x] **Resolución intrabarra:** si una barra toca stop y take-profit, `stop_first` por defecto; `tp_first` disponible
      para la parte 07. Gaps: si la apertura ya cruzó el nivel, se llena a la apertura (no al nivel).
- [x] **Semi-Markov:** cada `step` devuelve el estado al cierre y la duración real τ en `info["duration"]`; "esperar" avanza
      una barra (τ = 1) o lo que se decida.
- [x] **Episodio:** definición explícita (p. ej. ventanas de longitud fija con inicio aleatorio en entrenamiento; recorrido
      completo en orden en evaluación) y qué se pierde con ella; `terminated` solo por ruina, fin de datos = `truncated`.
- [x] Parámetros del periodo de tenencia (H, SL, TP) justificados con datos (p. ej. distribución de rangos/volatilidad
      a H horas en el train).
- [x] `test_env_check.py` (prueba obligatoria 1): `check_env` sin advertencias.
- [x] `test_reference_trade.py` (prueba obligatoria 3): trayectoria corta calculada a mano con razón de salida, duración
      y P&L conocidos, reproducida exactamente (un caso por razón de salida, incluido el empate intrabarra).
- [x] Validación de la clase: siempre largo en el entorno reproduce el rendimiento del activo menos costos (tolerancia de punto flotante).

**Estado (2026-11-24, JME):** terminado en la rama `feat/jme-entorno` (sale de `feat/jme-datos`). El corto y la ausencia de cierre explícito quedaron decididos con evidencia en `decisiones.md`.

## Criterio de terminado

`check_env` limpio, la operación de referencia y el siempre largo cuadran a mano, y la parte 05 puede instanciar el
entorno con cualquier fold.

## Trampas

- Llenar al cierre de la barra con la que se decidió (look-ahead).
- Revisar stop y take-profit con el cierre en lugar de con máximo y mínimo; ignorar gaps de apertura.
- Marcar el fin de datos como `terminated` (S18: el agente aprende a liquidar antes del final).
- Meter información de `info` en la observación.

## Preguntas probables

- ¿A qué precio llena su simulador? ¿Qué pasa si stop y take-profit caen en la misma barra?
- ¿Por qué esto es semi-Markov y dónde se guarda la duración? ¿Qué pierden con su definición de episodio?
