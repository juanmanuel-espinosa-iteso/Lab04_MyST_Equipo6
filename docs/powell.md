# Formulación del Lab04 como problema de decisión secuencial (Powell)

Notación de clase (`docs/notas_clase/01_sequential_decision_analytics.md`): estado S = (R, I, B), decisión x,
información exógena W, transición S^M, contribución C, objetivo y política X^π. Junto a cada elemento va la función que
lo implementa. Las decisiones de diseño, con su alternativa descartada y la evidencia, están en `docs/decisiones.md`.

**Autoría:** secciones 1–5 y 7 JME (estado, decisión, información exógena, transición, ejemplo); sección 6 GC
(contribución, objetivo y política).

## 1. Narrativa y elementos centrales (pasos 1–2)

Un agente opera BTCUSDT con barras horarias de Binance. Al cierre de una barra, si no tiene posición, decide esperar una
hora o abrir una posición larga o corta con todo su capital, sin apalancamiento. La orden se llena a la apertura de la
barra siguiente y un **gestor de posiciones** la cierra en el primero de estos eventos: se toca el stop-loss, se toca el
take-profit, se cumple el horizonte H o se acaban los datos. Cada lado paga 0.125 % sobre su nocional.

- **Métrica:** crecimiento del capital después de costos, comparado contra comprar y mantener BTC y contra el S&P 500.
- **Decisiones:** cuándo abrir y en qué dirección. Horizonte y barreras son parámetros del gestor.
- **Incertidumbre:** la trayectoria del precio dentro de la posición, con sus huecos de datos.

**Es semi-Markov.** El agente no decide mientras está en posición, así que el tiempo entre dos decisiones es endógeno:
1 h al esperar, y de 1 a H horas (más si hay huecos) al operar. Se indexan las **épocas de decisión** n = 0, 1, 2, …, que
ocurren en los instantes t_n (cierre de una barra en la que el agente está plano). La duración de la transición n es
τ_n = t_{n+1} − t_n en horas de reloj.

## 2. Estado S_{t_n} = (R_{t_n}, I_{t_n}, B_{t_n}) (paso 3, elemento 1)

### R: estado físico (recursos)

| Componente | Símbolo | En una época de decisión | Dentro de la transición | En el código |
|---|---|---|---|---|
| Posición | q_t ∈ {−1, 0, +1} | 0 (siempre plano) | s = ±1 mientras está abierta | `side` en la bitácora |
| Precio de entrada | P^ent_t | no definido | apertura de la barra de entrada | `entry_price` |
| Tiempo en posición | h_t | 0 | horas desde la entrada | `duration_hours` |
| Efectivo / capital | E_t | capital después de la última salida | E_t marcado a mercado al cierre de cada barra | `TradingEnv.equity`, `equity_series()` |

Como el agente solo decide estando plano, en las épocas de decisión posición, precio de entrada y tiempo en posición son
degenerados: evolucionan **dentro** de la transición (sección 5). Lo que sí cambia entre épocas es la cuenta, y entra a
la observación con 3 variables (`env.ACCOUNT_FEATURES`, `TradingEnv._obs`):

- caída del capital desde su máximo, E_t / máx_{s ≤ t} E_s − 1;
- rendimiento neto de la última operación (× 10, recortado a ±5);
- horas plano desde la última salida (/168, máximo 1).

### I: estado de información (lo observado del mercado)

18 variables causales calculadas con barras hasta t inclusive (`data.compute_features`) y normalizadas con el train de
cada fold (`data.Normalizer`):

| Familia | Variables |
|---|---|
| Rendimientos | `ret_1h`, `ret_4h`, `ret_24h`, `ret_168h`: ln(P_t / P_{t−k}) en horas de reloj |
| Volatilidad | `rv_24h`, `rv_168h` (raíz de la media de r²/Δh en 24 y 168 barras), `parkinson_24h`, `vol_ratio_24_168` |
| Tendencia | `ema_gap_24h`, `ema_gap_168h`, `ema_gap_720h` (P_t / EMA − 1), `drawdown_720h` |
| Volumen | `volume_z_168h`, `volume_ratio_24_168` |
| Calendario | hora del día y día de la semana en seno y coseno |

La evidencia para conservarlas (autocorrelación, estacionariedad e información mutua en el train) está en la parte 02 y
en `docs/decisiones.md`.

### B: estado de creencia (lo estimado, con su incertidumbre)

El agente no conoce la deriva ni la volatilidad del precio; las estima recursivamente (`data._belief_features`):

- **Deriva:** μ̂_t = EWMA de los rendimientos horarios, con vida media de 168 h.
- **Incertidumbre de la deriva:** se(μ̂_t) = σ̂_t / √n_eff, con n_eff = (1 + λ)/(1 − λ) y λ = 0.5^{1/168}. Es el tamaño
  efectivo de muestra de los pesos geométricos (aproximación normal, rendimientos i.i.d.).
- **Señal relativa a su ruido:** μ̂_t / se(μ̂_t).
- **Volatilidad:** σ̂²_t = Σ_i w_i · r_i²/Δh_i / Σ_i w_i sobre las barras observadas i ≤ t, con
  w_i = 0.5^{(t − t_i)/24 h}. Es una EWMA en el tiempo con vida media de 24 h (`pandas.ewm(times=…)`). Dividir el
  rendimiento entre las horas Δh_i que abarca lo lleva a varianza por hora, así que los huecos no la sesgan. σ̂_t fija
  también las barreras del gestor.

Son estadísticos suficientes que se actualizan con cada barra nueva: no se carga el historial crudo. Una EWMA es el
filtro de Kalman de nivel local en estado estacionario.

**Observación del agente.** Un vector de 25 componentes, `Box(−5, 5)`: las 22 features de I y B normalizadas en la
barra t_n (t_n − 1 si hay retraso, parte 07) y las 3 de la cuenta.

## 3. Decisión x_{t_n} (elemento 2)

- 𝒳 = {0: esperar, 1: abrir largo, 2: abrir corto} (`Discrete(3)`; `env.WAIT, LONG, SHORT`). La dirección es
  s = 0, +1, −1.
- **Acciones inválidas:** no hay, por construcción. El agente solo decide estando plano, así que nunca se le pide abrir
  una posición abierta ni cerrar estando plano.
- **Sin cierre explícito:** el evento 4 de §3.3 no existe en este diseño. Decidir en cada barra con {mantener, cerrar}
  rompería la formulación semi-Markov y diluiría la asignación de crédito (`docs/decisiones.md`).
- **Parámetros del gestor** (fijos durante la operación; candidatos a ajustarse en validación): horizonte H = 24 h y
  barreras a k_stop = k_take = 1 vez σ̂·√H.

## 4. Información exógena W_{t_n+1} (elemento 3)

Es la trayectoria de barras que llega **después** de la decisión, hasta la siguiente época de decisión:

W_{t_n+1} = {(marca_j, O_j, H_j, L_j, C_j, V_j) : j = e_n, …, x_n},

donde e_n es la barra siguiente a t_n (la de entrada) y x_n la barra de salida. Incluye los **huecos de datos**: una
barra puede llegar varias horas después de la anterior (1,114 horas faltantes, sobre todo en 2025–2026). En la
evaluación W se reproduce de la serie histórica (backtest); al entrenar, el inicio del episodio se muestrea al azar con
`self.np_random`. Ningún elemento de W entra a la decisión x_{t_n}: el subíndice t_n + 1 es la regla de causalidad
(`tests/test_causality.py`).

## 5. Función de transición S_{t_{n+1}} = S^M(S_{t_n}, x_{t_n}, W_{t_n+1}) (elemento 4)

Implementación: `TradingEnv.step` → `_wait` o `_trade` → `barrier_levels` y `resolve_exit`.

**Esperar (s = 0).** t_{n+1} es la barra observada siguiente; E no cambia; τ_n = horas entre ambas barras (1, o más si
falta la siguiente).

**Operar (s = ±1).**

1. **Entrada** a la apertura de la barra e = t_n + 1: P^ent = O_e. Cantidad Q = E_{t_n} / P^ent (todo el capital, 1×).
2. **Barreras** con la volatilidad estimada al decidir (`barrier_levels`):
   - stop L = P^ent · (1 − s · k_stop · σ̂_{t_n} · √H);
   - take-profit U = P^ent · (1 + s · k_take · σ̂_{t_n} · √H).
3. **Gestor de posiciones** (`resolve_exit`). Recorre las barras j = e, e+1, … que empiezan antes de marca_e + H:
   - si j > e y la apertura ya cruzó un nivel (gap, incluidas horas faltantes), sale a la apertura O_j como stop o
     take-profit;
   - si el mínimo o máximo de la barra toca un solo nivel, sale a ese nivel;
   - si toca **ambos** en la misma barra, solo se observa OHLC y no se sabe cuál fue primero: el supuesto por defecto
     (conservador) es que **el stop se ejecuta primero**. `tp_first` es el supuesto contrario para la parte 07, y la
     operación queda marcada como ambigua;
   - si nada se toca: sale al cierre de la última barra cuando esta termina justo en marca_e + H (horizonte). Si el
     horizonte vence dentro de un hueco, sale a la apertura de la primera barra posterior, revisando ahí las barreras;
     al cierre previo al hueco todavía no se sabía que faltarían barras. Si los datos se acaban, sale al último cierre
     (`fin_datos`).
4. **Contabilidad**, con c = 0.00125 por lado:
   - P&L bruto = s · Q · (P^sal − P^ent);
   - costos = c · Q · (P^ent + P^sal);
   - E_{t_{n+1}} = E_{t_n} + bruto − costos.

   Dentro de la posición, E se marca a mercado al cierre de cada barra para medir drawdown y volatilidad.
5. **Siguiente época:** t_{n+1} = barra de salida; τ_n = t_{n+1} − t_n en horas de reloj (`info["duration"]`), que el
   replay buffer usa para descontar con γ^{τ_n}.
6. **Actualización de I y B:** las features y la creencia se recalculan con las barras hasta t_{n+1} (recursiones de la
   sección 2). La cuenta se actualiza con el nuevo E: máximo, última operación y horas plano.

**Episodio.**
- **Entrenamiento:** 720 barras (≈ 30 días) con inicio aleatorio. `terminated` solo si E cae por debajo del 50 % del
  capital inicial del episodio (ruina). El fin de la ventana es `truncated` (S18: el fin de datos no es un evento de la
  tarea).
- **Evaluación:** un solo episodio que recorre la partición completa, sin corte por ruina.
- **Qué se pierde:** el valor después de 30 días solo entra por bootstrap, y el estado de la cuenta se reinicia en cada
  episodio.

## 6. Contribución, objetivo y política (GC)

### Contribución C(S_{t_n}, x_{t_n}, W_{t_n+1}) (elemento 5)

Se paga **una vez por decisión, al cierre** (terminal), y es el rendimiento log neto de la operación con todo el capital
y c = 0.00125 por lado sobre el nocional de entrada y el de salida (`reward.net_log_return`,
`REWARDS[DEFAULT_REWARD]`):

- operar (s = ±1): C = ln(E_{t_{n+1}} / E_{t_n}) = ln(1 + s · (P^sal − P^ent)/P^ent − c · (P^ent + P^sal)/P^ent);
- esperar (s = 0): C = 0.

P^sal depende de W_{t_n+1} (la trayectoria hasta la salida), así que C se conoce hasta t_{n+1}: el entorno la arma con el
`TradeOutcome` de la transición y la devuelve en el `step` que la cierra. El costo de salida se cobra también al cerrar
por stop o take-profit (`test_env_stop_exit_pays_exit_cost`).

- **Por qué esta:** como todo el capital entra en cada operación, Σ_n C_n = ln(E_final / E_inicial). Maximizar la suma de
  contribuciones es maximizar el crecimiento log del capital neto de costos, que es lo que se evalúa, sin coeficientes
  de moldeo. Queda en el orden de 10⁻² por operación de 24 h.
- **Por qué terminal y no por barra:** el agente no actúa en posición, así que las recompensas de las barras intermedias
  no dependen de nada que pueda cambiar. Pagada por barra, la operación suma lo mismo (`reward.per_bar_contributions`),
  pero la relación media/desviación baja de −0.063 por operación a −0.015 por barra (siempre largo, train del fold 1).
- **Por qué esperar vale 0:** el descuento γ^τ ya cobra el tiempo plano. Un costo de oportunidad o una penalización
  serían coeficientes de moldeo más (§3.4).
- **Alternativas en el registro** (comparadas sobre las mismas operaciones en `notebooks/04_contribucion.ipynb`):
  - `vol_scaled_return`: C / (σ̂ · √H); su suma deja de ser el capital;
  - `gross_log_return`: sin costos; operar al azar casi no pierde y premia rotar.

### Objetivo

Criterio de agregación: **recompensa acumulada descontada por la duración real**. Con t_n en horas de reloj,

max_π 𝔼 [ Σ_{n ≥ 0} γ^{t_n − t_0} · C(S_{t_n}, X^π(S_{t_n}), W_{t_n+1}) | S_{t_0} ],   γ por hora,  t_{n+1} − t_n = τ_n.

- El descuento acumulado entre dos épocas es γ^{τ_n}, no γ. Una hora de espera descuenta γ; una operación de 24 h,
  γ^{24}. Descontar γ por decisión haría a una operación larga tan barata en tiempo como una espera de 1 h.
- La esperanza es sobre W (las trayectorias de precio) y sobre el estado inicial: el inicio aleatorio del episodio de
  720 barras en el train. γ < 1 es un parámetro de la solución (estabiliza el bootstrap), no del problema. Con
  γ → 1 el objetivo es 𝔼[ln E_final].
- La métrica de la evaluación es el crecimiento del capital y el Sharpe neto de costos en prueba contra comprar y
  mantener BTC y el S&P 500 (parte 06, `evaluate.performance_metrics`). La contribución está alineada con ella.

La ecuación de Bellman semi-Markov que el agente aproxima es

Q(S, x) = 𝔼 [ C(S, x, W) + γ^{τ} · max_{x'} Q(S', x') ],   S' = S^M(S, x, W),

sin término de bootstrap si la transición termina en ruina (`terminated`). Al truncar por fin de ventana (`truncated`) sí
se hace bootstrap, porque el mercado sigue.

### Política X^π: clase de Powell y cómo se busca

**Clase: VFA para la entrada, con una PFA fija para la salida (híbrido).**

- **Entrada (VFA):** X^π(S_{t_n}) = argmax_{x ∈ {0, 1, 2}} Q̄(S_{t_n}, x | θ). Q̄ es una red neuronal (DQN) que
  aproxima C + γ^τ · V̄(S_{t_{n+1}}), la contribución de la decisión más el valor del estado al que lleva.
  - La observación de 25 variables (sección 2) es la entrada de la red.
  - La política es determinista en la evaluación y ε-greedy en el entrenamiento. La determinista es `train.as_policy`,
    que toma el argmax de la red Q directamente: elige lo mismo que `model.predict(obs, deterministic=True)` en cada
    decisión (`test_fast_policy_matches_sb3_predict`), pero más rápido.
- **Salida (PFA):** el gestor de posiciones es una regla analítica con θ^{gestor} = (H, k_stop, k_take) = (24 h, 1, 1)
  (`env.barrier_levels`, `env.resolve_exit`). No hay optimización embebida: cierra por la primera barrera o por el
  horizonte. Por eso la política completa es un híbrido VFA + PFA.

**Cómo se busca (parte 05, `code/train.py`).**

1. **θ de la red:** DQN de Stable-Baselines3. Es Q-learning con red, off-policy, con replay buffer y red objetivo. El
   buffer es semi-Markov: guarda (S_{t_n}, x, C, S_{t_{n+1}}, τ_n) con τ_n de `info["duration"]`, y el objetivo usa
   γ^{τ_n}.
2. **Hiperparámetros:** rejilla de 16 configuraciones, solo en **validación**, con 3 semillas (0, 1, 2) por
   configuración: tasa de aprendizaje {1e-4, 5e-4} × γ por hora {0.99, 0.995} × red {[64, 64], [128, 128]} ×
   exploración {0.1, 0.3}. H y k quedan fijos (su sensibilidad se mide en la parte 07). Los pasos de entrenamiento
   también son un hiperparámetro: cada corrida se puntúa en validación cada 50k pasos, de 50k a 300k (parada temprana).
   Todas las combinaciones configuración × semilla × pasos quedan en `artifacts/validation_results_fold_{k}.csv`.
3. **Selección (criterio A, fijado antes de ver los resultados):** la mediana entre semillas del **crecimiento log en
   validación** (ln del capital final de un recorrido completo), que es lo mismo que suma la contribución. Se prefirió
   al Sharpe porque el Sharpe de no operar es indefinido. Empates: menos pasos y luego menor índice. Va a prueba la
   **semilla mediana** de la configuración elegida, y se reporta la dispersión entre semillas, no la mejor semilla.
4. **Un modelo por fold:** se entrena con la configuración elegida, se congela y toca la prueba **una sola vez** (parte 06).

**Qué no es.**

- No es PFA pura, porque la entrada sale de un argmax sobre un valor estimado y no de una regla.
- No es DLA, porque no se simulan escenarios futuros al decidir.
- No es CFA, porque no hay un problema determinista con parámetros ajustados.

## 7. Ejemplo numérico de una transición completa

Es la operación de referencia de `tests/test_reference_trade.py` (caso `take_profit_largo`), con H = 4 h, σ̂ = 1 % por
hora y k = 1:

| Paso | Valor |
|---|---|
| Época de decisión t_0 | cierre de la barra 0 (C = 99.6); E = 1; el agente está plano |
| Decisión | x = 1 (largo), s = +1 |
| Entrada (barra 1) | P^ent = O_1 = 100 (no 99.6: la orden se llena a la apertura siguiente) |
| Barreras | σ̂·√H = 0.01·2 = 0.02 → L = 98, U = 102 |
| W (barra 1) | (O, H, L, C) = (100, 101, 99, 100): no toca ninguna barrera |
| W (barra 2) | (100.5, 102.5, 100, 102): el máximo toca U = 102 → sale por take-profit a 102 |
| Cantidad | Q = 1 / 100 = 0.01 BTC |
| P&L bruto | 0.01 · (102 − 100) = 0.02 |
| Costos | 0.00125 · 0.01 · (100 + 102) = 0.002525 |
| Capital | E_1 = 1 + 0.02 − 0.002525 = 1.017475 |
| Contribución (sección 6) | C = ln(1.017475) = 0.017324 |
| Duración | τ_0 = 2 h (de la decisión al cierre de la barra de salida); 2 h en posición |
| Estado siguiente | t_1 = cierre de la barra 2, plano; caída desde el máximo = 0; última operación = +1.75 % |

Si la barra 2 hubiera sido (100, 102.5, 97.5, 100), tocaría ambos niveles: con stop primero sale a 98 (neto −0.022475) y
con take-profit primero a 102 (`test_intrabar_ambiguity`).

## 8. Decisiones de diseño de la formulación

Resumen. Cada fila tiene su alternativa descartada y su evidencia completa en `docs/decisiones.md`.

| Elemento | Elección | Alternativa descartada | Evidencia (train) |
|---|---|---|---|
| Contenido del estado (I, B) | 22 features estacionarias en 6 familias + 3 de la cuenta | ventana de precios crudos (plantilla de clase) | precios no estacionarios (BTC de ≈ 4 mil a ≈ 100 mil USDT); la volatilidad informa el tamaño del movimiento en los 25 desfases (parte 02) |
| Estado físico | posición, entrada y tiempo en posición dentro de la transición; cuenta en la observación | meterlas en la observación | serían constantes en todas las épocas de decisión |
| Espacio de decisión | esperar / largo / corto, solo estando plano | decidir en cada barra; solo largo | ≈ 95 % de "mantener" diluye el crédito; 46.8 % de las ventanas de 24 h son de caída y 2018 cae −73 % |
| Periodo de tenencia | H = 24 h; barreras a 1·σ̂·√H | barreras fijas en %; k = (2, 2) | σ̂ varía 6.3× (p5 a p95); con k = (2, 2) el 83 % sale por horizonte |
| Resolución intrabarra | stop primero | orden por cercanía a la apertura | 0.13 % de operaciones ambiguas en entradas al azar; Δ en prueba en la parte 07 |
| Episodio | 720 barras con inicio aleatorio; ruina = `terminated` | todo el train como un episodio | evita memorizar una trayectoria (S18); ≈ 30 operaciones por episodio |
| Contribución (GC) | log neto de la operación, terminal, 0.125 % por lado; esperar = 0 | (C) log bruto; (B) log neto / (σ̂·√H); pago por barra | sobre las mismas operaciones del train del fold 1, sin costos siempre largo suma +0.53 mientras su capital real cae a 0.6 % (−5.11 en log); (B) da −184 al aleatorio contra −5.86 y su suma ya no es el capital; por barra, media/desv. de −0.063 a −0.015 (≈ 17 barras por operación) |
| Clase de política (GC) | VFA (DQN con γ^τ) para entrar + PFA fija (gestor) para salir | PFA pura (regla de bandas); decidir cierre con la red | el agente solo decide plano (sección 3); hiperparámetros solo en validación con ≥ 3 semillas (`validation_results_fold_{k}.csv`, parte 05) |
