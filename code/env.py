"""Parte 03 — Entorno Gymnasium semi-Markov con gestor de posiciones para BTCUSDT horario.

Una época de decisión es el cierre de una barra en la que el agente está **plano**. Con la acción elige esperar una
barra o abrir una posición larga o corta; la orden se llena en la apertura de la barra siguiente y el gestor de
posiciones la cierra en el primero de estos eventos, recorriendo la trayectoria OHLC barra por barra:

1. stop-loss tocado (por la apertura si hubo gap, o por el mínimo/máximo dentro de la barra);
2. take-profit tocado (ídem);
3. horizonte H alcanzado (salida al cierre de la última barra dentro de las H horas);
4. fin de los datos de la partición (`fin_datos`, cierre forzado al último cierre).

Un `step` va de una época de decisión a la siguiente: devuelve el estado al cierre, la contribución pagada al cierre y,
en `info["duration"]`, la duración real τ en horas de reloj (formulación semi-Markov, §3.3). No hay cierre explícito:
el agente no decide mientras está en posición (ver `docs/decisiones.md`).

Supuestos de ejecución (§3.10): se llena a la apertura de la barra siguiente sin slippage ni impacto, al nivel exacto del
stop o del take-profit cuando se tocan dentro de la barra y a la apertura cuando la barra abre más allá del nivel. Si en
una misma barra se tocan ambos niveles, `intrabar="stop_first"` (por defecto, conservador) ejecuta el stop.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

COST_PER_SIDE = 0.00125
HORIZON_HOURS = 24
K_STOP = 1.0
K_TAKE = 1.0
INTRABAR = "stop_first"
ALLOW_SHORT = True
EPISODE_BARS = 720
RUIN_FRACTION = 0.5
INITIAL_EQUITY = 1.0
OBS_CLIP = 5.0

WAIT, LONG, SHORT = 0, 1, 2
EXIT_REASONS = ("horizonte", "stop", "take_profit", "fin_datos")
ACCOUNT_FEATURES = ["equity_drawdown", "last_trade_return", "hours_flat"]


@dataclass
class TradeOutcome:
    """Resultado de una transición que se le pasa a la función de contribución (`code/reward.py`).

    Para una espera (`side = 0`) los campos de precios y costos valen 0. Rendimientos sobre el capital antes de la
    decisión; `entry_vol` es la volatilidad horaria estimada en la decisión (la que fijó las barreras).
    """

    side: int
    duration_hours: float
    gross_return: float
    cost_return: float
    net_return: float
    net_log_return: float
    entry_vol: float
    exit_reason: str = ""


class TradingEnv(gym.Env):
    """Entorno semi-Markov de BTCUSDT: el agente decide solo estando plano; el gestor cierra la posición.

    Parámetros
    ----------
    ohlcv : DataFrame de una partición (`FoldData.ohlcv[nombre]`) con `open, high, low, close` e índice horario.
    features : DataFrame normalizado alineado con `ohlcv` (`FoldData.features[nombre]`).
    barrier_vol : Serie alineada con la volatilidad horaria **sin normalizar** que fija las barreras
        (`FoldData.raw_features[nombre]["belief_vol"]`).
    reward_fn : función `TradeOutcome -> float` (la contribución, parte 04).
    mode : "train" (episodios de `episode_bars` barras con inicio aleatorio; `terminated` si el capital cae por
        debajo de `RUIN_FRACTION` del inicial) o "eval" (un solo episodio que recorre la partición completa en orden,
        sin corte por ruina, con bitácora y curva de capital).
    horizon_hours, k_stop, k_take : periodo de tenencia; stop y take-profit a k·σ̂·√H del precio de entrada.
    intrabar : "stop_first" o "tp_first" (resolución cuando ambos niveles se tocan en la misma barra).
    obs_lag : barras de retraso de la observación (0 = sin retraso; 1 para el experimento de la parte 07).
    cost : costo por lado (0.125 %), sobre el nocional de entrada y el de salida.
    """

    metadata = {"render_modes": []}

    def __init__(self, ohlcv, features, barrier_vol, reward_fn, mode="train", horizon_hours=HORIZON_HOURS,
                 k_stop=K_STOP, k_take=K_TAKE, intrabar=INTRABAR, allow_short=ALLOW_SHORT,
                 episode_bars=EPISODE_BARS, obs_lag=0, cost=COST_PER_SIDE):
        super().__init__()
        if mode not in ("train", "eval") or intrabar not in ("stop_first", "tp_first"):
            raise ValueError("mode ∈ {train, eval} e intrabar ∈ {stop_first, tp_first}")
        self.index = ohlcv.index
        self.hours = ((ohlcv.index - ohlcv.index[0]).total_seconds() / 3600).to_numpy()
        self.open, self.high, self.low, self.close = (ohlcv[c].to_numpy(float) for c in ("open", "high", "low", "close"))
        self.features = features.to_numpy(np.float32)
        self.feature_names = list(features.columns)
        self.barrier_vol = np.asarray(barrier_vol, float)
        self.reward_fn = reward_fn
        self.mode, self.horizon, self.k_stop, self.k_take = mode, horizon_hours, k_stop, k_take
        self.intrabar, self.episode_bars, self.obs_lag, self.cost = intrabar, episode_bars, obs_lag, cost
        self.n = len(ohlcv)

        self.action_space = spaces.Discrete(3 if allow_short else 2)
        n_obs = self.features.shape[1] + len(ACCOUNT_FEATURES)
        self.observation_space = spaces.Box(-OBS_CLIP, OBS_CLIP, shape=(n_obs,), dtype=np.float32)

    # ------------------------------------------------------------------ Gymnasium

    def reset(self, *, seed=None, options=None):
        """Inicia un episodio: inicio aleatorio con `self.np_random` en entrenamiento, el principio en evaluación."""
        super().reset(seed=seed, options=options)
        first = self.obs_lag
        if self.mode == "train":
            last_start = max(first, self.n - self.episode_bars - 1)
            self.t = int(self.np_random.integers(first, last_start + 1))
            self.end = min(self.t + self.episode_bars, self.n - 1)
        else:
            self.t, self.end = first, self.n - 1
        self.equity = self.peak = self.episode_start_equity = INITIAL_EQUITY
        self.last_trade_return, self.flat_since = 0.0, self.hours[self.t]
        self.trades, self.equity_curve = [], [(self.index[self.t], self.equity)]
        return self._obs(), {"time": self.index[self.t]}

    def step(self, action):
        """Ejecuta la decisión tomada al cierre de la barra `t` y avanza a la siguiente época de decisión."""
        action = int(action)
        side = {WAIT: 0, LONG: 1, SHORT: -1}[action]
        t0 = self.t
        if side == 0:
            outcome, info = self._wait()
        else:
            outcome, info = self._trade(side)
        reward = float(self.reward_fn(outcome))
        self.peak = max(self.peak, self.equity)
        # La ruina termina el episodio solo al entrenar; en evaluación se recorre la partición completa para medirla.
        terminated = bool(self.mode == "train" and self.equity < RUIN_FRACTION * self.episode_start_equity)
        truncated = bool((not terminated) and self.t >= self.end)
        info.update(duration=float(self.hours[self.t] - self.hours[t0]), decision_time=self.index[t0],
                    equity=self.equity, outcome=outcome)
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ transición

    def _wait(self):
        """Esperar: el agente sigue plano una barra; la contribución recibe una espera de duración τ."""
        self.t += 1
        self._mark(self.t, self.equity)
        tau = self.hours[self.t] - self.hours[self.t - 1]
        outcome = TradeOutcome(0, tau, 0.0, 0.0, 0.0, 0.0, float(self.barrier_vol[self.t - 1]))
        return outcome, {"exit_reason": ""}

    def _trade(self, side):
        """Abre en la apertura de t + 1 y deja que el gestor cierre la posición; actualiza capital y bitácora."""
        t, entry_bar = self.t, self.t + 1
        entry_price = self.open[entry_bar]
        vol = float(self.barrier_vol[t])
        stop, take = barrier_levels(entry_price, side, vol, self.horizon, self.k_stop, self.k_take)
        exit_bar, exit_price, reason, ambiguous = resolve_exit(
            self.open, self.high, self.low, self.close, self.hours, entry_bar, side, stop, take,
            self.horizon, self.intrabar, last_bar=self.n - 1)

        equity_before = self.equity
        qty = equity_before / entry_price
        gross = side * qty * (exit_price - entry_price)
        costs = self.cost * qty * (entry_price + exit_price)
        net = gross - costs
        self._mark_path(entry_bar, exit_bar, side, qty, entry_price, equity_before)
        self.equity = equity_before + net
        self.t = exit_bar
        self.equity_curve[-1] = (self.index[exit_bar], self.equity)
        self.last_trade_return = net / equity_before
        self.flat_since = self.hours[exit_bar]

        outcome = TradeOutcome(side, self.hours[exit_bar] - self.hours[t], gross / equity_before,
                               costs / equity_before, net / equity_before, float(np.log(self.equity / equity_before)),
                               vol, reason)
        record = {
            "decision_time": self.index[t], "entry_time": self.index[entry_bar], "exit_time": self.index[exit_bar],
            "side": "largo" if side == 1 else "corto", "exit_reason": reason, "entry_price": entry_price,
            "exit_price": exit_price, "stop_level": stop, "take_level": take, "quantity": qty,
            "notional": qty * entry_price, "gross_pnl": gross, "costs": costs, "net_pnl": net,
            "equity_before": equity_before, "equity_after": self.equity,
            "duration_hours": self.hours[exit_bar] - self.hours[entry_bar] + 1, "bars_held": exit_bar - entry_bar + 1,
            "intrabar_ambiguous": ambiguous, "entry_vol": vol,
        }
        if self.mode == "eval":
            self.trades.append(record)
        return outcome, dict(record)

    # ------------------------------------------------------------------ observación y contabilidad

    def _obs(self):
        """Features normalizadas de la barra t − obs_lag + estado de la cuenta (el agente siempre está plano aquí)."""
        account = np.array([
            self.equity / self.peak - 1,
            np.clip(10 * self.last_trade_return, -OBS_CLIP, OBS_CLIP),
            min(self.hours[self.t] - self.flat_since, 168) / 168,
        ], dtype=np.float32)
        return np.concatenate([self.features[self.t - self.obs_lag], account]).astype(np.float32)

    def _mark(self, bar, equity):
        """Agrega un punto a la curva de capital (solo en evaluación)."""
        if self.mode == "eval":
            self.equity_curve.append((self.index[bar], equity))

    def _mark_path(self, entry_bar, exit_bar, side, qty, entry_price, equity_before):
        """Valúa la posición a mercado al cierre de cada barra mientras está abierta (para MDD y volatilidad)."""
        if self.mode != "eval":
            return
        entry_cost = self.cost * qty * entry_price
        for bar in range(entry_bar, exit_bar + 1):
            self._mark(bar, equity_before - entry_cost + side * qty * (self.close[bar] - entry_price))

    def trade_log(self):
        """Bitácora de operaciones del episodio de evaluación como DataFrame."""
        return pd.DataFrame(self.trades)

    def equity_series(self):
        """Curva de capital (valuada al cierre de cada barra) del episodio de evaluación."""
        times, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(times), name="equity")


def barrier_levels(entry_price, side, vol, horizon_hours, k_stop, k_take):
    """Niveles de stop-loss y take-profit a k·σ̂·√H (rendimiento simple) del precio de entrada.

    `vol` es la volatilidad horaria estimada al decidir, así que las barreras se adaptan al régimen.
    Para un largo el stop queda abajo y el take-profit arriba; para un corto, al revés.
    """
    width = vol * np.sqrt(horizon_hours)
    stop = entry_price * (1 - side * k_stop * width)
    take = entry_price * (1 + side * k_take * width)
    return stop, take


def resolve_exit(open_, high, low, close, hours, entry_bar, side, stop, take, horizon_hours, intrabar, last_bar):
    """Recorre la trayectoria OHLC desde la barra de entrada y devuelve (barra, precio, razón, ambigua).

    En cada barra j (la de entrada incluida, que abre al precio de entrada):
    - si j > entrada y la apertura ya cruzó un nivel (gap, incluidas horas faltantes), se llena a la apertura;
    - si el mínimo/máximo toca un solo nivel, se llena al nivel;
    - si toca ambos, decide `intrabar` y la operación se marca como ambigua.
    Si nada se toca, sale al cierre de la última barra que empieza antes de entrada + H horas; si los datos se
    acaban antes, sale al cierre de `last_bar` con razón `fin_datos`.
    """
    deadline = hours[entry_bar] + horizon_hours
    j = entry_bar
    while j <= last_bar and hours[j] < deadline:
        if j > entry_bar:
            if side * (open_[j] - stop) <= 0:
                return j, open_[j], "stop", False
            if side * (open_[j] - take) >= 0:
                return j, open_[j], "take_profit", False
        adverse, favorable = (low[j], high[j]) if side == 1 else (high[j], low[j])
        hit_stop = side * (adverse - stop) <= 0
        hit_take = side * (favorable - take) >= 0
        if hit_stop and hit_take:
            return (j, stop, "stop", True) if intrabar == "stop_first" else (j, take, "take_profit", True)
        if hit_stop:
            return j, stop, "stop", False
        if hit_take:
            return j, take, "take_profit", False
        j += 1
    if j > last_bar:
        return last_bar, close[last_bar], "fin_datos", False
    return j - 1, close[j - 1], "horizonte", False


def make_env(fold_data, split, reward_fn, **params):
    """Construye el entorno de una partición de un fold (`FoldData` de `code/data.py`)."""
    return TradingEnv(fold_data.ohlcv[split], fold_data.features[split],
                      fold_data.raw_features[split]["belief_vol"], reward_fn, **params)
