"""Prueba obligatoria 3 (§3.11): operaciones de referencia calculadas a mano, reproducidas exactamente por el entorno.

Trayectorias sintéticas con H = 4 h, volatilidad horaria σ̂ = 1 % y k = 1, así que las barreras quedan a
σ̂·√H = 2 % del precio de entrada. El agente decide al cierre de la barra 0 y entra a la apertura de la barra 1
(precio 100): para un largo, stop = 98 y take-profit = 102; para un corto, stop = 102 y take-profit = 98.

Contabilidad (capital inicial 1): cantidad = 1 / 100 = 0.01 BTC; costo = 0.125 % × cantidad × (entrada + salida).
Ejemplo de la salida por take-profit en 102: bruto = 0.01 × 2 = 0.02; costos = 0.00125 × 0.01 × 202 = 0.002525;
neto = 0.017475; capital final = 1.017475.
"""

import numpy as np
import pandas as pd
import pytest

from code import env

COST = 0.00125
CALM = (100.0, 101.0, 99.0, 100.0)  # barra que no toca ninguna barrera
# Barra de decisión: cierra en 99.6 y la siguiente abre en 100, así que una entrada al cierre de t (look-ahead) y
# una a la apertura de t + 1 dan P&L distintos y las pruebas los distinguen.
DECISION = (100.0, 101.0, 99.0, 99.6)


def make_env(bars, hours=None, intrabar="stop_first"):
    """Entorno de evaluación sobre barras (open, high, low, close); `hours` permite simular horas faltantes."""
    hours = list(range(len(bars))) if hours is None else hours
    index = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(hours, unit="h")
    ohlcv = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=index)
    features = pd.DataFrame(np.zeros((len(bars), 2)), index=index, columns=["f1", "f2"])
    trading_env = env.TradingEnv(ohlcv, features, np.full(len(bars), 0.01), lambda o: o.net_log_return,
                                 mode="eval", horizon_hours=4, k_stop=1.0, k_take=1.0, intrabar=intrabar)
    trading_env.reset(seed=0)
    return trading_env


def expected_net(side, entry, exit_):
    """P&L neto a mano con capital 1: cantidad = 1/entrada."""
    qty = 1 / entry
    return side * qty * (exit_ - entry) - COST * qty * (entry + exit_)


CASES = {
    # nombre: (barras, horas, lado, acción, barra de salida, precio de salida, razón, duración en h, ambigua)
    "take_profit_largo": ([DECISION, CALM, (100.5, 102.5, 100.0, 102.0), CALM, CALM, CALM], None,
                          1, env.LONG, 2, 102.0, "take_profit", 2, False),
    "stop_largo": ([DECISION, CALM, (100.0, 100.8, 97.5, 98.5), CALM, CALM, CALM], None,
                   1, env.LONG, 2, 98.0, "stop", 2, False),
    "horizonte": ([DECISION, CALM, CALM, CALM, (100.0, 101.0, 99.0, 100.7), CALM], None,
                  1, env.LONG, 4, 100.7, "horizonte", 4, False),
    # La barra 2 abre en 97, debajo del stop (98): se llena a la apertura, así que estuvo 1 h en posición.
    "gap_en_apertura": ([DECISION, CALM, (97.0, 97.5, 96.0, 97.2), CALM, CALM, CALM], None,
                        1, env.LONG, 2, 97.0, "stop", 1, False),
    # Entrada en la hora 1, horizonte en la hora 5; faltan las horas 3 a 6. Al cierre de la barra de la hora 2 no se
    # sabe que vienen huecos: se sale a la apertura de la primera barra posterior (hora 7), con 6 h en posición.
    "horizonte_en_hueco": ([DECISION, CALM, CALM, (100.9, 101.5, 100.2, 101.0), CALM], [0, 1, 2, 7, 8],
                           1, env.LONG, 3, 100.9, "horizonte", 6, False),
    # Igual, pero la primera barra después del hueco abre más allá del take-profit (102): sale ahí, a la apertura.
    "take_profit_tras_hueco": ([DECISION, CALM, CALM, (102.3, 103.0, 101.8, 102.5), CALM], [0, 1, 2, 7, 8],
                               1, env.LONG, 3, 102.3, "take_profit", 6, False),
    "take_profit_corto": ([DECISION, CALM, (99.0, 99.5, 97.9, 98.2), CALM, CALM, CALM], None,
                          -1, env.SHORT, 2, 98.0, "take_profit", 2, False),
    # Falta la hora 3: el horizonte es de reloj (entrada en la hora 1 → última barra que empieza antes de la hora 5),
    # así que sale al cierre de la barra de la hora 4 (índice 3) con 3 barras y 4 horas en posición.
    "hora_faltante": ([DECISION, CALM, CALM, (100.0, 101.0, 99.0, 100.3), CALM], [0, 1, 2, 4, 5],
                      1, env.LONG, 3, 100.3, "horizonte", 4, False),
    "fin_de_datos": ([DECISION, CALM, (100.0, 101.0, 99.0, 100.4)], None,
                     1, env.LONG, 2, 100.4, "fin_datos", 2, False),
}


@pytest.mark.parametrize("name", CASES)
def test_reference_trade(name):
    """Razón de salida, barra, precio, duración y P&L idénticos al cálculo a mano."""
    bars, hours, side, action, exit_bar, exit_price, reason, duration, ambiguous = CASES[name]
    trading_env = make_env(bars, hours)
    _, reward, terminated, truncated, info = trading_env.step(action)
    net = expected_net(side, 100.0, exit_price)

    assert info["exit_reason"] == reason
    assert info["exit_time"] == trading_env.index[exit_bar]
    assert info["exit_price"] == pytest.approx(exit_price, abs=1e-12)
    assert info["duration_hours"] == duration
    assert info["intrabar_ambiguous"] is ambiguous
    assert info["net_pnl"] == pytest.approx(net, abs=1e-12)
    assert trading_env.equity == pytest.approx(1 + net, abs=1e-12)
    assert reward == pytest.approx(np.log(1 + net), abs=1e-12)
    assert not terminated


def test_take_profit_numbers_by_hand():
    """El caso del docstring, con los números escritos: neto 0.017475 y capital 1.017475."""
    trading_env = make_env(CASES["take_profit_largo"][0])
    _, _, _, _, info = trading_env.step(env.LONG)
    assert info["gross_pnl"] == pytest.approx(0.02, abs=1e-12)
    assert info["costs"] == pytest.approx(0.002525, abs=1e-12)
    assert trading_env.equity == pytest.approx(1.017475, abs=1e-12)


@pytest.mark.parametrize("intrabar, reason, price", [("stop_first", "stop", 98.0), ("tp_first", "take_profit", 102.0)])
def test_intrabar_ambiguity(intrabar, reason, price):
    """Si una barra toca stop y take-profit, el supuesto intrabarra decide y la operación queda marcada como ambigua."""
    bars = [DECISION, CALM, (100.0, 102.5, 97.5, 100.0), CALM, CALM, CALM]
    trading_env = make_env(bars, intrabar=intrabar)
    _, _, _, _, info = trading_env.step(env.LONG)
    assert (info["exit_reason"], info["exit_price"], info["intrabar_ambiguous"]) == (reason, price, True)


def test_buy_and_hold_equals_asset_return_minus_costs():
    """Validación de clase (S18–S19) sobre datos reales: un largo sin barreras que dura todo el tramo gana exactamente
    el rendimiento del activo entre la apertura de entrada y el último cierre, menos el costo de entrar y salir."""
    from code import data
    from main import BTC_CSV

    ohlcv, _ = data.load_ohlcv(BTC_CSV)
    piece = ohlcv.loc["2023-03-01":"2023-03-31"]
    features = pd.DataFrame(np.zeros((len(piece), 1)), index=piece.index)
    trading_env = env.TradingEnv(piece, features, np.full(len(piece), 0.01), lambda o: o.net_log_return,
                                 mode="eval", horizon_hours=10_000, k_stop=1e6, k_take=1e6)
    trading_env.reset(seed=0)
    _, _, _, truncated, info = trading_env.step(env.LONG)
    entry, exit_ = piece["open"].iloc[1], piece["close"].iloc[-1]
    assert info["exit_reason"] == "fin_datos" and truncated
    assert trading_env.equity == pytest.approx(exit_ / entry - COST * (1 + exit_ / entry), rel=1e-12)


def test_entry_is_next_open_not_decision_close():
    """La entrada es la apertura de t + 1 (100), no el cierre de t (99.6) con el que se decidió."""
    trading_env = make_env(CASES["take_profit_largo"][0])
    _, _, _, _, info = trading_env.step(env.LONG)
    assert info["entry_price"] == 100.0 != trading_env.close[0]


def test_exit_at_open_flag():
    """La bitácora marca las salidas a la apertura (gap y horizonte vencido en un hueco) y no las demás."""
    flags = {}
    for name in ["gap_en_apertura", "horizonte_en_hueco", "take_profit_tras_hueco", "horizonte", "stop_largo"]:
        bars, hours, *_ = CASES[name]
        trading_env = make_env(bars, hours)
        flags[name] = trading_env.step(env.LONG)[4]["exit_at_open"]
    assert flags == {"gap_en_apertura": True, "horizonte_en_hueco": True, "take_profit_tras_hueco": True,
                     "horizonte": False, "stop_largo": False}


def test_obs_lag_also_delays_barrier_volatility():
    """Con obs_lag = 1 las barreras usan la volatilidad de t − 1, igual que la observación."""
    bars = [DECISION, CALM, CALM, CALM, CALM, CALM]
    index = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(range(len(bars)), unit="h")
    ohlcv = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=index)
    features = pd.DataFrame(np.zeros((len(bars), 1)), index=index)
    vol = np.array([0.01, 0.02, 0.03, 0.03, 0.03, 0.03])
    trading_env = env.TradingEnv(ohlcv, features, vol, lambda o: 0.0, mode="eval", horizon_hours=4, obs_lag=1)
    trading_env.reset(seed=0)  # empieza en t = 1, observa la barra 0
    _, _, _, _, info = trading_env.step(env.LONG)
    assert info["entry_vol"] == 0.01
    assert info["stop_level"] == pytest.approx(CALM[0] * (1 - 0.01 * 2))


def test_trade_log_survives_vecenv_auto_reset():
    """Con DummyVecEnv, el reset automático al terminar no borra la bitácora del episodio de evaluación."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    bars = [DECISION] + [CALM] * 30
    index = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(range(len(bars)), unit="h")
    ohlcv = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=index)
    features = pd.DataFrame(np.zeros((len(bars), 1)), index=index)
    vec = DummyVecEnv([lambda: env.TradingEnv(ohlcv, features, np.full(len(bars), 0.01), lambda o: 0.0,
                                              mode="eval", horizon_hours=4)])
    vec.reset()
    done = False
    while not done:
        _, _, dones, _ = vec.step(np.array([env.LONG]))
        done = bool(dones[0])
    log = vec.envs[0].trade_log()
    assert len(log) == 8 and log["exit_reason"].iloc[-1] == "fin_datos"  # salidas en las barras 4, 8, ..., 28 y 30


def test_semi_markov_duration_and_wait():
    """Esperar avanza una barra (τ = 1 h); una operación avanza de la decisión al cierre de la salida (τ = 2 h)."""
    trading_env = make_env(CASES["take_profit_largo"][0])
    _, reward, _, _, info = trading_env.step(env.WAIT)
    assert (info["duration"], reward, trading_env.t) == (1.0, 0.0, 1)
    trading_env = make_env(CASES["take_profit_largo"][0])
    _, _, _, _, info = trading_env.step(env.LONG)
    assert (info["duration"], trading_env.t) == (2.0, 2)
