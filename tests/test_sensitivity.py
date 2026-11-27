"""Pruebas de la parte 07: escenarios de sensibilidad (intrabarra, retraso, latencia y costo extra)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from code import data, env, evaluate
from main import BTC_CSV, FOLDS

COST = env.COST_PER_SIDE
DECISION = (100.0, 101.0, 99.0, 99.6)
CALM = (100.0, 101.0, 99.0, 100.0)
AMBIGUOUS = (100.0, 102.5, 97.5, 100.0)  # toca el stop (98) y el take-profit (102) en la misma barra


def synthetic_fold(bars, vol=0.01):
    """`FoldData` mínimo con una prueba sintética de barras horarias y σ̂ constante (barreras a ±2 % con H = 4 h)."""
    index = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(range(len(bars)), unit="h")
    ohlcv = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=index)
    raw = pd.DataFrame({"belief_vol": np.full(len(bars), vol)}, index=index)
    return SimpleNamespace(ohlcv={"test": ohlcv}, features={"test": raw.astype(np.float32)},
                           raw_features={"test": raw, "train": raw})


def test_each_scenario_changes_one_assumption():
    """Cada escenario cambia un solo parámetro del entorno respecto a la base."""
    scenarios = evaluate.sensitivity_scenarios(COST, latency=0.0002)
    assert scenarios["base"] == {}
    assert scenarios["tp_first"] == {"intrabar": "tp_first"}
    assert scenarios["retraso_1_barra"] == {"obs_lag": 1}
    assert scenarios["latencia_2s"] == {"cost": pytest.approx(COST + 0.0002)}
    assert scenarios["costo_+10pb"] == {"cost": pytest.approx(COST + 0.0010)}


def test_latency_cost_by_hand():
    """σ̂ mediana 0.0072 por hora → en 2 s: 0.0072 · √(2/3600) = 0.0072 / √1800."""
    features = pd.DataFrame({"belief_vol": [0.006, 0.0072, 0.009]})
    assert evaluate.latency_cost(features) == pytest.approx(0.0072 / np.sqrt(1800))


def test_intrabar_scenario_moves_only_the_ambiguous_trade():
    """Una sola operación, ambigua: con stop primero sale en 98 y con take-profit primero en 102. La diferencia de
    rendimiento total es exactamente la de esas dos salidas, y la proporción de ambiguas es 1."""
    fold = synthetic_fold([DECISION, CALM, AMBIGUOUS])
    # El periodo empieza un día antes para que el capital diario parta del inicial (1) y termine en el final.
    table = evaluate.run_sensitivity(fold, lambda: (lambda obs: env.LONG), ("2023-12-31", "2024-01-01"), 0, COST,
                                     env_params={"horizon_hours": 4})

    def net(exit_price):
        return (exit_price - 100) / 100 - COST * (100 + exit_price) / 100

    assert table.loc["base", "total_return"] == pytest.approx(net(98))
    assert table.loc["tp_first", "total_return"] == pytest.approx(net(102))
    assert table.loc["tp_first", "delta_total_return"] == pytest.approx(net(102) - net(98))
    assert table.loc["base", "ambiguous_share"] == 1.0


def test_extra_cost_keeps_the_path_and_charges_each_side():
    """Con siempre largo, el costo extra no cambia las operaciones (mismas entradas y salidas) y cada una paga
    exactamente 10 pb más por lado sobre su nocional de entrada y de salida (prueba real de 2026)."""
    ohlcv, _ = data.load_ohlcv(BTC_CSV)
    fold = data.build_fold_dataset(ohlcv, data.compute_features(ohlcv), 4, FOLDS[4])
    logs = {}
    for cost in (COST, COST + 0.0010):
        trading_env = env.make_env(fold, "test", lambda outcome: 0.0, mode="eval", cost=cost)
        logs[cost] = evaluate.run_policy(trading_env, lambda obs: env.LONG, seed=0)[1]
    base, extra = logs[COST], logs[COST + 0.0010]
    pd.testing.assert_frame_equal(base[["entry_time", "exit_time", "exit_reason"]], extra[["entry_time", "exit_time", "exit_reason"]])
    base_r = base["net_pnl"] / base["equity_before"]
    extra_r = extra["net_pnl"] / extra["equity_before"]
    np.testing.assert_allclose(base_r - extra_r, 0.0010 * (1 + base["exit_price"] / base["entry_price"]), atol=1e-12)
