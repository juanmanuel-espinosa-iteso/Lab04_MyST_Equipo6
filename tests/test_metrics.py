"""Pruebas de la parte 06: métricas, benchmarks y análisis de operaciones contra valores calculados a mano."""

import numpy as np
import pandas as pd
import pytest

from code import evaluate

COST = 0.00125


def daily(values, start="2024-01-01"):
    """Capital diario sintético en UTC."""
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D", tz="UTC"), dtype=float)


def trades(net, notional=1.0, exit_price=1.0, hours=24):
    """Bitácora sintética con P&L neto `net` por operación."""
    n = len(net)
    return pd.DataFrame({"net_pnl": net, "notional": [notional] * n, "quantity": [1.0] * n,
                         "exit_price": [exit_price] * n, "duration_hours": [hours] * n})


def test_metrics_by_hand():
    """Capital 1 → 1.1 → 0.99 → 1.089: rendimientos +10 %, −10 %, +10 %; MDD −10 %; 2 de 3 operaciones ganan."""
    equity = daily([1.0, 1.1, 0.99, 1.089])
    m = evaluate.performance_metrics(equity, trades([0.1, -0.11, 0.099]))
    r = np.array([0.1, -0.1, 0.1])
    assert m["total_return"] == pytest.approx(0.089)
    assert m["annual_return"] == pytest.approx(1.089 ** (365 / 3) - 1)
    assert m["annual_volatility"] == pytest.approx(r.std(ddof=1) * np.sqrt(365))
    assert m["sharpe"] == pytest.approx(r.mean() * 365 / (r.std(ddof=1) * np.sqrt(365)))
    assert m["sortino"] == pytest.approx(r.mean() * 365 / (np.sqrt((0.1 ** 2) / 3) * np.sqrt(365)))
    assert m["max_drawdown"] == pytest.approx(-0.1)
    assert m["hit_rate"] == pytest.approx(2 / 3)
    assert m["n_trades"] == 3


def test_buy_and_hold_pays_costs_on_both_sides():
    """Comprar en 100 y vender en 110 con 0.125 % por lado: neto = 0.1 − 0.00125 − 0.00125 × 1.1."""
    prices = daily([100.0, 105.0, 110.0])
    equity, log = evaluate.buy_and_hold(prices, prices.index[0], prices.index[-1], COST)
    expected = 1 + 0.1 - COST - COST * 1.1
    assert equity.iloc[-1] == pytest.approx(expected)
    assert equity.iloc[1] == pytest.approx(1 - COST + 0.05)
    assert log["net_pnl"].iloc[0] == pytest.approx(expected - 1)


def test_daily_equity_fills_days_without_data():
    """Un día sin datos repite el último capital; antes del primer dato el capital es el inicial."""
    hourly = pd.Series([1.0, 1.2, 1.3], index=pd.to_datetime(["2024-01-02 05:00", "2024-01-02 23:00",
                                                               "2024-01-04 10:00"], utc=True))
    out = evaluate.daily_equity(hourly, "2024-01-01", "2024-01-05")
    assert out.to_list() == [1.0, 1.2, 1.2, 1.3, 1.3]


def test_alpha_and_aggregation():
    """Alfa = rendimiento anualizado menos el del benchmark en el mismo fold; agregados = media y desviación entre folds."""
    index = pd.MultiIndex.from_product([[1, 2], ["agente", "btc_buy_hold", "sp500"]], names=["fold", "strategy"])
    table = pd.DataFrame({"annual_return": [0.10, 0.30, 0.05, -0.20, 0.10, 0.15]}, index=index)
    out = evaluate.add_alpha(table)
    assert out.loc[(1, "agente"), "alpha_vs_btc_buy_hold"] == pytest.approx(-0.20)
    assert out.loc[(2, "agente"), "alpha_vs_sp500"] == pytest.approx(-0.35)
    agg = evaluate.aggregate_folds(out)
    assert agg.loc[("media", "agente"), "annual_return"] == pytest.approx(-0.05)
    assert agg.loc[("desv", "agente"), "annual_return"] == pytest.approx(np.std([0.10, -0.20], ddof=1))


def test_max_drawdown_window_and_correlation():
    """El mayor drawdown va del máximo (día 2) al fondo (día 4) y se recupera el día 6; un clon de BTC correlaciona 1."""
    equity = daily([1.0, 1.2, 1.0, 0.9, 1.1, 1.25])
    window = evaluate.max_drawdown_window(equity)
    assert (window["peak"].day, window["trough"].day, window["recovery"].day) == (2, 4, 6)
    assert window["max_drawdown"] == pytest.approx(0.9 / 1.2 - 1)
    corr = evaluate.return_correlation(equity, equity * 3)
    assert corr["correlation"] == pytest.approx(1.0) and corr["beta"] == pytest.approx(1.0)
