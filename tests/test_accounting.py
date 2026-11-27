"""Prueba obligatoria 4 (§3.11): identidad contable.

El capital final es igual al capital inicial más la suma del P&L neto de todas las operaciones de la bitácora. Se corre
el agente aleatorio y el siempre largo sobre la prueba real del fold 4 (2026, con horas faltantes y gaps), donde cada
operación paga costos de entrada y salida.
"""

import numpy as np
import pytest

from code import data, env, evaluate
from main import BTC_CSV, FOLDS


@pytest.fixture(scope="module")
def fold_4():
    """Datos del fold 4 (la prueba es enero–abril de 2026)."""
    ohlcv, _ = data.load_ohlcv(BTC_CSV)
    return data.build_fold_dataset(ohlcv, data.compute_features(ohlcv), 4, FOLDS[4])


@pytest.mark.parametrize("policy", ["aleatorio", "siempre_largo"])
def test_final_equity_equals_initial_plus_net_pnl(fold_4, policy):
    """Capital final = inicial + Σ P&L neto; además, en cada operación, neto = bruto − costos y el capital encadena."""
    trading_env = env.make_env(fold_4, "test", lambda outcome: outcome.net_log_return, mode="eval")
    equity, log = evaluate.run_policy(trading_env, evaluate.baseline_policies(3, seed=0)[policy], seed=0)

    assert len(log) > 50
    assert trading_env.equity == pytest.approx(env.INITIAL_EQUITY + log["net_pnl"].sum(), abs=1e-12)
    assert equity.iloc[-1] == pytest.approx(trading_env.equity, abs=1e-12)
    np.testing.assert_allclose(log["gross_pnl"] - log["costs"], log["net_pnl"], atol=1e-12)
    np.testing.assert_allclose(log["equity_before"].iloc[1:], log["equity_after"].iloc[:-1], atol=1e-12)
    np.testing.assert_allclose(log["costs"], env.COST_PER_SIDE * log["quantity"] * (log["entry_price"] + log["exit_price"]))


def test_always_flat_keeps_initial_equity(fold_4):
    """Sin operaciones no hay P&L ni costos: el capital no se mueve."""
    trading_env = env.make_env(fold_4, "test", lambda outcome: outcome.net_log_return, mode="eval")
    equity, log = evaluate.run_policy(trading_env, evaluate.baseline_policies(3, seed=0)["siempre_plano"], seed=0)
    assert log.empty and (equity == env.INITIAL_EQUITY).all()
