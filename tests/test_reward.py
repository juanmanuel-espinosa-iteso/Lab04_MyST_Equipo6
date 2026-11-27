"""Pruebas de la parte 04: la contribución de operaciones conocidas coincide con el cálculo a mano.

La contribución principal es el rendimiento log neto de la operación, pagado al cierre, con 0.125 % por lado sobre el
nocional de entrada y el de salida. Esperar vale 0.
"""

import numpy as np
import pandas as pd
import pytest

from code import data, env, reward

C = env.COST_PER_SIDE


def outcome(side, entry, exit_, vol=0.01, reason="horizonte"):
    """`TradeOutcome` de una operación a mano con todo el capital (1.0): rendimientos sobre el capital de entrada."""
    gross = side * (exit_ - entry) / entry
    cost = C * (entry + exit_) / entry
    net = gross - cost
    return env.TradeOutcome(side, 24.0, gross, cost, net, float(np.log1p(net)), vol, reason)


def test_net_log_return_by_hand():
    """Largo 100 → 102 por take-profit: ln(1 + 0.02 − 0.00125 · 2.02) = ln(1.017475)."""
    assert reward.net_log_return(outcome(1, 100.0, 102.0, reason="take_profit")) == pytest.approx(np.log(1.017475))


def test_exit_cost_is_charged_on_stop():
    """Una salida por stop también paga el costo de salida, no solo el de entrada."""
    expected = np.log(1 - 0.02 - C * 1.98)
    assert reward.net_log_return(outcome(1, 100.0, 98.0, reason="stop")) == pytest.approx(expected)


def test_short_flips_sign_but_not_costs():
    """Corto 100 → 98: gana el 2 % y paga los mismos costos."""
    assert reward.net_log_return(outcome(-1, 100.0, 98.0)) == pytest.approx(np.log(1 + 0.02 - C * 1.98))


def test_wait_is_zero():
    """Esperar no paga ni cobra nada, en todas las contribuciones del registro."""
    wait = env.TradeOutcome(0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.01)
    assert {name: fn(wait) for name, fn in reward.REWARDS.items()} == {name: 0.0 for name in reward.REWARDS}


def test_vol_scaled_divides_by_sigma_sqrt_h():
    """La alternativa B divide el log neto entre σ̂ · √H; con σ̂ = 0 sigue siendo finita."""
    trade = outcome(1, 100.0, 102.0, vol=0.01)
    assert reward.vol_scaled_return(trade) == pytest.approx(np.log(1.017475) / (0.01 * np.sqrt(24)))
    assert np.isfinite(reward.vol_scaled_return(outcome(1, 100.0, 102.0, vol=0.0)))


def test_gross_ignores_costs():
    """La alternativa C (bruta) no descuenta costos."""
    assert reward.gross_log_return(outcome(1, 100.0, 102.0)) == pytest.approx(np.log(1.02))


def test_registry_names_the_default():
    """La contribución por defecto es la principal y está en el registro."""
    assert reward.REWARDS[reward.DEFAULT_REWARD] is reward.net_log_return


def test_matches_env_on_reference_trade():
    """Dentro del entorno, la contribución de una operación es el log del capital después de cerrarla (inicial = 1)."""
    calm, decision = (100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 99.6)
    bars = [decision, calm, (100.5, 102.5, 100.0, 102.0), calm, calm, calm]
    index = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(range(len(bars)), unit="h")
    ohlcv = pd.DataFrame(bars, columns=["open", "high", "low", "close"], index=index)
    features = pd.DataFrame(np.zeros((len(bars), 1)), index=index)
    trading_env = env.TradingEnv(ohlcv, features, np.full(len(bars), 0.01), reward.REWARDS["net_log_return"],
                                 mode="eval", horizon_hours=4)
    trading_env.reset(seed=0)
    _, contribution, _, _, info = trading_env.step(env.LONG)
    assert info["exit_reason"] == "take_profit"
    assert contribution == pytest.approx(np.log(info["equity"]))
    assert contribution == pytest.approx(np.log(1.017475))


# ------------------------------------------------------------------ comparación con datos y pago por barra

def test_per_bar_sums_to_terminal():
    """Pagar por barra reparte el mismo log neto: suma igual a la terminal, con el costo de entrada en la primera barra."""
    parts = reward.per_bar_contributions(100.0, [101.0, 99.0, 103.0], 102.0, side=1)
    assert parts.sum() == pytest.approx(np.log(1.017475))
    assert parts[0] < np.log(101.0 / 100.0)


@pytest.fixture(scope="module")
def comparison():
    """Comparación de contribuciones en el train del fold 1 (mismas operaciones para las tres)."""
    from main import BTC_CSV, FOLDS

    ohlcv, _ = data.load_ohlcv(BTC_CSV)
    fold = data.build_fold_dataset(ohlcv, data.compute_features(ohlcv), 1, FOLDS[1])
    return reward.reward_comparison(fold, "train", seed=42)


def test_random_agent_loses_with_costs(comparison):
    """Criterio de terminado: el aleatorio tiene contribución media negativa con costos; sin costos, mucho mejor."""
    net = comparison.loc[("aleatorio", "net_log_return")]
    gross = comparison.loc[("aleatorio", "gross_log_return")]
    assert net["mean_per_decision"] < 0
    assert gross["mean_per_trade"] > net["mean_per_trade"] + 0.002


def test_comparison_uses_same_trades(comparison):
    """Las tres contribuciones se calculan sobre las mismas operaciones de cada política."""
    assert set(comparison.index.get_level_values("policy")) == {"aleatorio", "siempre_largo", "siempre_corto"}
    assert (comparison.groupby(level="policy")["trades"].nunique() == 1).all()
    assert (comparison.groupby(level="policy")["decisions"].nunique() == 1).all()
