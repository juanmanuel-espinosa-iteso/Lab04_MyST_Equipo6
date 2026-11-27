"""Pruebas de la parte 05: el replay buffer descuenta con γ^τ (semi-Markov) y la búsqueda solo usa validación."""

import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from code import data, train

OBS = spaces.Box(-5, 5, shape=(2,), dtype=np.float32)
ACT = spaces.Discrete(3)


def filled_buffer(durations, dones=None, truncated=None, gamma=0.99):
    """Buffer con una transición por duración (un solo entorno)."""
    buffer = train.SemiMarkovReplayBuffer(16, OBS, ACT, device="cpu", gamma=gamma, handle_timeout_termination=True)
    dones = dones or [False] * len(durations)
    truncated = truncated or [False] * len(durations)
    for i, (tau, done, trunc) in enumerate(zip(durations, dones, truncated)):
        obs = np.full((1, 2), i, dtype=np.float32)
        buffer.add(obs, obs + 1, np.array([[1]]), np.array([0.01 * i]), np.array([done]),
                   [{"duration": tau, "TimeLimit.truncated": trunc}])
    return buffer


def test_discount_is_gamma_to_the_duration():
    """Cada transición se descuenta con γ^τ: 1 h de espera → γ; una operación de 24 h → γ^24."""
    buffer = filled_buffer([1.0, 24.0, 7.0], gamma=0.99)
    samples = buffer._get_samples(np.array([0, 1, 2]))
    np.testing.assert_allclose(samples.discounts.numpy().ravel(), 0.99 ** np.array([1.0, 24.0, 7.0]), rtol=1e-6)
    np.testing.assert_allclose(samples.rewards.numpy().ravel(), [0.0, 0.01, 0.02])
    np.testing.assert_allclose(samples.observations.numpy()[:, 0], [0, 1, 2])


def test_terminated_does_not_bootstrap_but_truncated_does():
    """Ruina (`terminated`) corta el bootstrap; el fin de ventana (`truncated`) no."""
    buffer = filled_buffer([3.0, 5.0], dones=[True, True], truncated=[False, True])
    samples = buffer._get_samples(np.array([0, 1]))
    np.testing.assert_array_equal(samples.dones.numpy().ravel(), [1.0, 0.0])


def test_missing_duration_is_an_error():
    """Sin `info["duration"]` no hay γ^τ: se falla en vez de descontar como si fuera 1 h."""
    buffer = train.SemiMarkovReplayBuffer(4, OBS, ACT, device="cpu", gamma=0.99)
    with pytest.raises(KeyError):
        obs = np.zeros((1, 2), dtype=np.float32)
        buffer.add(obs, obs, np.array([[0]]), np.array([0.0]), np.array([False]), [{}])


# ---------------------------------------------------------------------- recorridos, puntaje y líneas base


@pytest.fixture(scope="module")
def fold_1():
    """Datos del fold 1."""
    from main import BTC_CSV, FOLDS

    ohlcv, _ = data.load_ohlcv(BTC_CSV)
    return data.build_fold_dataset(ohlcv, data.compute_features(ohlcv), 1, FOLDS[1])


def test_score_by_hand():
    """Crecimiento log = ln(capital final); el Sharpe diario cuenta desde el capital inicial (incluye el primer día)."""
    index = pd.date_range("2024-01-01 00:00", periods=49, freq="h", tz="UTC")
    equity = pd.Series(np.r_[np.ones(24) * 1.1, np.ones(24) * 1.21, 1.21], index=index)
    score = train.score(equity)
    assert score["log_growth"] == pytest.approx(np.log(1.21))
    # rendimientos diarios: +10 % (día 1, desde 1.0), +10 % (día 2), 0 % (día 3) → media 0.0667, desv. 0.0577
    r = np.array([0.1, 0.1, 0.0])
    assert score["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(365))


def test_baselines_with_costs(fold_1):
    """Siempre plano no se mueve; el aleatorio pierde con costos en todas las particiones del fold 1."""
    table = train.run_baselines(fold_1, seed=42)
    assert set(table.index.get_level_values("split")) == {"train", "validation", "test"}
    flat = table.xs("siempre_plano", level="policy")
    assert (flat["log_growth"] == 0).all() and (flat["n_trades"] == 0).all()
    assert (table.xs("aleatorio", level="policy")["log_growth"] < 0).all()
