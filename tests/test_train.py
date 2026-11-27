"""Pruebas de la parte 05: el replay buffer descuenta con γ^τ (semi-Markov) y la búsqueda solo usa validación."""

import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from code import data, env, reward, train

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


# ---------------------------------------------------------------------- rejilla y entrenamiento de una corrida

def test_grid_is_small_and_complete():
    """16 configuraciones distintas (2 × 2 × 2 × 2), cada una con todos los hiperparámetros buscados."""
    assert len(train.GRID) == 16
    assert {tuple(sorted(c.items())) for c in train.GRID}.__len__() == 16
    assert all(set(c) == {"learning_rate", "gamma", "net_arch", "exploration_fraction"} for c in train.GRID)


def test_dqn_uses_the_semi_markov_buffer_with_its_gamma(fold_1):
    """El DQN usa el buffer semi-Markov con la misma γ que el modelo (y sin el modo que comparte obs y next_obs)."""
    model = train.make_dqn(env.make_env(fold_1, "train", reward.net_log_return), train.GRID[3], seed=0)
    assert isinstance(model.replay_buffer, train.SemiMarkovReplayBuffer)
    assert model.replay_buffer.gamma == model.gamma == train.GRID[3]["gamma"]
    assert not model.replay_buffer.optimize_memory_usage


def test_training_is_reproducible_with_the_seed(fold_1):
    """Misma configuración y semilla → mismo puntaje en validación; la curva de entrenamiento registra episodios."""
    runs = [train.train_agent(fold_1, train.GRID[0], seed=7, total_timesteps=8_000) for _ in range(2)]  # > learning_starts
    scores = [train.evaluate_policy(fold_1, "validation", train.as_policy(model), seed=7) for model, _ in runs]
    assert scores[0] == scores[1]
    curve = runs[0][1]
    assert len(curve) > 0 and {"timesteps", "episode_reward"} <= set(curve.columns)


# ---------------------------------------------------------------------- búsqueda y selección

def results_table(val_by_candidate):
    """Tabla de resultados a mano: {(config_id, timesteps): [puntaje de validación por semilla]}."""
    return pd.DataFrame([{"config_id": c, "timesteps": n, "seed": s, "val_log_growth": v}
                         for (c, n), values in val_by_candidate.items() for s, v in enumerate(values)])


def test_select_config_uses_the_median_across_seeds():
    """Gana la mediana entre semillas, no la mejor semilla: la configuración 1 tiene una semilla suertuda."""
    results = results_table({(0, 100): [0.05, 0.06, 0.04], (1, 100): [0.50, -0.30, -0.20]})
    assert train.select_config(results) == (0, 100, 0)  # mediana 0.05 > −0.20; semilla mediana de la 0: la 0


def test_select_config_treats_the_checkpoint_as_a_hyperparameter():
    """El punto de control (pasos de entrenamiento) se elige igual, por mediana en validación (parada temprana)."""
    results = results_table({(0, 100): [0.01, 0.02, 0.03], (0, 200): [-0.4, -0.5, -0.6], (1, 200): [0.0, 0.0, 0.0]})
    assert train.select_config(results) == (0, 100, 1)


def test_select_config_breaks_ties_with_fewer_steps_then_lower_id():
    """Empate en la mediana → menos pasos y luego menor índice (regla fija, no la que se vea mejor después)."""
    results = results_table({(5, 100): [0.0] * 3, (2, 100): [0.0] * 3, (2, 50): [0.0] * 3})
    assert train.select_config(results)[:2] == (2, 50)


def test_search_touches_only_train_and_validation(fold_1, tmp_path):
    """Una búsqueda chica: una fila por (configuración, semilla, punto de control) con métricas de train y validación,
    nunca de prueba, y el modelo guardado en cada punto de control."""
    from main import BTC_CSV, FOLDS

    jobs = train.search_jobs(1, FOLDS[1], BTC_CSV, tmp_path, config_ids=[0], seeds=[0, 1], total_timesteps=1_000,
                             checkpoints=(500, 1_000))
    results, curves = train.hyperparameter_search(jobs, processes=1)
    assert len(results) == 4 and set(curves) == {(0, 0), (0, 1)}
    assert sorted(results["timesteps"].unique()) == [500, 1_000]
    assert not [c for c in results.columns if "test" in c]
    assert {"val_log_growth", "val_sharpe", "train_log_growth"} <= set(results.columns)
    assert all(train.model_path(tmp_path, 1, 0, s, n).exists() for s in (0, 1) for n in (500, 1_000))
