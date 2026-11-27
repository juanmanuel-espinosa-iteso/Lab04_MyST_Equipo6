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
    assert len(results) == 4 and set(curves) == {(1, 0, 0), (1, 0, 1)}
    assert sorted(results["timesteps"].unique()) == [500, 1_000]
    assert not [c for c in results.columns if "test" in c]
    assert {"val_log_growth", "val_sharpe", "train_log_growth"} <= set(results.columns)
    assert all(train.model_path(tmp_path, 1, 0, s, n).exists() for s in (0, 1) for n in (500, 1_000))


def test_training_curves_figure(tmp_path):
    """La figura de curvas se escribe con curvas y puntos de control de dos semillas."""
    from code import plots

    curves = {s: pd.DataFrame({"timesteps": [100, 200, 300], "episode_reward": [0.1, -0.2, 0.3]}) for s in (0, 1)}
    runs = pd.DataFrame({"seed": [0, 0, 1, 1], "timesteps": [150, 300] * 2, "val_log_growth": [0.1, -0.1, 0.0, 0.2]})
    path = tmp_path / "curvas.png"
    plots.plot_training_curves(curves, runs, train.GRID[0], 150, 1, path)
    assert path.stat().st_size > 10_000


# ---------------------------------------------------------------------- velocidad sin cambiar resultados

def test_fast_policy_matches_sb3_predict(fold_1):
    """La política rápida (red Q directa) elige exactamente lo mismo que `model.predict(deterministic=True)` en cada
    decisión de un recorrido completo de validación."""
    model, _ = train.train_agent(fold_1, train.GRID[0], seed=3, total_timesteps=8_000)
    trading_env = env.make_env(fold_1, "validation", reward.net_log_return, mode="eval")
    obs, _ = trading_env.reset(seed=0)
    fast, done, n = train.as_policy(model), False, 0
    while not done:
        expected = int(model.predict(obs, deterministic=True)[0])
        assert fast(obs) == expected
        obs, _, terminated, truncated, _ = trading_env.step(expected)
        done, n = terminated or truncated, n + 1
    assert n > 500


def test_train_is_scored_only_at_the_last_checkpoint(fold_1, tmp_path):
    """La validación se puntúa en todos los puntos de control; el train (informativo) solo en el último."""
    from main import BTC_CSV, FOLDS

    jobs = train.search_jobs(1, FOLDS[1], BTC_CSV, tmp_path, config_ids=[0], seeds=[0], total_timesteps=1_000,
                             checkpoints=(500, 1_000))
    results, _ = train.hyperparameter_search(jobs, processes=1)
    assert results["val_log_growth"].notna().all()
    assert results.set_index("timesteps")["train_log_growth"].isna().to_dict() == {500: True, 1_000: False}


def test_saved_search_is_reused_only_with_the_same_settings(tmp_path):
    """Reusar una búsqueda guardada solo si sus corridas (folds, configuraciones, semillas, pasos y puntos de control)
    son las mismas; si cambió algo, no se reusa y se vuelve a entrenar."""
    from main import BTC_CSV, FOLDS

    jobs = train.search_jobs(1, FOLDS[1], BTC_CSV, tmp_path, config_ids=[0, 1], seeds=[0])
    results = pd.DataFrame({"fold": [1], "val_log_growth": [0.1]})
    curves = {(1, 0, 0): pd.DataFrame({"timesteps": [10], "episode_reward": [0.0]})}
    path = tmp_path / "search.pkl"
    train.save_search(path, jobs, results, curves)
    loaded = train.load_search(path, jobs)
    assert loaded is not None and loaded[0].equals(results) and loaded[1][(1, 0, 0)].equals(curves[(1, 0, 0)])
    other = train.search_jobs(1, FOLDS[1], BTC_CSV, tmp_path, config_ids=[0, 1], seeds=[0], total_timesteps=1_000)
    assert train.load_search(path, other) is None
    assert train.load_search(tmp_path / "no_existe.pkl", jobs) is None


@pytest.mark.parametrize("module, name, value", [
    ("env", "HORIZON_HOURS", 12), ("env", "K_STOP", 2.0), ("env", "COST_PER_SIDE", 0.002), ("env", "INTRABAR", "tp_first"),
    ("env", "EPISODE_BARS", 360), ("data", "FEATURES", ["ret_1h"]),
])
def test_reuse_is_refused_if_the_environment_or_features_changed(tmp_path, monkeypatch, module, name, value):
    """Si cambia un parámetro del entorno o las features, la búsqueda guardada no se reusa (se volvería a entrenar)."""
    from main import BTC_CSV, FOLDS

    jobs = train.search_jobs(1, FOLDS[1], BTC_CSV, tmp_path, config_ids=[0], seeds=[0])
    path = tmp_path / "search.pkl"
    train.save_search(path, jobs, pd.DataFrame({"fold": [1]}), {})
    assert train.load_search(path, jobs) is not None
    monkeypatch.setattr({"env": env, "data": data}[module], name, value)
    assert train.load_search(path, jobs) is None


# ---------------------------------------------------------------------- ablación por familias (parte 07, GC)

def test_ablation_variants_cover_every_family_and_the_account():
    """Una variante por familia de features, más la cuenta y la base sin quitar nada."""
    assert train.ABLATION_VARIANTS == ["ninguna", *data.FEATURE_FAMILIES, "cuenta"]


@pytest.mark.parametrize("family", list(data.FEATURE_FAMILIES))
def test_ablate_drops_one_family_from_the_observation(fold_1, family):
    """Quitar una familia la saca de las features de todas las particiones (la observación se achica), pero no de las
    features sin normalizar: las barreras siguen usando la volatilidad estimada."""
    ablated = train.ablate(fold_1, family)
    for split in train.SPLITS:
        assert list(ablated.features[split].columns) == [c for c in data.FEATURES
                                                         if c not in data.FEATURE_FAMILIES[family]]
        assert ablated.raw_features[split] is fold_1.raw_features[split]
    trading_env = train.make_train_env(ablated, "ninguna")
    assert trading_env.observation_space.shape == (len(data.FEATURES) - len(data.FEATURE_FAMILIES[family]) + 3,)


def test_account_ablation_zeroes_the_account_in_training_and_evaluation(fold_1):
    """Sin cuenta: al entrenar, las 3 variables de la cuenta llegan en 0; al evaluar, la política las pone en 0 antes de
    decidir, así que dos observaciones que solo difieren en la cuenta dan la misma acción."""
    trading_env = train.make_train_env(fold_1, "cuenta")
    obs, _ = trading_env.reset(seed=0)
    for _ in range(50):
        obs, *_ = trading_env.step(env.LONG)
        assert np.all(obs[-3:] == 0)
    model = train.make_model("dqn", trading_env, train.GRID[0], seed=0)
    policy = train.as_policy(model, mask_account=True)
    base = np.zeros(trading_env.observation_space.shape, dtype=np.float32)
    rng = np.random.default_rng(0)
    for _ in range(20):
        other = base.copy()
        other[-3:] = rng.normal(size=3)
        assert policy(other) == policy(base)


# ---------------------------------------------------------------------- segundo algoritmo: PPO (parte 07, GC)

def test_ppo_grid_is_small(fold_1):
    """8 configuraciones de PPO (2 × 2 × 2), con la red fija en [64, 64] como la elegida de DQN."""
    assert len(train.PPO_GRID) == 8 and all(c["net_arch"] == (64, 64) for c in train.PPO_GRID)
    assert isinstance(train.make_model("ppo", train.make_train_env(fold_1), train.PPO_GRID[0], 0), train.PPO)


def test_fast_policy_matches_predict_for_ppo(fold_1):
    """Con PPO, la política rápida (moda de la distribución) elige lo mismo que `model.predict(deterministic=True)`."""
    model, _ = train.train_agent(fold_1, train.PPO_GRID[0], seed=3, total_timesteps=4_096, algo="ppo")
    trading_env = env.make_env(fold_1, "validation", reward.net_log_return, mode="eval")
    obs, _ = trading_env.reset(seed=0)
    fast, done = train.as_policy(model), False
    while not done:
        expected = int(model.predict(obs, deterministic=True)[0])
        assert fast(obs) == expected
        obs, _, terminated, truncated, _ = trading_env.step(expected)
        done = terminated or truncated


# ---------------------------------------------------------------------- corridas de ablación y de PPO, y sus tablas

def test_ablation_jobs_use_the_chosen_config_and_score_test(fold_1, tmp_path):
    """La ablación reentrena la configuración elegida, en su punto de control, una corrida por variante × semilla; esas
    corridas sí se puntúan en prueba (se reportan, no eligen nada)."""
    from main import BTC_CSV, FOLDS

    jobs = train.ablation_jobs(1, FOLDS[1], BTC_CSV, tmp_path, config_id=8, timesteps=1_000, seeds=[0],
                               total_timesteps=2_000)
    assert [j["variant"] for j in jobs] == train.ABLATION_VARIANTS
    assert all(j["config_id"] == 8 and j["checkpoints"] == (1_000,) and j["score_test"] for j in jobs)
    results, _ = train.hyperparameter_search([j for j in jobs if j["variant"] in ("volume", "cuenta")], processes=1)
    assert set(results["variant"]) == {"volume", "cuenta"}
    assert {"val_log_growth", "test_log_growth", "test_sharpe", "test_n_trades"} <= set(results.columns)


def test_ppo_search_jobs_use_the_ppo_grid(tmp_path):
    """La búsqueda de PPO recorre su propia rejilla (8) con las mismas semillas y puntos de control que DQN."""
    from main import BTC_CSV, FOLDS

    jobs = train.search_jobs(1, FOLDS[1], BTC_CSV, tmp_path, algo="ppo")
    assert len(jobs) == len(train.PPO_GRID) * len(train.SEEDS)
    assert {j["algo"] for j in jobs} == {"ppo"} and not any(j["score_test"] for j in jobs)


def test_ablation_table_by_hand():
    """Por fold y variante: medianas entre semillas en validación y prueba, y su diferencia contra la base."""
    rows = []
    for variant, val, test in (("ninguna", [0.1, 0.2, 0.3], [0.0, -0.1, 0.1]),
                               ("volume", [-0.1, 0.0, 0.1], [-0.2, -0.3, -0.1])):
        rows += [{"fold": 1, "variant": variant, "seed": s, "val_log_growth": v, "test_log_growth": t,
                  "test_sharpe": t, "test_n_trades": 10} for s, (v, t) in enumerate(zip(val, test))]
    table = train.ablation_table(pd.DataFrame(rows))
    row = table.loc[(1, "volume")]
    assert row["val_log_growth"] == pytest.approx(0.0) and row["delta_val"] == pytest.approx(-0.2)
    assert row["test_log_growth"] == pytest.approx(-0.2) and row["delta_test"] == pytest.approx(-0.2)
    assert table.loc[("media", "volume"), "delta_val"] == pytest.approx(-0.2)


def test_training_stops_at_the_last_checkpoint_with_the_same_schedule(fold_1):
    """Con un solo punto de control antes del total, el entrenamiento se detiene ahí, pero con el calendario de
    exploración del total: el modelo es el mismo que el de ese punto en una corrida completa."""
    saved = {}
    full, _ = train.train_agent(fold_1, train.GRID[0], seed=5, total_timesteps=8_000, checkpoints=(6_000, 8_000),
                                on_checkpoint=lambda m, n: saved.setdefault(n, train.evaluate_policy(
                                    fold_1, "validation", train.as_policy(m), 5)))
    short, curve = train.train_agent(fold_1, train.GRID[0], seed=5, total_timesteps=8_000, checkpoints=(6_000,),
                                     on_checkpoint=lambda m, n: None)
    assert short.num_timesteps == 6_000
    assert train.evaluate_policy(fold_1, "validation", train.as_policy(short), 5) == saved[6_000]


def test_chosen_summary_by_hand():
    """Resumen del candidato elegido de un algoritmo: configuración, pasos, semilla mediana y dispersión en validación."""
    results = results_table({(0, 100): [0.05, 0.06, 0.04], (1, 100): [0.50, -0.30, -0.20]})
    summary = train.chosen_summary(results)
    assert (summary["config_id"], summary["timesteps"], summary["seed"]) == (0, 100, 0)
    assert summary["val_median"] == pytest.approx(0.05)
    assert (summary["val_min"], summary["val_max"]) == (pytest.approx(0.04), pytest.approx(0.06))
    assert summary["candidates_above_zero"] == 1 and summary["candidates"] == 2
