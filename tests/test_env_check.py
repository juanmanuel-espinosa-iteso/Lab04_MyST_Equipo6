"""Prueba obligatoria 1 (§3.11): el entorno pasa `check_env` de Stable-Baselines3 sin advertencias.

Se corre sobre el entrenamiento real del fold 1 en los dos modos (episodios aleatorios y recorrido completo), con las
advertencias convertidas en errores para que cualquier aviso haga fallar la prueba.
"""

import warnings

import pytest
from stable_baselines3.common.env_checker import check_env

from code import data, env
from main import BTC_CSV, FOLDS


@pytest.fixture(scope="module")
def fold_1():
    """Datos del fold 1 (el normalizador se ajusta con su entrenamiento)."""
    ohlcv, _ = data.load_ohlcv(BTC_CSV)
    return data.build_fold_dataset(ohlcv, data.compute_features(ohlcv), 1, FOLDS[1])


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_check_env_without_warnings(fold_1, mode):
    """`check_env` no lanza errores ni advertencias."""
    trading_env = env.make_env(fold_1, "train", lambda outcome: outcome.net_log_return, mode=mode)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(trading_env, warn=True)


def test_observation_matches_features_and_account(fold_1):
    """La primera observación son las features normalizadas de la barra de decisión más el estado de la cuenta."""
    trading_env = env.make_env(fold_1, "train", lambda outcome: 0.0, mode="eval")
    obs, _ = trading_env.reset(seed=0)
    n_features = len(data.FEATURES)
    assert obs.shape == (n_features + len(env.ACCOUNT_FEATURES),)
    assert (obs[:n_features] == fold_1.features["train"].iloc[0].to_numpy()).all()
    assert (obs[n_features:] == 0).all()
