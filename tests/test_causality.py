"""Prueba obligatoria 2 (§3.11): causalidad de la observación.

Reconstruir la observación con los datos hasta el índice i debe devolver el mismo vector que el cálculo sobre la
serie completa. Si una feature usara barras posteriores a i (shift negativo, ventana centrada, back-fill o un
estadístico de la serie completa), el vector truncado sería distinto.
"""

import numpy as np
import pandas as pd
import pytest

from code import data, env
from main import BTC_CSV, FOLDS

# Tramo corto (≈ 6 meses, hasta mediados de febrero de 2018) que incluye el hueco de 32 h (ya limpio) y las 43 barras
# desfasadas de febrero de 2018: la barra 4,189 (2018-02-09 09:00) es la primera después de ese hueco.
N_BARS = 4_500
FIRST_AFTER_GAP = 4_189


@pytest.fixture(scope="module")
def ohlcv():
    """OHLCV limpio, recortado a las primeras `N_BARS` barras para que la prueba sea rápida."""
    df, _ = data.load_ohlcv(BTC_CSV)
    return df.iloc[:N_BARS]


@pytest.fixture(scope="module")
def full_features(ohlcv):
    """Features calculadas una sola vez sobre todo el tramo."""
    return data.compute_features(ohlcv)


def test_slice_contains_the_february_2018_gap(ohlcv):
    """El tramo de la prueba sí cubre el hueco de 32 h de febrero de 2018."""
    assert ohlcv["gap_hours"].iloc[FIRST_AFTER_GAP] == 32
    assert str(ohlcv.index[FIRST_AFTER_GAP]) == "2018-02-09 09:00:00+00:00"


@pytest.mark.parametrize("i", [data.WARMUP_HOURS, 1_200, 3_354, FIRST_AFTER_GAP, FIRST_AFTER_GAP + 1,
                               FIRST_AFTER_GAP + 24, N_BARS - 1])
def test_features_use_only_past_bars(ohlcv, full_features, i):
    """Las features de la fila i son idénticas si se calculan solo con las barras 0..i."""
    truncated = data.compute_features(ohlcv.iloc[: i + 1])
    np.testing.assert_allclose(truncated.iloc[-1].to_numpy(), full_features.iloc[i].to_numpy(), rtol=1e-10, atol=1e-12)


def test_normalized_observation_is_causal(ohlcv, full_features):
    """Con el normalizador congelado del fold, la observación normalizada en i tampoco depende del futuro."""
    normalizer = data.Normalizer.fit(full_features.dropna().iloc[:1_000])
    i = FIRST_AFTER_GAP + 10
    truncated = normalizer.transform(data.compute_features(ohlcv.iloc[: i + 1]).iloc[[-1]])
    full = normalizer.transform(full_features.iloc[[i]])
    np.testing.assert_allclose(truncated.to_numpy(), full.to_numpy(), rtol=1e-6)


@pytest.mark.parametrize("horizon", [1, 4, 24])
def test_forward_return_matches_env_horizon_exit(ohlcv, horizon):
    """El objetivo de la parte 02 es el rendimiento de una salida por horizonte del entorno (sin barreras ni costos):
    entrada a la apertura siguiente y, si el horizonte vence en un hueco, salida a la apertura posterior al hueco.
    Se revisan barras normales y las previas al hueco de 32 h de febrero de 2018."""
    target = data.forward_log_return(ohlcv, horizon)
    arrays = [ohlcv[c].to_numpy() for c in ("open", "high", "low", "close")]
    hours = ((ohlcv.index - ohlcv.index[0]).total_seconds() / 3600).to_numpy()
    for i in [1_000, 2_500, FIRST_AFTER_GAP - 30, FIRST_AFTER_GAP - 5, FIRST_AFTER_GAP - 1, FIRST_AFTER_GAP]:
        exit_ = env.resolve_exit(*arrays, hours, i + 1, 1, -np.inf, np.inf, horizon, "stop_first", len(ohlcv) - 1)
        assert exit_.reason == "horizonte"
        assert target.iloc[i] == pytest.approx(np.log(exit_.price / ohlcv["open"].iloc[i + 1]), abs=1e-12)
    assert np.isnan(target.iloc[-1])
    assert np.isfinite(target.iloc[:-horizon - 1]).all()


@pytest.mark.parametrize("k", [1, 4])
def test_normalizer_is_fit_only_on_train(k):
    """Los estadísticos del normalizador del fold k son la media y la desviación de **su** train (y no las de
    validación); sin el recorte, el train normalizado queda con media 0 y desviación 1. Las particiones van en orden."""
    df, _ = data.load_ohlcv(BTC_CSV)
    features = data.compute_features(df)
    fold = data.build_fold_dataset(df, features, k, FOLDS[k])
    raw = data.split_fold(features, FOLDS[k])

    np.testing.assert_allclose(fold.normalizer.mean.to_numpy(), raw["train"].mean().to_numpy())
    np.testing.assert_allclose(fold.normalizer.std.to_numpy(), raw["train"].std().to_numpy())
    assert not np.allclose(fold.normalizer.std.to_numpy(), raw["validation"].std().to_numpy(), rtol=0.01)

    unclipped = data.Normalizer(fold.normalizer.mean, fold.normalizer.std, clip=np.inf).transform(raw["train"])
    np.testing.assert_allclose(unclipped.mean().to_numpy(), 0, atol=1e-5)
    np.testing.assert_allclose(unclipped.std().to_numpy(), 1, atol=1e-5)

    assert fold.features["train"].index.max() < fold.features["validation"].index.min()
    assert fold.features["validation"].index.max() < fold.features["test"].index.min()
