"""Prueba obligatoria 2 (§3.11): causalidad de la observación.

Reconstruir la observación con los datos hasta el índice i debe devolver el mismo vector que el cálculo sobre la
serie completa. Si una feature usara barras posteriores a i (shift negativo, ventana centrada, back-fill o un
estadístico de la serie completa), el vector truncado sería distinto.
"""

import numpy as np
import pytest

from code import data
from main import BTC_CSV, FOLDS

# Tramo corto (≈ 4 meses) que incluye el hueco de 75 h y las barras desfasadas de febrero de 2018.
N_BARS = 3_000


@pytest.fixture(scope="module")
def ohlcv():
    """OHLCV limpio, recortado a las primeras `N_BARS` barras para que la prueba sea rápida."""
    df, _ = data.load_ohlcv(BTC_CSV)
    return df.iloc[:N_BARS]


@pytest.fixture(scope="module")
def full_features(ohlcv):
    """Features calculadas una sola vez sobre todo el tramo."""
    return data.compute_features(ohlcv)


@pytest.mark.parametrize("i", [data.WARMUP_HOURS, 1_200, 1_350, 2_000, N_BARS - 1])
def test_features_use_only_past_bars(ohlcv, full_features, i):
    """Las features de la fila i son idénticas si se calculan solo con las barras 0..i."""
    truncated = data.compute_features(ohlcv.iloc[: i + 1])
    np.testing.assert_allclose(truncated.iloc[-1].to_numpy(), full_features.iloc[i].to_numpy(), rtol=1e-10, atol=1e-12)


def test_normalized_observation_is_causal(ohlcv, full_features):
    """Con el normalizador congelado del fold, la observación normalizada en i tampoco depende del futuro."""
    normalizer = data.Normalizer.fit(full_features.dropna().iloc[:1_000])
    i = 2_500
    truncated = normalizer.transform(data.compute_features(ohlcv.iloc[: i + 1]).iloc[[-1]])
    full = normalizer.transform(full_features.iloc[[i]])
    np.testing.assert_allclose(truncated.to_numpy(), full.to_numpy(), rtol=1e-6)


def test_normalizer_is_fit_only_on_train():
    """El normalizador de cada fold reproduce media 0 y desviación 1 en su train, y no en validación o prueba."""
    df, _ = data.load_ohlcv(BTC_CSV)
    fold = data.build_fold_dataset(df, data.compute_features(df), 1, FOLDS[1])
    train = fold.features["train"]
    raw_train = data.split_fold(data.compute_features(df), FOLDS[1])["train"]
    np.testing.assert_allclose(fold.normalizer.mean.to_numpy(), raw_train.mean().to_numpy())
    assert train.index.max() < fold.features["validation"].index.min()
    assert fold.features["validation"].index.max() < fold.features["test"].index.min()
