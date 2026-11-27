"""Pruebas de la parte 02: regímenes, estadísticos y análisis de features (solo entrenamiento)."""

import numpy as np
import pandas as pd
import pytest

from code import data, evaluate
from main import BTC_CSV, FOLDS, SEED


def daily(values):
    """Serie diaria sintética de precios."""
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="D", tz="UTC"), dtype=float)


@pytest.fixture(scope="module")
def market():
    """OHLCV limpio y features de la serie completa."""
    ohlcv, _ = data.load_ohlcv(BTC_CSV)
    return ohlcv, data.compute_features(ohlcv)


# ------------------------------------------------------------------ regímenes

def test_turning_points_by_hand():
    """100 → 200 → 100 → 200 (días 1 a 4) con umbral de 45 %: mínimo en el día 1, máximo en el 2 y mínimo en el 3.

    El último día (200) no es giro: no hay una caída posterior de 45 % que lo confirme."""
    pivots = evaluate.turning_points(daily([100, 200, 100, 200]), 0.45)
    assert [(str(t.date()), kind) for t, kind in pivots] == [("2020-01-01", "minimo"), ("2020-01-02", "maximo"),
                                                             ("2020-01-03", "minimo")]


def test_small_moves_are_not_turning_points():
    """Una caída de 30 % no alcanza el umbral de 45 %, así que no parte el régimen."""
    assert evaluate.turning_points(daily([100, 110, 77, 110]), 0.45) == []


def test_regime_table_by_hand():
    """Rendimiento y drawdown de cada tramo iguales al cálculo a mano."""
    table = evaluate.regime_table(daily([100, 160, 120, 200, 150, 100, 200]), 0.45)
    assert list(table["direction"]) == ["alza", "baja", "alza"]
    assert table["total_return"].to_numpy() == pytest.approx([1.0, -0.5, 1.0])
    assert table["max_drawdown"].to_numpy() == pytest.approx([-0.25, -0.5, 0.0])


def test_regimes_do_not_overlap_and_cover_train(market):
    """Los tramos de cada fold cubren su train completo, sin traslape, y hay al menos 4 (pregunta 1)."""
    ohlcv, _ = market
    for dates in FOLDS.values():
        close = evaluate.daily_close(evaluate.train_slice(ohlcv, dates))
        table = evaluate.regime_table(close, evaluate.REGIME_THRESHOLD)
        assert len(table) >= 4
        assert table["start"].iloc[0] == close.index[0] and table["end"].iloc[-1] == close.index[-1]
        assert (table["start"].iloc[1:].to_numpy() == table["end"].iloc[:-1].to_numpy()).all()


# ------------------------------------------------------------------ estadísticos

def test_returns_skip_missing_hours():
    """El rendimiento que cruza un hueco no es de una hora, así que se excluye."""
    index = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 05:00"], utc=True)
    ohlcv = pd.DataFrame({"close": [100.0, 110.0, 150.0], "gap_hours": [0, 0, 3]}, index=index)
    assert evaluate.hourly_returns(ohlcv).to_list() == pytest.approx([np.log(1.1)])


def test_stationarity_tells_random_walk_from_noise():
    """ADF no rechaza la raíz unitaria de una caminata aleatoria y sí la de ruido blanco; KPSS, al revés."""
    noise = pd.Series(np.random.default_rng(SEED).normal(size=3_000))
    table = evaluate.stationarity_table(pd.DataFrame({"ruido": noise, "caminata": noise.cumsum()}))
    assert table.loc["ruido", "adf_p"] < 0.01 and table.loc["ruido", "kpss_p"] >= 0.05
    assert table.loc["caminata", "adf_p"] > 0.05 and table.loc["caminata", "kpss_p"] <= 0.01


# ------------------------------------------------------------------ análisis de features

def test_sampling_avoids_overlap_and_covers_every_hour(market):
    """Las muestras están separadas más que el horizonte y recorren las 24 horas del día."""
    ohlcv, features = market
    _, target = evaluate.feature_target_sample(*evaluate.train_frames(ohlcv, features, FOLDS[1]))
    assert target.index.to_series().diff().min() > pd.Timedelta(hours=evaluate.TARGET_HORIZON)
    assert set(target.index.hour) == set(range(24))


def test_target_never_uses_prices_after_train(market):
    """El objetivo de la última muestra termina dentro del train."""
    ohlcv, features = market
    _, target = evaluate.feature_target_sample(*evaluate.train_frames(ohlcv, features, FOLDS[1]), step=1)
    last_exit = target.index.max() + pd.Timedelta(hours=evaluate.TARGET_HORIZON)
    assert last_exit <= evaluate.train_slice(ohlcv, FOLDS[1]).index.max()


def test_analysis_ignores_data_after_train(market):
    """Cambiar todos los precios posteriores al train no cambia el análisis del fold."""
    ohlcv, _ = market
    ohlcv = ohlcv.loc[:"2022-03-31"]
    altered = ohlcv.copy()
    altered.loc["2022-01-01":, ["open", "high", "low", "close"]] *= 3
    tables = []
    for frame in (ohlcv, altered):
        train_ohlcv, train_features = evaluate.train_frames(frame, data.compute_features(frame), FOLDS[1])
        tables.append(evaluate.feature_analysis(train_ohlcv, train_features, SEED, n_permutations=3))
    pd.testing.assert_frame_equal(*tables)


def test_mutual_information_finds_a_planted_signal():
    """Una feature que determina el objetivo supera el umbral de azar; una de ruido independiente queda muy por debajo."""
    rng = np.random.default_rng(SEED)
    signal = rng.normal(size=800)
    x = pd.DataFrame({"senal": signal, "ruido": rng.normal(size=800)})
    table = evaluate.mutual_information(x, pd.Series(signal + 0.2 * rng.normal(size=800)), SEED, n_permutations=50)
    assert table.loc["senal", "mi"] > table.loc["senal", "mi_null_max_95"]
    assert table.loc["ruido", "mi"] < 0.1 * table.loc["senal", "mi"]
    again = evaluate.mutual_information(x, pd.Series(signal), SEED, n_permutations=5)
    pd.testing.assert_frame_equal(again, evaluate.mutual_information(x, pd.Series(signal), SEED, n_permutations=5))
