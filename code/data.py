"""Parte 01 — Datos: carga del CSV horario de BTCUSDT, features causales, folds y normalizador por fold.

Convención de tiempo: la barra con marca t cubre [t, t + 1 h). El agente decide con la barra t ya cerrada y la orden se
llena en la apertura de la barra siguiente, así que toda feature de la fila t usa solo barras con marca ≤ t.

Las features se calculan sobre una malla horaria regular con el cierre arrastrado hacia adelante (forward-fill, nunca
back-fill) y después se conservan solo las horas observadas: el agente nunca decide en una hora sin datos. Las horas
faltantes no se inventan como barras; la columna `gap_hours` dice cuántas horas faltan antes de cada barra.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Ventanas en horas (la malla es horaria, así que una ventana de k filas son k horas de reloj).
RETURN_LAGS = (1, 4, 24, 168)
VOL_WINDOWS = (24, 168)
EMA_SPANS = (24, 168, 720)
DRAWDOWN_WINDOW = 720
VOLUME_WINDOW = 168
DRIFT_HALFLIFE = 168
VOL_HALFLIFE = 24
WARMUP_HOURS = 720
CLIP_Z = 5.0

FEATURE_FAMILIES = {
    "returns": [f"ret_{k}h" for k in RETURN_LAGS],
    "volatility": [f"rv_{w}h" for w in VOL_WINDOWS] + ["parkinson_24h", "vol_ratio_24_168"],
    "trend": [f"ema_gap_{s}h" for s in EMA_SPANS] + [f"drawdown_{DRAWDOWN_WINDOW}h"],
    "volume": [f"volume_z_{VOLUME_WINDOW}h", "volume_ratio_24_168"],
    "calendar": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
    "belief": ["belief_drift", "belief_drift_se", "belief_drift_t", "belief_vol"],
}
FEATURES = [name for family in FEATURE_FAMILIES.values() for name in family]

_RAW_COLUMNS = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
                "Volume BTC": "volume_btc", "Volume USDT": "volume_usdt", "tradecount": "trades"}


def load_ohlcv(path):
    """Lee el CSV de CryptoDataDownload, lo limpia y lo valida.

    Parámetros
    ----------
    path : ruta a `Binance_BTCUSDT_1h.csv` (primera línea = URL de la fuente; orden descendente).

    Retorna
    -------
    (ohlcv, report) : DataFrame con índice horario UTC ascendente, columnas `open, high, low, close, volume_btc,
        volume_usdt, trades, gap_hours`; y un dict con lo que se encontró y se corrigió.

    Limpieza: la columna `Unix` mezcla milisegundos y microsegundos (desde 2025), lo que duplica barras; se convierte
    todo a milisegundos y se quitan los duplicados exactos. 43 barras de febrero de 2018 empiezan a los :28 minutos
    (mantenimiento de Binance) y se redondean hacia abajo a la hora. No se rellenan horas faltantes.
    """
    raw = pd.read_csv(path, skiprows=1)
    unix_ms = raw["Unix"].where(raw["Unix"] < 10**13, raw["Unix"] // 1000)
    stamps = pd.to_datetime(unix_ms, unit="ms", utc=True)
    off_hour = int((stamps.dt.minute != 0).sum())

    df = raw.rename(columns=_RAW_COLUMNS)[list(_RAW_COLUMNS.values())]
    df.index = stamps.dt.floor("h").rename("datetime")
    n_duplicates = int(df.index.duplicated().sum())
    identical = df[df.index.duplicated(keep=False)].groupby(level=0).nunique().le(1).all().all()
    if not identical:
        raise ValueError("Hay marcas de tiempo duplicadas con valores distintos")
    df = df[~df.index.duplicated()].sort_index()

    validate_ohlcv(df)
    df["gap_hours"] = (df.index.to_series().diff().dt.total_seconds().div(3600).fillna(1) - 1).astype(int)

    grid = pd.date_range(df.index[0], df.index[-1], freq="h")
    missing = grid.difference(df.index)
    report = {
        "rows_read": int(len(raw)),
        "duplicates_removed": n_duplicates,
        "off_hour_bars_floored": off_hour,
        "bars": int(len(df)),
        "first_bar": str(df.index[0]),
        "last_bar": str(df.index[-1]),
        "hours_in_range": int(len(grid)),
        "missing_hours": int(len(missing)),
        "missing_hours_by_year": {int(y): int(n) for y, n in pd.Series(missing.year).value_counts().sort_index().items()},
        "missing_midnight_bars": int((missing.hour == 0).sum()),
        "longest_gap_hours": int(df["gap_hours"].max()),
        "zero_volume_bars": int((df["volume_btc"] == 0).sum()),
        "rows_stated_in_assignment": 75558,
    }
    return df, report


def validate_ohlcv(df):
    """Verifica el OHLCV limpio: índice ordenado y único, sin NaN, precios positivos y low ≤ open, close ≤ high.

    Lanza ValueError con el primer problema encontrado. No modifica `df`.
    """
    prices = df[["open", "high", "low", "close"]]
    checks = {
        "índice ordenado": df.index.is_monotonic_increasing,
        "índice único": df.index.is_unique,
        "sin NaN": not df.isna().any().any(),
        "precios positivos": bool((prices > 0).all().all()),
        "low ≤ open, close ≤ high": bool(((df["low"] <= prices[["open", "close"]].min(axis=1))
                                          & (df["high"] >= prices[["open", "close"]].max(axis=1))).all()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Validación del OHLCV fallida: {failed}")


def _hourly_grid(ohlcv):
    """Pasa el OHLCV a una malla horaria regular: cierre con forward-fill (causal), high/low/volumen en NaN donde falta."""
    grid = ohlcv.reindex(pd.date_range(ohlcv.index[0], ohlcv.index[-1], freq="h", name="datetime"))
    grid["close"] = grid["close"].ffill()
    return grid


def compute_features(ohlcv):
    """Calcula las features causales de `FEATURE_FAMILIES` para cada barra observada.

    Cada valor de la fila t usa solo barras con marca ≤ t (ventanas móviles hacia atrás y EWM con `adjust=True`,
    que solo pondera el pasado). Las primeras `WARMUP_HOURS` horas quedan con NaN y se descartan en `split_fold`.

    Familias
    --------
    returns    : rendimiento log de 1, 4, 24 y 168 h.
    volatility : desviación de rendimientos horarios a 24 y 168 h, Parkinson a 24 h y razón 24/168.
    trend      : distancia del cierre a su EMA de 24, 168 y 720 h; caída desde el máximo de 720 h.
    volume     : z-score del log-volumen en USDT contra 168 barras; razón del volumen medio 24/168 barras.
    (Parkinson y volumen usan barras observadas, ver `_bar_features`; el resto, la malla horaria de reloj.)
    calendar   : hora del día y día de la semana en seno y coseno (BTC opera 24/7, pero la liquidez no es uniforme).
    belief     : creencia sobre la tendencia y el riesgo (componente B_t del estado): deriva μ̂ (EWMA de los rendimientos,
                 vida media 168 h), su error estándar σ̂/√n_eff, el cociente μ̂/se y la volatilidad EWMA de 24 h.

    Retorna un DataFrame con índice = índice de `ohlcv` y columnas = `FEATURES`.
    """
    g = _hourly_grid(ohlcv)
    log_close = np.log(g["close"])
    r = log_close.diff()
    f = pd.DataFrame(index=g.index)

    for k in RETURN_LAGS:
        f[f"ret_{k}h"] = log_close - log_close.shift(k)

    for w in VOL_WINDOWS:
        f[f"rv_{w}h"] = r.rolling(w, min_periods=w // 2).std()
    f["vol_ratio_24_168"] = f["rv_24h"] / f["rv_168h"]

    for s in EMA_SPANS:
        f[f"ema_gap_{s}h"] = g["close"] / g["close"].ewm(span=s, adjust=True).mean() - 1
    f[f"drawdown_{DRAWDOWN_WINDOW}h"] = g["close"] / g["close"].rolling(DRAWDOWN_WINDOW, min_periods=DRAWDOWN_WINDOW).max() - 1

    hour, dow = g.index.hour, g.index.day_of_week
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    f = f.join(_belief_features(r)).loc[ohlcv.index]
    f = f.join(_bar_features(ohlcv))[FEATURES]
    f.iloc[:WARMUP_HOURS] = np.nan
    return f


def _bar_features(ohlcv):
    """Features de rango y volumen sobre las últimas k barras **observadas** (no sobre la malla de reloj).

    Una hora faltante no tiene máximo, mínimo ni volumen; con ventanas de reloj, después de un hueco largo (hasta
    145 h en 2025) estas features quedarían en NaN varios días y el agente no podría decidir. Con ventanas por barras
    observadas siguen definidas, a costa de cubrir un poco más de tiempo de reloj cuando hay huecos.
    """
    hl = np.log(ohlcv["high"] / ohlcv["low"]) ** 2 / (4 * np.log(2))
    log_volume = np.log1p(ohlcv["volume_usdt"])
    roll = log_volume.rolling(VOLUME_WINDOW)
    volume = ohlcv["volume_usdt"]
    return pd.DataFrame({
        "parkinson_24h": np.sqrt(hl.rolling(24).mean()),
        f"volume_z_{VOLUME_WINDOW}h": (log_volume - roll.mean()) / roll.std(),
        "volume_ratio_24_168": np.log(volume.rolling(24).mean() / volume.rolling(168).mean()),
    })


def _belief_features(r):
    """Creencia sobre la deriva y la volatilidad, con su incertidumbre (aproximación normal, rendimientos i.i.d.).

    μ̂_t = EWMA de r con vida media `DRIFT_HALFLIFE`; σ̂_t = raíz de la varianza EWMA con la misma vida media;
    n_eff = (1 + λ) / (1 − λ) con λ = 0.5^(1/vida media) (tamaño efectivo de muestra de pesos geométricos);
    se(μ̂_t) = σ̂_t / √n_eff. `belief_vol` es la volatilidad EWMA de vida media `VOL_HALFLIFE`.
    """
    lam = 0.5 ** (1 / DRIFT_HALFLIFE)
    n_eff = (1 + lam) / (1 - lam)
    drift = r.ewm(halflife=DRIFT_HALFLIFE, adjust=True).mean()
    sigma = np.sqrt(r.ewm(halflife=DRIFT_HALFLIFE, adjust=True).var(bias=False))
    se = sigma / np.sqrt(n_eff)
    return pd.DataFrame({
        "belief_drift": drift,
        "belief_drift_se": se,
        "belief_drift_t": drift / se,
        "belief_vol": np.sqrt(r.pow(2).ewm(halflife=VOL_HALFLIFE, adjust=True).mean()),
    })


def forward_log_return(ohlcv, horizon):
    """Objetivo del análisis de features (parte 02): log(cierre en t + horizon h / apertura de la barra siguiente a t).

    **No es causal** y nunca entra al estado: mide lo que pasa después de la decisión tomada al cierre de t, con el
    llenado en la apertura siguiente. Con horas faltantes usa el último cierre conocido en t + horizon.
    """
    g = _hourly_grid(ohlcv)
    entry = ohlcv["open"].shift(-1)
    exit_close = g["close"].shift(-horizon).reindex(ohlcv.index)
    return np.log(exit_close / entry).rename(f"fwd_ret_{horizon}h")


def split_fold(df, fold_dates):
    """Corta `df` (índice horario) en las particiones de un fold.

    `fold_dates` = {"train": (inicio, fin), "validation": (...), "test": (...)} con fechas 'YYYY-MM-DD' inclusivas
    (el fin incluye todas las horas de ese día). Las filas con NaN (calentamiento de las ventanas) se descartan.
    """
    return {name: df.loc[start:end].dropna() for name, (start, end) in fold_dates.items()}


@dataclass
class Normalizer:
    """z-score por feature con estadísticos del entrenamiento de un fold, recortado a ±`clip`."""

    mean: pd.Series
    std: pd.Series
    clip: float = CLIP_Z

    @classmethod
    def fit(cls, train_features):
        """Ajusta medias y desviaciones **solo** con las features de entrenamiento de un fold."""
        std = train_features.std().replace(0, 1.0)
        return cls(mean=train_features.mean(), std=std)

    def transform(self, features):
        """Aplica los estadísticos congelados (sin reajustar) y recorta a ±clip; devuelve float32."""
        z = (features[self.mean.index] - self.mean) / self.std
        return z.clip(-self.clip, self.clip).astype(np.float32)


@dataclass
class FoldData:
    """Datos de un fold listos para el entorno, alineados por partición.

    `features` (normalizadas) son la observación del agente; `raw_features` (sin normalizar) las usa el entorno donde
    importan las unidades (la volatilidad que fija las barreras) y la bitácora para reportar el estado en la entrada.
    """

    fold: int
    ohlcv: dict
    features: dict
    raw_features: dict
    normalizer: Normalizer


def build_fold_dataset(ohlcv, features, fold, fold_dates):
    """Arma un `FoldData`: corta las particiones, ajusta el normalizador con el train del fold y lo aplica a todas.

    El OHLCV de cada partición se alinea con las filas de features que sobrevivieron al calentamiento.
    """
    parts = split_fold(features, fold_dates)
    normalizer = Normalizer.fit(parts["train"])
    return FoldData(
        fold=fold,
        ohlcv={name: ohlcv.loc[part.index] for name, part in parts.items()},
        features={name: normalizer.transform(part) for name, part in parts.items()},
        raw_features=parts,
        normalizer=normalizer,
    )


def describe_folds(ohlcv, features, folds):
    """Resumen por fold y partición para `artifacts/folds.json`: fechas, barras útiles y horas faltantes."""
    out = {}
    for k, dates in folds.items():
        parts = split_fold(features, dates)
        out[str(k)] = {
            name: {
                "start": str(part.index[0]),
                "end": str(part.index[-1]),
                "bars": int(len(part)),
                "missing_hours": int(ohlcv.loc[part.index, "gap_hours"].iloc[1:].sum()),
            }
            for name, part in parts.items()
        }
    return out
