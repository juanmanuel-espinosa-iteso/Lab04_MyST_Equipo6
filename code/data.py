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
ASSIGNMENT_ROWS = 75_558  # registros que menciona el enunciado (§3.1)

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
        "microsecond_unix_rows": int((raw["Unix"] >= 10**13).sum()),
        "microsecond_unix_range": [str(stamps[raw["Unix"] >= 10**13].min()), str(stamps[raw["Unix"] >= 10**13].max())],
        "rows_stated_in_assignment": ASSIGNMENT_ROWS,
        "row_count_note": (
            f"El enunciado menciona {ASSIGNMENT_ROWS:,} registros. El archivo trae {len(raw):,} filas de datos "
            f"(sin la línea de la URL ni el encabezado), {len(raw) - ASSIGNMENT_ROWS} más que el enunciado; ningún "
            f"conteo natural da {ASSIGNMENT_ROWS:,}: hay {len(df):,} marcas de tiempo únicas y {len(grid):,} horas en "
            f"el rango. La diferencia con las barras únicas son {n_duplicates} filas repetidas: de las "
            f"{int((raw['Unix'] >= 10**13).sum())} filas con Unix en microsegundos (2025-02-12 a 2025-03-11), "
            f"{n_duplicates} repiten una barra que también viene en milisegundos (OHLCV idéntico). "
            f"Probablemente el conteo del enunciado viene de otra descarga del archivo; se usan las {len(df):,} barras únicas."
        ),
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
    volatility : volatilidad realizada horaria (raíz de la media de r²/Δh) de 24 y 168 barras, Parkinson de 24
                 barras y razón 24/168.
    trend      : distancia del cierre a su EMA de 24, 168 y 720 h; caída desde el máximo de 720 h.
    volume     : z-score del log-volumen en USDT contra 168 barras; razón del volumen medio 24/168 barras.
    (Volatilidad realizada, Parkinson y volumen usan barras observadas, ver `_bar_features`; el resto, la malla
    horaria de reloj.)
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

    for s in EMA_SPANS:
        f[f"ema_gap_{s}h"] = g["close"] / g["close"].ewm(span=s, adjust=True).mean() - 1
    f[f"drawdown_{DRAWDOWN_WINDOW}h"] = g["close"] / g["close"].rolling(DRAWDOWN_WINDOW, min_periods=DRAWDOWN_WINDOW).max() - 1

    hour, dow = g.index.hour, g.index.day_of_week
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    f = f.loc[ohlcv.index].join(_belief_features(r, ohlcv))
    f = f.join(_bar_features(ohlcv))[FEATURES]
    f.iloc[:WARMUP_HOURS] = np.nan
    return f


def _bar_features(ohlcv):
    """Features de volatilidad realizada, rango y volumen sobre las últimas k barras **observadas**.

    Una hora faltante no tiene máximo, mínimo ni volumen; con ventanas de reloj, después de un hueco largo (hasta
    145 h en 2025) estas features quedarían en NaN varios días y el agente no podría decidir. Con ventanas por barras
    observadas siguen definidas, a costa de cubrir un poco más de tiempo de reloj cuando hay huecos.

    Volatilidad realizada: raíz de la media de r_i² / Δh_i, con r_i el rendimiento log entre barras observadas
    consecutivas y Δh_i las horas entre ellas. Dividir entre Δh_i lleva el rendimiento que abarca un hueco a varianza
    por hora (bajo rendimientos i.i.d.), así que los huecos no la sesgan. En la malla de reloj, cada hora faltante
    contaba como rendimiento 0 y la volatilidad de 24 h salía ≈ 20 % más baja tras huecos de 24 h o más.
    """
    hours_between = ohlcv["gap_hours"] + 1
    r2_per_hour = np.log(ohlcv["close"]).diff().pow(2) / hours_between
    hl = np.log(ohlcv["high"] / ohlcv["low"]) ** 2 / (4 * np.log(2))
    log_volume = np.log1p(ohlcv["volume_usdt"])
    roll = log_volume.rolling(VOLUME_WINDOW)
    volume = ohlcv["volume_usdt"]
    rv = {f"rv_{w}h": np.sqrt(r2_per_hour.rolling(w).mean()) for w in VOL_WINDOWS}
    return pd.DataFrame({
        **rv,
        "vol_ratio_24_168": rv["rv_24h"] / rv["rv_168h"],
        "parkinson_24h": np.sqrt(hl.rolling(24).mean()),
        f"volume_z_{VOLUME_WINDOW}h": (log_volume - roll.mean()) / roll.std(),
        "volume_ratio_24_168": np.log(volume.rolling(24).mean() / volume.rolling(168).mean()),
    })


def _belief_features(r, ohlcv):
    """Creencia sobre la deriva y la volatilidad, con su incertidumbre (aproximación normal, rendimientos i.i.d.).

    μ̂_t = EWMA de los rendimientos horarios r de la malla con vida media `DRIFT_HALFLIFE` (una hora faltante vale 0 y
    el rendimiento que abarca el hueco se cuenta una vez, así que la deriva por hora no se sesga).
    Las varianzas son EWMA **en el tiempo** de r_i²/Δh_i sobre barras observadas (varianza por hora del rendimiento
    entre barras consecutivas; `times=` con la vida media en horas de reloj), para que los huecos no las sesguen:
    σ̂_t con vida media `DRIFT_HALFLIFE` y `belief_vol` con `VOL_HALFLIFE`.
    n_eff = (1 + λ) / (1 − λ) con λ = 0.5^(1/vida media) (tamaño efectivo de muestra de pesos geométricos horarios);
    se(μ̂_t) = σ̂_t / √n_eff.
    """
    lam = 0.5 ** (1 / DRIFT_HALFLIFE)
    n_eff = (1 + lam) / (1 - lam)
    drift = r.ewm(halflife=DRIFT_HALFLIFE, adjust=True).mean().loc[ohlcv.index]
    r2_per_hour = np.log(ohlcv["close"]).diff().pow(2) / (ohlcv["gap_hours"] + 1)
    times = ohlcv.index

    def ewm_vol(halflife):
        return np.sqrt(r2_per_hour.ewm(halflife=pd.Timedelta(hours=halflife), times=times).mean())

    se = ewm_vol(DRIFT_HALFLIFE) / np.sqrt(n_eff)
    return pd.DataFrame({
        "belief_drift": drift,
        "belief_drift_se": se,
        "belief_drift_t": drift / se,
        "belief_vol": ewm_vol(VOL_HALFLIFE),
    })


def forward_log_return(ohlcv, horizon):
    """Objetivo del análisis de features (parte 02): rendimiento log de una posición que entra a la apertura de la
    barra siguiente a t y sale por horizonte de `horizon` horas, con la **misma convención que el entorno**
    (`code/env.py`, `resolve_exit` sin barreras ni costos).

    Entrada: apertura de la barra e siguiente a t. Salida: cierre de la barra que termina justo en hora(e) + horizon;
    si esa hora cae dentro de un hueco de datos, la apertura de la primera barra posterior (al cierre previo al hueco
    todavía no se sabe que faltarán barras). NaN en la última barra o si los datos se acaban antes de la salida.

    **No es causal** y nunca entra al estado: mide lo que pasa después de la decisión tomada al cierre de t.
    """
    hour_ns = 3_600 * 10**9
    times = ohlcv.index.as_unit("ns").asi8  # nanosegundos desde la época (con o sin zona horaria)
    n = len(times)
    out = np.full(n, np.nan)
    entry_bar = np.arange(1, n)
    deadline = times[entry_bar] + horizon * hour_ns
    last_before = np.searchsorted(times, deadline, side="left") - 1  # última barra que empieza antes de la salida
    closes_on_time = times[last_before] + hour_ns == deadline
    after_gap = last_before + 1
    has_after = after_gap < n
    exit_price = np.where(closes_on_time, ohlcv["close"].to_numpy()[last_before],
                          np.where(has_after, ohlcv["open"].to_numpy()[np.minimum(after_gap, n - 1)], np.nan))
    out[:-1] = np.log(exit_price / ohlcv["open"].to_numpy()[entry_bar])
    return pd.Series(out, index=ohlcv.index, name=f"fwd_ret_{horizon}h")


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

    def to_frame(self, train_features):
        """Tabla de los estadísticos por feature (para `artifacts/normalizer_fold_{k}.csv`): media, desviación,
        recorte y % de valores del train que el recorte toca."""
        z = (train_features[self.mean.index] - self.mean) / self.std
        return pd.DataFrame({
            "family": [next(fam for fam, cols in FEATURE_FAMILIES.items() if name in cols) for name in self.mean.index],
            "mean": self.mean, "std": self.std, "clip": self.clip,
            "pct_train_clipped": 100 * (z.abs() > self.clip).mean(),
        }).rename_axis("feature")


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


SP500_TICKER = "^GSPC"


def load_sp500(path, start="2017-08-01", end="2026-04-30"):
    """Cierre diario del S&P 500 (`^GSPC`) **solo para el benchmark** de §3.7; nunca entra al estado del agente.

    Lee el snapshot versionado `path` (columnas `date, close`); solo si falta lo descarga con yfinance y lo guarda, para
    que `main.py` no dependa de Yahoo el día de la evaluación. Devuelve una Serie con índice diario en UTC.
    """
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        import yfinance as yf

        raw = yf.download(SP500_TICKER, start=start, end=end, auto_adjust=False, progress=False)
        close = raw["Close"].squeeze().rename("close")
        close.index = pd.DatetimeIndex(close.index).tz_localize(None).rename("date")
        close.to_frame().to_csv(path, float_format="%.4f")
    sp = pd.read_csv(path, index_col="date", parse_dates=True)["close"]
    sp.index = sp.index.tz_localize("UTC")
    return sp
