"""Parte 02 — Estadísticos del análisis exploratorio: regímenes, distribución, autocorrelación, estacionariedad y features.

Todo se calcula **solo con el entrenamiento** de un fold: las funciones reciben datos ya recortados con `train_slice` o
`train_frames`. El objetivo a 24 h también se calcula sobre ese recorte, así que las últimas horas del train quedan en
NaN en vez de usar precios de validación.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from code import data

REGIME_THRESHOLD = 0.45
TARGET_HORIZON = 24
SAMPLE_STEP = 25
N_PERMUTATIONS = 200
ADF_LAGS = 24
LJUNG_BOX_LAGS = (24, 168)
DAYS_PER_YEAR = 365


# ---------------------------------------------------------------------- recorte al entrenamiento

def train_slice(df, fold_dates):
    """Recorta `df` (índice horario) a las fechas de entrenamiento del fold, inclusivas."""
    start, end = fold_dates["train"]
    return df.loc[start:end]


def train_frames(ohlcv, features, fold_dates):
    """(ohlcv, features) del entrenamiento de un fold; las features sin las filas de calentamiento."""
    return train_slice(ohlcv, fold_dates), data.split_fold(features, fold_dates)["train"]


# ---------------------------------------------------------------------- rendimientos y regímenes

def hourly_returns(ohlcv):
    """Rendimientos log entre barras contiguas; se excluyen los que cruzan horas faltantes (no son de una hora)."""
    r = np.log(ohlcv["close"]).diff()
    return r[ohlcv["gap_hours"] == 0].dropna().rename("ret_1h")


def daily_close(ohlcv):
    """Último cierre observado de cada día UTC; un día sin barras repite el cierre anterior."""
    return ohlcv["close"].resample("D").last().ffill()


def turning_points(close, threshold):
    """Máximos y mínimos que separan movimientos de al menos `threshold`, como lista de (fecha, "maximo" | "minimo").

    Un máximo se confirma cuando el precio cae `threshold` desde él; un mínimo, cuando sube 1 / (1 − threshold) − 1
    desde él (el mismo tamaño en logaritmos: 45 % de caída equivale a 82 % de alza). Los giros se conocen **después**
    de ocurrir: sirven para describir el train y nunca entran al estado.
    """
    pivots, direction = [], 0  # direction: 1 sube, −1 baja, 0 aún sin definir
    high = low = close.iloc[0]
    high_date = low_date = close.index[0]
    for date, price in close.items():
        if price > high:
            high, high_date = price, date
        if price < low:
            low, low_date = price, date
        if direction >= 0 and price <= high * (1 - threshold):
            pivots.append((high_date, "maximo"))
            direction, low, low_date = -1, price, date
        elif direction <= 0 and price >= low / (1 - threshold):
            pivots.append((low_date, "minimo"))
            direction, high, high_date = 1, price, date
    return pivots


def regime_table(close, threshold):
    """Características de cada tramo entre giros consecutivos de `close` (diario).

    Por tramo: fechas, días, dirección, rendimiento total, tendencia anualizada (log), volatilidad anualizada y máximo
    drawdown, más su nivel de volatilidad contra la mediana de los tramos. `status` marca el primer y el último tramo,
    que no están delimitados por dos giros confirmados.
    """
    dates = [date for date, _ in turning_points(close, threshold)]
    edges = sorted({close.index[0], *dates, close.index[-1]})
    rows = []
    for start, end in zip(edges[:-1], edges[1:]):
        piece = close.loc[start:end]
        log_returns = np.log(piece).diff().dropna()
        days = (end - start).days
        rows.append({
            "start": start, "end": end, "days": days,
            "direction": "alza" if piece.iloc[-1] > piece.iloc[0] else "baja",
            "total_return": piece.iloc[-1] / piece.iloc[0] - 1,
            "annual_trend_log": log_returns.sum() * DAYS_PER_YEAR / days,
            "annual_volatility": log_returns.std() * np.sqrt(DAYS_PER_YEAR),
            "max_drawdown": (piece / piece.cummax() - 1).min(),
            "status": "completo" if start in dates and end in dates else "incompleto",
        })
    table = pd.DataFrame(rows, index=pd.RangeIndex(1, len(rows) + 1, name="regime"))
    median = table["annual_volatility"].median()
    table["volatility_level"] = np.where(table["annual_volatility"] > median, "alta", "baja")
    return table


# ---------------------------------------------------------------------- distribución, autocorrelación, estacionariedad

def distribution_stats(returns):
    """Media, desviación, asimetría, exceso de curtosis y prueba de normalidad de Jarque–Bera."""
    jb = stats.jarque_bera(returns)
    return pd.Series({"n": len(returns), "mean": returns.mean(), "std": returns.std(), "skew": stats.skew(returns),
                      "excess_kurtosis": stats.kurtosis(returns), "jarque_bera": jb.statistic, "jarque_bera_p": jb.pvalue,
                      "min": returns.min(), "max": returns.max()})


def ljung_box_table(returns):
    """Ljung–Box de rendimientos, absolutos y cuadrados, con la autocorrelación de rezago 1.

    Con n ≈ 37 mil horas la prueba rechaza casi cualquier correlación: el tamaño (`acf_1`) importa más que el valor p.
    """
    rows = []
    for name, x in {"rendimientos": returns, "absolutos": returns.abs(), "cuadrados": returns ** 2}.items():
        test = acorr_ljungbox(x, lags=list(LJUNG_BOX_LAGS))
        rows += [{"series": name, "lags": lag, "acf_1": x.autocorr(1), "statistic": row["lb_stat"], "p_value": row["lb_pvalue"]}
                 for lag, row in test.iterrows()]
    return pd.DataFrame(rows)


def stationarity_table(frame):
    """ADF (H0: raíz unitaria) y KPSS (H0: estacionaria en nivel) de cada columna de `frame`.

    El valor p de KPSS está acotado a [0.01, 0.10] por sus tablas. `verdict` es "estacionaria" si ADF rechaza y KPSS no,
    "no estacionaria" si ocurre lo contrario, "mixta" si ambas rechazan (memoria larga o cambios de nivel) y "no
    concluyente" si ninguna rechaza, todo al 5 %.
    """
    rows = {}
    for name, x in frame.items():
        x = x.dropna()
        adf_stat, adf_p = adfuller(x, maxlag=ADF_LAGS, autolag=None, result_object=False)[:2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)  # valor p fuera de tabla: queda en el límite
            kpss_stat, kpss_p = kpss(x, regression="c", nlags="auto", result_object=False)[:2]
        verdict = {(True, False): "estacionaria", (False, True): "no estacionaria",
                   (True, True): "mixta", (False, False): "no concluyente"}[(adf_p < 0.05, kpss_p < 0.05)]
        rows[name] = {"adf_stat": adf_stat, "adf_p": adf_p, "kpss_stat": kpss_stat, "kpss_p": kpss_p, "verdict": verdict}
    return pd.DataFrame(rows).T.rename_axis("series").astype({c: float for c in ("adf_stat", "adf_p", "kpss_stat", "kpss_p")})


# ---------------------------------------------------------------------- features contra el objetivo

def feature_target_sample(train_ohlcv, train_features, step=SAMPLE_STEP, offset=0):
    """(X, y) del train: features en t y rendimiento a 24 h de una posición que entra a la apertura de t + 1.

    La muestra es una **malla de horas de reloj**: las t con (horas desde el inicio del train) mod `step` = `offset`.
    Con `step` = 25 dos objetivos consecutivos no se traslapan y, como 25 no comparte factores con 24, la muestra recorre
    todas las horas del día (cada 24 h tomaría siempre la misma y anularía las variables de calendario). Fijarla en el
    reloj, y no contar filas válidas, hace que un hueco o un NaN del objetivo solo quite sus puntos sin desplazar el resto
    de la malla. `step` = 1 usa todas las horas (objetivos traslapados).
    """
    target = data.forward_log_return(train_ohlcv, TARGET_HORIZON).reindex(train_features.index)
    hours = (target.index - train_ohlcv.index[0]) // pd.Timedelta(hours=1)
    target = target[np.asarray(hours % step == offset)].dropna()
    return train_features.loc[target.index], target


def mutual_information(x, y, seed, n_permutations=N_PERMUTATIONS):
    """Información mutua (vecinos cercanos, en nats) de cada columna de `x` con `y`, contra su nivel de azar.

    El azar se mide permutando `y`: `mi_p` es la proporción de permutaciones con una IM igual o mayor para esa feature, y
    `mi_null_max_95` el percentil 95 del **máximo** entre features (corrige por mirar todas a la vez). Supone muestras
    intercambiables: el agrupamiento de volatilidad no lo cumple del todo, así que `mi_p` es optimista.
    """
    rng = np.random.default_rng(seed)
    values = y.to_numpy()
    observed = mutual_info_regression(x, values, random_state=seed)
    null = np.array([mutual_info_regression(x, rng.permutation(values), random_state=seed) for _ in range(n_permutations)])
    return pd.DataFrame({
        "mi": observed,
        "mi_p": (1 + (null >= observed).sum(axis=0)) / (1 + n_permutations),
        "mi_null_max_95": np.quantile(null.max(axis=1), 0.95),
    }, index=pd.Index(x.columns, name="feature"))


def offset_robustness(train_ohlcv, train_features, seed, n_permutations=N_PERMUTATIONS, step=SAMPLE_STEP):
    """En cuántos de los `step` desfases de la malla sale significativa cada feature (`feature_robustness_fold_{k}.csv`).

    Por desfase: Spearman con el objetivo (dirección) y con su valor absoluto (tamaño), con p < 0.05, y la IM contra un
    **umbral de azar común** del fold: el percentil 95 del máximo entre features de `n_permutations` permutaciones
    repartidas entre los desfases (`n_permutations` / `step` en cada uno), así el umbral incluye la variación entre
    desfases y cuesta lo mismo que el de una sola muestra. Las muestras de distintos desfases se traslapan: los conteos
    miden qué tan frágil es la conclusión ante la fase de la malla, no son pruebas independientes.
    """
    rng = np.random.default_rng(seed)
    per_offset = max(1, n_permutations // step)
    mi, rho_p, rho_abs_p, rho, null_max = [], [], [], [], []
    for offset in range(step):
        x, y = feature_target_sample(train_ohlcv, train_features, step, offset)
        values = y.to_numpy()
        mi.append(mutual_info_regression(x, values, random_state=seed))
        null_max += [mutual_info_regression(x, rng.permutation(values), random_state=seed).max() for _ in range(per_offset)]
        tests = [stats.spearmanr(x[c], y) for c in x]
        rho.append([t.statistic for t in tests])
        rho_p.append([t.pvalue for t in tests])
        rho_abs_p.append([stats.spearmanr(x[c], y.abs()).pvalue for c in x])
    threshold = np.quantile(null_max, 0.95)
    mi, rho = np.array(mi), np.array(rho)
    return pd.DataFrame({
        "mi_median": np.median(mi, axis=0),
        "mi_null_max_95": threshold,
        "mi_above_null": (mi > threshold).sum(axis=0),
        "spearman_median": np.median(rho, axis=0),
        "spearman_sig": (np.array(rho_p) < 0.05).sum(axis=0),
        "spearman_abs_sig": (np.array(rho_abs_p) < 0.05).sum(axis=0),
        "n_offsets": step,
    }, index=pd.Index(train_features.columns, name="feature"))


def feature_analysis(train_ohlcv, train_features, seed, n_permutations=N_PERMUTATIONS):
    """Tabla por feature de un fold (`feature_analysis_fold_{k}.csv`), solo con su train.

    Sobre la muestra sin traslape: información mutua, Pearson y Spearman con el objetivo, y Spearman con su valor absoluto
    (¿informa la dirección o el tamaño del movimiento?). `mi_all_hours` usa todas las horas, con objetivos traslapados:
    está inflada y se reporta para mostrarlo. Sobre todas las barras: ADF, KPSS y autocorrelación de rezago 1.
    """
    x, y = feature_target_sample(train_ohlcv, train_features)
    x_all, y_all = feature_target_sample(train_ohlcv, train_features, step=1)
    family = {name: fam for fam, names in data.FEATURE_FAMILIES.items() for name in names}
    table = mutual_information(x, y, seed, n_permutations)
    table.insert(0, "family", [family[name] for name in table.index])
    table["mi_all_hours"] = mutual_info_regression(x_all, y_all.to_numpy(), random_state=seed)
    table["pearson"] = [stats.pearsonr(x[c], y).statistic for c in x]
    for label, target in (("spearman", y), ("spearman_abs_target", y.abs())):
        tests = [stats.spearmanr(x[c], target) for c in x]
        table[label], table[f"{label}_p"] = [t.statistic for t in tests], [t.pvalue for t in tests]
    stationarity = stationarity_table(train_features)
    table = table.join(stationarity[["adf_p", "kpss_p", "verdict"]])
    table["acf_1"] = [train_features[c].autocorr(1) for c in table.index]
    table["n_samples"] = len(y)
    return table
