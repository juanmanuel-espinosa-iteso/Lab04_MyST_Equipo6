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

def feature_target_sample(train_ohlcv, train_features, step=SAMPLE_STEP):
    """(X, y) del train: features en t y rendimiento de la apertura de t + 1 al cierre de t + 24 h, cada `step` barras.

    Con `step` = 25 dos objetivos consecutivos no se traslapan y, como 25 no comparte factores con 24, la muestra recorre
    todas las horas del día (cada 24 barras tomaría siempre la misma y anularía las variables de calendario).
    """
    target = data.forward_log_return(train_ohlcv, TARGET_HORIZON).reindex(train_features.index).dropna()
    target = target.iloc[::step]
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


# ====================================================================== Parte 06: evaluación en prueba

DAYS_PER_YEAR_CALENDAR = 365  # BTC opera 24/7; el S&P 500 se arrastra en fines de semana y feriados (mismo calendario)
INITIAL_EQUITY = 1.0
METRICS = ["total_return", "annual_return", "annual_volatility", "sharpe", "sortino", "max_drawdown", "calmar",
           "hit_rate", "avg_trade_hours", "turnover", "n_trades"]


def run_policy(trading_env, policy, seed):
    """Corre `policy(obs) -> acción` en un entorno de evaluación hasta el fin de la partición.

    Devuelve (equity, bitácora): la curva de capital valuada al cierre de cada barra y la bitácora de operaciones del
    episodio. `policy` puede ser un modelo de SB3 (`lambda obs: model.predict(obs, deterministic=True)[0]`) o una
    línea base. La prueba se recorre una sola vez por fold (§3.6).
    """
    obs, _ = trading_env.reset(seed=seed)
    done = False
    while not done:
        obs, _, terminated, truncated, _ = trading_env.step(policy(obs))
        done = terminated or truncated
    return trading_env.equity_series(), trading_env.trade_log()


def baseline_policies(n_actions, seed):
    """Líneas base triviales de §3.5 como funciones `obs -> acción`: aleatorio (con su propio generador), siempre
    largo y siempre plano."""
    rng = np.random.default_rng(seed)
    return {
        "aleatorio": lambda obs: int(rng.integers(n_actions)),
        "siempre_largo": lambda obs: 1,
        "siempre_plano": lambda obs: 0,
    }


def buy_and_hold(prices, start, end, cost, entry=None):
    """Benchmark de comprar y mantener con el mismo costo por lado que la estrategia.

    `prices` es la serie de cierres (horaria para BTC, diaria para el S&P 500). Se compra en `entry` (por defecto el
    primer cierre del periodo) pagando `cost` sobre el nocional, se valúa a mercado en cada cierre y se vende en el último
    cierre del periodo pagando `cost`. Devuelve (equity, bitácora de una operación) con la misma contabilidad del entorno.
    """
    piece = prices.loc[start:end]
    entry_price = piece.iloc[0] if entry is None else entry
    qty = INITIAL_EQUITY / entry_price
    entry_cost = cost * qty * entry_price
    equity = INITIAL_EQUITY - entry_cost + qty * (piece - entry_price)
    exit_cost = cost * qty * piece.iloc[-1]
    equity.iloc[-1] -= exit_cost
    gross = qty * (piece.iloc[-1] - entry_price)
    log = pd.DataFrame([{
        "entry_time": piece.index[0], "exit_time": piece.index[-1], "side": "largo", "exit_reason": "fin_periodo",
        "entry_price": entry_price, "exit_price": piece.iloc[-1], "quantity": qty, "notional": qty * entry_price,
        "gross_pnl": gross, "costs": entry_cost + exit_cost, "net_pnl": gross - entry_cost - exit_cost,
        "equity_before": INITIAL_EQUITY, "duration_hours": (piece.index[-1] - piece.index[0]).total_seconds() / 3600,
    }])
    return equity.rename("equity"), log


def daily_equity(equity, start, end):
    """Capital al final de cada día calendario de [start, end] (último valor conocido; antes del inicio, el inicial).

    Pone a la estrategia, BTC y el S&P 500 en el mismo calendario de 365 días: un día sin datos repite el último capital,
    así que su rendimiento es 0.
    """
    days = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="D")
    eod = equity.groupby(equity.index.floor("D")).last()
    return eod.reindex(days).ffill().fillna(INITIAL_EQUITY)


def performance_metrics(equity_daily, trades):
    """Métricas de §3.7 a partir del capital diario y la bitácora (tasa libre de riesgo 0, anualización con 365 días).

    total_return = E_fin / E_ini − 1; annual_return = (1 + total)^(365/días) − 1; volatilidad, Sharpe y Sortino con
    rendimientos diarios simples (Sortino con la semidesviación respecto a 0); max_drawdown = mín(E / máx previo − 1);
    calmar = annual_return / |MDD|; hit_rate = % de operaciones con P&L neto > 0; avg_trade_hours = duración media;
    turnover = nocional operado (entrada + salida) / capital medio, por año.
    """
    r = equity_daily.pct_change().dropna()
    days = (equity_daily.index[-1] - equity_daily.index[0]).days or 1
    years = days / DAYS_PER_YEAR_CALENDAR
    start_value = equity_daily.iloc[0] if equity_daily.iloc[0] > 0 else INITIAL_EQUITY
    total = equity_daily.iloc[-1] / start_value - 1
    annual = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = r.std() * np.sqrt(DAYS_PER_YEAR_CALENDAR)
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(DAYS_PER_YEAR_CALENDAR)
    mean_annual = r.mean() * DAYS_PER_YEAR_CALENDAR
    mdd = (equity_daily / equity_daily.cummax() - 1).min()
    n_trades = len(trades)
    traded = (trades["notional"] + trades["quantity"] * trades["exit_price"]).sum() if n_trades else 0.0
    return pd.Series({
        "total_return": total, "annual_return": annual, "annual_volatility": vol,
        "sharpe": mean_annual / vol if vol > 0 else np.nan,
        "sortino": mean_annual / downside if downside > 0 else np.nan,
        "max_drawdown": mdd, "calmar": annual / abs(mdd) if mdd < 0 else np.nan,
        "hit_rate": (trades["net_pnl"] > 0).mean() if n_trades else np.nan,
        "avg_trade_hours": trades["duration_hours"].mean() if n_trades else np.nan,
        "turnover": traded / equity_daily.mean() / years, "n_trades": n_trades,
    })


def add_alpha(table, benchmarks=("btc_buy_hold", "sp500")):
    """Agrega, por fold, el rendimiento anualizado en exceso de cada estrategia sobre cada benchmark (alfa simple).

    `table` tiene índice (fold, estrategia) y la columna `annual_return`.
    """
    out = table.copy()
    for bench in benchmarks:
        bench_return = out.xs(bench, level="strategy")["annual_return"]
        out[f"alpha_vs_{bench}"] = out["annual_return"] - bench_return.reindex(out.index.get_level_values("fold")).to_numpy()
    return out


def aggregate_folds(table):
    """Media y desviación estándar entre folds de cada métrica, por estrategia (§3.6: los resultados de prueba se
    reportan agregados). Devuelve filas con índice (`media` | `desv`, estrategia)."""
    grouped = table.groupby(level="strategy")
    return pd.concat({"media": grouped.mean(), "desv": grouped.std()}, names=["fold"])


# ---------------------------------------------------------------------- análisis de operaciones (§3.9)

def worst_trades(trade_logs, raw_features, n=3):
    """Las `n` operaciones de prueba con mayor pérdida neta entre todos los folds, con las variables de estado (sin
    normalizar) en el momento de la decisión. `trade_logs` y `raw_features` son dicts fold -> DataFrame."""
    rows = []
    for k, log in trade_logs.items():
        if len(log):
            log = log.assign(fold=k, net_return=log["net_pnl"] / log["equity_before"])
            state = raw_features[k].reindex(log["decision_time"]).reset_index(drop=True).add_prefix("state_")
            rows.append(pd.concat([log.reset_index(drop=True), state], axis=1))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).nsmallest(n, "net_return")


def max_drawdown_window(equity):
    """Periodo del mayor drawdown de una curva de capital: (máximo previo, fondo, recuperación o None) y su tamaño."""
    drawdown = equity / equity.cummax() - 1
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()
    recovered = equity.loc[trough:][equity.loc[trough:] >= equity.loc[peak]]
    return {"peak": peak, "trough": trough, "recovery": recovered.index[0] if len(recovered) else None,
            "max_drawdown": float(drawdown.min())}


def return_correlation(equity_daily, btc_daily):
    """Correlación y beta de los rendimientos diarios del agente contra los de BTC (§3.9.3). Una correlación alta
    sugiere que el agente es un largo filtrado o apalancado de BTC; una beta cercana a 1, que replica el activo."""
    a, b = equity_daily.pct_change().dropna(), btc_daily.pct_change().dropna()
    a, b = a.align(b, join="inner")
    beta = np.cov(a, b)[0, 1] / b.var() if b.var() > 0 else np.nan
    return {"correlation": float(a.corr(b)), "beta": float(beta), "n_days": int(len(a))}


# ---------------------------------------------------------------------- orquestación por fold

AGENT = "agente"
BENCHMARKS = ("btc_buy_hold", "sp500")


def evaluate_test_fold(fold_data, policies, sp500, test_dates, seed, cost):
    """Evalúa en la prueba de un fold cada política de `policies` (nombre -> `obs -> acción`) y los dos benchmarks.

    La recompensa no interviene en la evaluación (solo la contabilidad), así que el entorno usa una contribución nula.
    Devuelve un dict con, por estrategia, el capital horario (`equity`), el diario (`daily`) y la bitácora (`trades`).
    """
    from code import env

    start, end = test_dates
    out = {}
    for name, policy in policies.items():
        trading_env = env.make_env(fold_data, "test", lambda outcome: 0.0, mode="eval", cost=cost)
        equity, log = run_policy(trading_env, policy, seed)
        out[name] = {"equity": equity, "daily": daily_equity(equity, start, end), "trades": log}
    btc = fold_data.ohlcv["test"]
    equity, log = buy_and_hold(btc["close"], btc.index[0], btc.index[-1], cost, entry=btc["open"].iloc[0])
    out["btc_buy_hold"] = {"equity": equity, "daily": daily_equity(equity, start, end), "trades": log}
    equity, log = buy_and_hold(sp500, start, end, cost)
    out["sp500"] = {"equity": equity, "daily": daily_equity(equity, start, end), "trades": log}
    return out


def metrics_table(results_by_fold):
    """`metrics_summary`: métricas por (fold, estrategia) con alfa, seguidas de la media y la desviación entre folds."""
    rows = {(k, name): performance_metrics(res["daily"], res["trades"])
            for k, results in results_by_fold.items() for name, res in results.items()}
    per_fold = add_alpha(pd.DataFrame(rows).T.rename_axis(["fold", "strategy"]))
    return pd.concat([per_fold, aggregate_folds(per_fold)])


def trade_analysis(results_by_fold, raw_test_features, btc_daily_by_fold, n=3):
    """Análisis de §3.9 del agente: peores operaciones, mayor drawdown por fold y correlación con BTC por fold."""
    logs = {k: res[AGENT]["trades"] for k, res in results_by_fold.items()}
    worst = worst_trades(logs, raw_test_features, n)
    drawdowns = {k: max_drawdown_window(res[AGENT]["equity"]) for k, res in results_by_fold.items()}
    worst_fold = min(drawdowns, key=lambda k: drawdowns[k]["max_drawdown"])
    correlation = {k: return_correlation(res[AGENT]["daily"], btc_daily_by_fold[k]) for k, res in results_by_fold.items()}
    return {"worst_trades": worst, "drawdowns": drawdowns, "worst_drawdown_fold": worst_fold, "correlation": correlation}
