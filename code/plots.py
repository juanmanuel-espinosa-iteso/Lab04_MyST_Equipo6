"""Figuras del proyecto. Parte 02: páginas de `artifacts/btcusdt_eda.pdf`, también exportadas a `docs/figuras/`.

Las funciones solo dibujan: los números vienen de `code/evaluate.py`. Todas las páginas usan datos de entrenamiento.
"""

import matplotlib

matplotlib.use("Agg")  # sin ventana: main.py corre igual en cualquier máquina

import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.tsa.stattools import acf, pacf  # noqa: E402

from code import data, evaluate  # noqa: E402

PAGE = (11, 8.5)
INK, MUTED, GRID = "#191919", "#666666", "#e7e7e7"
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
UP, DOWN = "#cce3fd", "#fde0d0"  # fondos de los tramos de alza y de baja
STYLE = {"font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
         "axes.edgecolor": "#c2c2c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False}
ACF_LAGS = 72


def _page(title, subtitle, **subplots):
    """Crea una página carta horizontal con título y subtítulo; devuelve (figura, ejes)."""
    fig, axes = plt.subplots(figsize=PAGE, **subplots)
    fig.suptitle(title, x=0.06, y=0.97, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.06, 0.918, subtitle, color=MUTED, fontsize=9, va="center")
    return fig, axes


def _table(ax, frame, widths=None):
    """Dibuja `frame` (ya formateado como texto) como tabla que llena `ax`; `widths` son anchos relativos."""
    ax.axis("off")
    widths = None if widths is None else np.array(widths) / sum(widths)
    table = ax.table(cellText=frame.to_numpy(), colLabels=frame.columns, cellLoc="center", colWidths=widths, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_text_props(fontweight="bold")


def _p_value(p):
    """Valor p como texto: "< 0.001" o con tres decimales."""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def summary_page(lines):
    """Página 1: hallazgos principales en texto (`lines` = lista de (título, párrafo))."""
    fig, ax = _page("BTCUSDT horario: análisis exploratorio del entrenamiento",
                    "Todas las cifras usan solo datos de entrenamiento; el fold 1 (2017-08 a 2021-12) justifica el estado.")
    ax.axis("off")
    y = 0.97
    for title, text in lines:
        ax.text(0, y, title, fontsize=11, fontweight="bold", va="top", transform=ax.transAxes)
        ax.text(0, y - 0.045, text, fontsize=10, va="top", transform=ax.transAxes, wrap=True, linespacing=1.45)
        y -= 0.075 + 0.041 * (text.count("\n") + 1)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.88, bottom=0.06)
    return fig


def regimes_page(close, regimes):
    """Página 2: precio diario (log) con los tramos sombreados y su tabla de características."""
    fig, (ax, ax_table) = _page("Regímenes del entrenamiento",
                                f"Train del fold 4 (2017-08 a 2024-12). Giros de {evaluate.REGIME_THRESHOLD:.0%}, "
                                "identificados a posteriori: describen los datos y no entran al estado.",
                                nrows=2, height_ratios=[1.15, 1])
    for number, row in regimes.iterrows():
        ax.axvspan(row["start"], row["end"], color=UP if row["direction"] == "alza" else DOWN, lw=0)
        middle = row["start"] + (row["end"] - row["start"]) / 2
        ax.text(middle, 0.97, str(number), transform=ax.get_xaxis_transform(), ha="center", va="top", fontweight="bold")
    ax.plot(close.index, close, color=INK, lw=1.1)
    ax.set(yscale="log", title="Cierre diario (escala logarítmica)", ylabel="USDT", xlabel="Fecha (UTC)")
    ax.margins(x=0)
    ax.legend(handles=[Line2D([], [], color=INK, label="BTCUSDT"), Patch(facecolor=UP, edgecolor=MUTED, label="alza"),
                       Patch(facecolor=DOWN, edgecolor=MUTED, label="baja")],
              loc="lower right", bbox_to_anchor=(1, 1), ncols=3)
    _table(ax_table, pd.DataFrame({
        "#": regimes.index, "Inicio": regimes["start"].dt.date, "Fin": regimes["end"].dt.date, "Días": regimes["days"],
        "Dirección": regimes["direction"], "Rendimiento": regimes["total_return"].map("{:+.0%}".format),
        "Tend. anual log": regimes["annual_trend_log"].map("{:+.2f}".format),
        "Volatilidad anual": regimes["annual_volatility"].map("{:.0%}".format),
        "Nivel de vol.": regimes["volatility_level"], "Drawdown máx.": regimes["max_drawdown"].map("{:.0%}".format),
        "Tramo": regimes["status"]}).astype(str),
        widths=[0.04, 0.1, 0.1, 0.06, 0.09, 0.1, 0.15, 0.13, 0.1, 0.12, 0.1])
    fig.subplots_adjust(left=0.07, right=0.96, top=0.87, bottom=0.05, hspace=0.33)
    return fig


def distribution_page(hourly, daily):
    """Página 3: histograma (densidad log) y cuantiles contra la normal, de rendimientos horarios y diarios."""
    fig, axes = _page("Distribución de rendimientos", "Train del fold 1 (2017-08 a 2021-12). Rendimientos logarítmicos.",
                      nrows=2, ncols=2)
    for row, (name, r) in zip(axes, {"horarios": hourly, "diarios": daily}.items()):
        info = evaluate.distribution_stats(r)
        z = (r - r.mean()) / r.std()
        grid = np.linspace(-10, 10, 400)
        row[0].hist(z.clip(-10, 10), bins=120, density=True, color=BLUE, alpha=0.75, label="observado")
        row[0].plot(grid, stats.norm.pdf(grid), color=ORANGE, lw=2, label="normal")
        row[0].set(yscale="log", ylim=(1e-5, 2), xlabel="Desviaciones estándar (recortado a ±10)", ylabel="Densidad (log)",
                   title=f"{name.capitalize()}: asimetría {info['skew']:.2f}, curtosis exc. {info['excess_kurtosis']:.1f}")
        row[0].legend()
        theoretical, ordered = stats.probplot(z, fit=False)
        row[1].plot(theoretical, ordered, ".", color=BLUE, ms=3, label="observado")
        row[1].axline((0, 0), slope=1, color=ORANGE, lw=1.5, label="normal")
        row[1].set(xlabel="Cuantil normal", ylabel="Cuantil observado",
                   title=f"Cuantiles {name}: Jarque–Bera p {_p_value(info['jarque_bera_p'])}")
        row[1].legend()
    fig.subplots_adjust(left=0.07, right=0.96, top=0.86, bottom=0.07, hspace=0.36, wspace=0.2)
    return fig


def autocorrelation_page(hourly):
    """Página 4: ACF y PACF de rendimientos, y ACF de absolutos y cuadrados (agrupamiento de volatilidad)."""
    fig, axes = _page("Autocorrelación de rendimientos horarios",
                      "Train del fold 1. Banda: ±1.96/√n bajo ruido blanco (estrecha porque n es grande; con volatilidad "
                      "agrupada subestima la incertidumbre).", nrows=2, ncols=2)
    band = 1.96 / np.sqrt(len(hourly))
    lags = np.arange(1, ACF_LAGS + 1)
    panels = [("ACF de rendimientos", acf(hourly, nlags=ACF_LAGS), (-0.08, 0.08)),
              ("PACF de rendimientos", pacf(hourly, nlags=ACF_LAGS), (-0.08, 0.08)),
              ("ACF de rendimientos absolutos", acf(hourly.abs(), nlags=ACF_LAGS), (-0.05, 0.5)),
              ("ACF de rendimientos al cuadrado", acf(hourly ** 2, nlags=ACF_LAGS), (-0.05, 0.5))]
    for ax, (title, values, limits) in zip(axes.ravel(), panels):
        ax.axhspan(-band, band, color=GRID, lw=0, label="banda de 95 %")
        ax.vlines(lags, 0, values[1:], color=BLUE, lw=2, label="autocorrelación")
        ax.axhline(0, color="#c2c2c2", lw=0.8)
        ax.set(title=f"{title} (rezago 1 = {values[1]:.3f})", xlabel="Rezago (horas)", ylabel="Correlación", ylim=limits)
        ax.legend(loc="upper right")
    fig.subplots_adjust(left=0.07, right=0.96, top=0.86, bottom=0.07, hspace=0.36, wspace=0.2)
    return fig


def stationarity_page(table, ljung_box):
    """Página 5: ADF y KPSS de log-precio, rendimientos y features, y Ljung–Box de los rendimientos."""
    fig, (ax, ax_lb) = _page("Estacionariedad y Ljung–Box",
                             "Train del fold 1. ADF: H0 = raíz unitaria. KPSS: H0 = estacionaria en nivel (p acotado a "
                             "[0.01, 0.10]). Mixta = ambas rechazan.", nrows=2, height_ratios=[3.6, 1])
    _table(ax, pd.DataFrame({
        "Serie": table.index, "ADF": table["adf_stat"].map("{:.2f}".format), "ADF p": table["adf_p"].map(_p_value),
        "KPSS": table["kpss_stat"].map("{:.2f}".format), "KPSS p": table["kpss_p"].map("{:.3f}".format),
        "Veredicto (5 %)": table["verdict"]}).astype(str))
    _table(ax_lb, pd.DataFrame({
        "Serie": ljung_box["series"], "Rezagos": ljung_box["lags"], "Autocorrelación de rezago 1": ljung_box["acf_1"].map("{:.3f}".format),
        "Ljung–Box": ljung_box["statistic"].map("{:,.0f}".format),
        "p": ljung_box["p_value"].map(_p_value)}).astype(str))
    fig.subplots_adjust(left=0.07, right=0.95, top=0.89, bottom=0.03, hspace=0.08)
    return fig


def features_page(analyses):
    """Página 6: IM de cada feature con el objetivo contra el azar, y Spearman con el objetivo y con su tamaño."""
    first = analyses[1]
    order = first.sort_values("mi").index
    y = np.arange(len(order))
    fig, (ax_mi, ax_rho) = _page("Features contra el rendimiento a 24 h",
                                 f"Muestra sin traslape (una barra cada {evaluate.SAMPLE_STEP}). El train de cada fold contiene al "
                                 "del anterior: los folds no son muestras independientes.", ncols=2, sharey=True)
    later = np.array([analyses[k].loc[order, "mi"] for k in analyses])
    ax_mi.hlines(y, later.min(axis=0), later.max(axis=0), color="#c2c2c2", lw=3, label="rango entre los 4 folds")
    ax_mi.plot(first.loc[order, "mi"], y, "o", color=BLUE, ms=7, label="fold 1")
    ax_mi.axvline(first["mi_null_max_95"].iloc[0], color=ORANGE, ls="--", lw=1.5, label="azar, fold 1 (máx. de 22, percentil 95)")
    ax_mi.set(title="Información mutua", xlabel="Nats", yticks=y, yticklabels=order)
    ax_mi.legend(loc="upper center", bbox_to_anchor=(0.4, -0.075), fontsize=8)
    ax_rho.axvline(0, color="#c2c2c2", lw=0.8)
    ax_rho.plot(first.loc[order, "spearman"], y, "o", color=BLUE, ms=7, label="con el rendimiento (dirección)")
    ax_rho.plot(first.loc[order, "spearman_abs_target"], y, "D", color=ORANGE, ms=6, label="con su valor absoluto (tamaño)")
    ax_rho.axvspan(*(np.array([-1.96, 1.96]) / np.sqrt(first["n_samples"].iloc[0])), color=GRID, lw=0, zorder=0, label="banda de 95 % sin relación")
    ax_rho.set(title="Correlación de Spearman, fold 1", xlabel="Correlación")
    ax_rho.legend(loc="upper center", bbox_to_anchor=(0.5, -0.075), fontsize=8)
    fig.subplots_adjust(left=0.16, right=0.96, top=0.86, bottom=0.16, wspace=0.08)
    return fig


def robustness_page(robustness):
    """Página 7: en cuántos de los 25 desfases de la malla sale significativa cada feature (fold 1 y mínimo entre folds)."""
    first = robustness[1]
    n = int(first["n_offsets"].iloc[0])
    order = first.sort_values(["mi_above_null", "spearman_abs_sig"]).index
    y = np.arange(len(order))
    fig, (ax_mi, ax_dir, ax_size) = _page(
        "Robustez a la fase de la malla de muestreo",
        f"En cuántos de los {n} desfases (malla cada {evaluate.SAMPLE_STEP} h de reloj) sale significativa cada feature. "
        "Barra: fold 1; punto: el fold con menos desfases.", ncols=3, sharey=True)
    panels = ((ax_mi, "mi_above_null", "IM sobre el azar (máx. de 22)", BLUE),
              (ax_dir, "spearman_sig", "Dirección: Spearman p < 0.05", ORANGE),
              (ax_size, "spearman_abs_sig", "Tamaño: Spearman con |y|, p < 0.05", AQUA))
    for ax, column, title, color in panels:
        worst = np.min([robustness[k].loc[order, column] for k in robustness], axis=0)
        ax.barh(y, first.loc[order, column], color=color, height=0.65)
        ax.plot(worst, y, "o", color=INK, ms=4)
        if column != "mi_above_null":  # el umbral de la IM ya corrige por las 22 features; el 5 % no aplica
            ax.axvline(0.05 * n, color=MUTED, ls=":", lw=1)  # lo que se espera por azar con p < 0.05
        ax.set(title=title, xlim=(0, n), xlabel="Desfases", yticks=y, yticklabels=order)
    ax_dir.text(0.05 * n, len(order) - 0.3, " 5 % (azar)", color=MUTED, fontsize=7.5, va="bottom")
    fig.subplots_adjust(left=0.16, right=0.97, top=0.86, bottom=0.08, wspace=0.1)
    return fig


def _summary_lines(regimes, hourly_info, ljung_box, stationarity, analyses, robustness):
    """Texto de la página 1; cada cifra sale de las tablas calculadas."""
    first, last = analyses[1], regimes[max(regimes)]
    r1 = regimes[1]
    share_up = r1.loc[r1["direction"] == "alza", "days"].sum() / r1["days"].sum()
    above = first.index[first["mi"] > first["mi_null_max_95"]]
    always = [f for f in above if all(t.loc[f, "mi"] > t.loc[f, "mi_null_max_95"] for t in analyses.values())]
    acf_1 = ljung_box.drop_duplicates("series").set_index("series")["acf_1"]
    features = stationarity.loc[data.FEATURES]
    directional = int((first["spearman_p"] < 0.05).sum())
    sized = int((first["spearman_abs_target_p"] < 0.05).sum())
    return [
        ("Regímenes", f"{len(r1)} tramos en el train del fold 1 y {len(last)} en el del fold 4. En el fold 1, {share_up:.0%} de los días es "
                      f"de alza.\nVolatilidad anual por tramo: de {last['annual_volatility'].min():.0%} a "
                      f"{last['annual_volatility'].max():.0%}. Peor drawdown de un tramo: {last['max_drawdown'].min():.0%}."),
        ("Distribución", f"Colas pesadas: curtosis en exceso de {hourly_info['excess_kurtosis']:.1f} en rendimientos horarios y asimetría de "
                         f"{hourly_info['skew']:.2f}.\nJarque–Bera rechaza la normalidad."),
        ("Autocorrelación", f"Rezago 1 de los rendimientos: {acf_1['rendimientos']:.3f}; de los absolutos: {acf_1['absolutos']:.3f}; de los "
                            f"cuadrados: {acf_1['cuadrados']:.3f}.\nLa volatilidad se agrupa; los rendimientos casi no se predicen a sí mismos."),
        ("Estacionariedad", f"Log-precio: serie {stationarity.loc['log_precio', 'verdict']}. Rendimientos horarios: serie {stationarity.loc['ret_horario', 'verdict']}.\n"
                            f"De las {len(features)} features, ADF rechaza la raíz unitaria en {(features['adf_p'] < 0.05).sum()} y KPSS rechaza la "
                            f"estacionariedad en {(features['kpss_p'] < 0.05).sum()}."),
        ("Features contra el rendimiento a 24 h",
         f"{len(above)} de {len(first)} superan el azar en el fold 1; {len(always)} lo superan en los 4 folds:\n{', '.join(always) or 'ninguna'}.\n"
         f"Con p < 0.05 (sin corregir por 22 pruebas), {directional} se relaciona(n) con la dirección y {sized} con el tamaño del movimiento.\n"
         f"Spearman absoluto máximo: {first['spearman'].abs().max():.3f} con la dirección y {first['spearman_abs_target'].abs().max():.3f} con el tamaño.\n"
         f"Con todas las horas (objetivos traslapados) la IM máxima sube de {first['mi'].max():.3f} a {first['mi_all_hours'].max():.3f}: está inflada."),
        ("Robustez a la fase de la malla", _robustness_text(robustness)),
    ]


def _robustness_text(robustness):
    """Párrafo del resumen sobre los 25 desfases; cada cifra sale de `offset_robustness`."""
    first = robustness[1]
    n = int(first["n_offsets"].iloc[0])
    half = n / 2
    stable = [f for f in first.index if all(r.loc[f, "mi_above_null"] > half for r in robustness.values())]
    best_dir = first["spearman_sig"].idxmax()
    sized = int((first["spearman_abs_sig"] > half).sum())
    return (f"Superan el azar en más de la mitad de los {n} desfases en los 4 folds: {', '.join(stable) or 'ninguna'}.\n"
            f"Fold 1: {sized} features se relacionan con el tamaño en más de la mitad de los desfases; con la dirección, la más "
            f"frecuente es {best_dir} ({first.loc[best_dir, 'spearman_sig']} de {n};\npor azar se esperarían ≈ {0.05 * n:.1f}).")


def write_eda_pdf(path, figures_dir, ohlcv, features, folds, regimes, analyses, robustness):
    """Escribe `btcusdt_eda.pdf` (7 páginas) y cada figura como PNG en `figures_dir`.

    Distribución, autocorrelación y estacionariedad usan el train del fold 1; los regímenes, el del último fold.
    """
    train_ohlcv, train_features = evaluate.train_frames(ohlcv, features, folds[1])
    hourly = evaluate.hourly_returns(train_ohlcv)
    daily = np.log(evaluate.daily_close(train_ohlcv)).diff().dropna()
    last = max(folds)
    last_close = evaluate.daily_close(evaluate.train_slice(ohlcv, folds[last]))
    stationarity = evaluate.stationarity_table(pd.concat(
        [np.log(train_ohlcv["close"]).rename("log_precio"), hourly.rename("ret_horario"), train_features], axis=1))
    ljung_box = evaluate.ljung_box_table(hourly)
    with plt.rc_context(STYLE), PdfPages(path, metadata={"Title": "BTCUSDT: EDA del entrenamiento", "CreationDate": None}) as pdf:
        pages = {
            "eda_resumen": summary_page(_summary_lines(regimes, evaluate.distribution_stats(hourly), ljung_box, stationarity, analyses, robustness)),
            "eda_regimenes": regimes_page(last_close, regimes[last]),
            "eda_distribucion": distribution_page(hourly, daily),
            "eda_autocorrelacion": autocorrelation_page(hourly),
            "eda_estacionariedad": stationarity_page(stationarity, ljung_box),
            "eda_features": features_page(analyses),
            "eda_robustez": robustness_page(robustness),
        }
        for name, fig in pages.items():
            pdf.savefig(fig)
            if name != "eda_resumen":
                fig.savefig(figures_dir / f"{name}.png", dpi=150, metadata={"Software": None})
            plt.close(fig)


# ====================================================================== Parte 05: entrenamiento

SEED_COLORS = (BLUE, ORANGE, AQUA, MUTED)


def plot_training_curves(curves, runs, config, chosen_timesteps, fold, path, window=25):
    """`training_curves_fold_{k}.png`: arriba, recompensa por episodio de entrenamiento (= ln del capital del episodio,
    media móvil de `window` episodios) contra pasos, una línea por semilla de la configuración elegida; abajo, el
    crecimiento log en validación de cada semilla en los puntos de control. Se marcan el fin de la exploración ε y el
    punto de control elegido."""
    total = max(int(runs["timesteps"].max()), 1)
    with plt.rc_context(STYLE):
        fig, (ax, ax_val) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        for i, (seed, curve) in enumerate(sorted(curves.items())):
            color = SEED_COLORS[i % len(SEED_COLORS)]
            smooth = curve["episode_reward"].rolling(window, min_periods=1).mean()
            ax.plot(curve["timesteps"], smooth, color=color, lw=1.1, label=f"semilla {seed}")
            val = runs[runs["seed"] == seed].sort_values("timesteps")
            ax_val.plot(val["timesteps"], val["val_log_growth"], "o-", color=color, lw=1.1, ms=4)
        end_eps = config["exploration_fraction"] * total
        for axis in (ax, ax_val):
            axis.axhline(0, color=INK, lw=0.6)
            axis.axvline(end_eps, color=MUTED, lw=0.8, ls=":")
            axis.axvline(chosen_timesteps, color=INK, lw=0.9, ls="--")
        ax.annotate("fin de la exploración ε", (end_eps, ax.get_ylim()[1]), xytext=(4, -12), textcoords="offset points",
                    fontsize=7, color=MUTED)
        ax.annotate("punto de control elegido", (chosen_timesteps, ax.get_ylim()[1]), xytext=(4, -24),
                    textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(f"Fold {fold}: recompensa por episodio de entrenamiento (media móvil de {window} episodios)")
        ax.set_ylabel("ln(capital del episodio)")
        ax.legend(loc="upper left", fontsize=8)
        ax_val.set_title("Crecimiento log en validación (recorrido completo) en cada punto de control")
        ax_val.set_ylabel("ln(capital final)")
        ax_val.set_xlabel("pasos de entrenamiento (decisiones)")
        fig.tight_layout()
        fig.savefig(path, dpi=150, metadata={"Software": None})


# ====================================================================== Parte 06: evaluación en prueba

STRATEGY_STYLE = {
    "agente": dict(color=BLUE, lw=1.8, label="Agente"),
    "btc_buy_hold": dict(color=ORANGE, lw=1.3, label="Comprar y mantener BTC"),
    "sp500": dict(color=AQUA, lw=1.3, label="S&P 500"),
    "siempre_largo": dict(color=MUTED, lw=0.9, ls="--", label="Siempre largo (24 h)"),
    "aleatorio": dict(color="#b0b0b0", lw=0.9, ls=":", label="Aleatorio"),
    "siempre_plano": dict(color="#b0b0b0", lw=0.9, ls="-.", label="Siempre plano"),
}


def plot_equity_curves(daily_curves, fold, test_dates, path):
    """`equity_curves_test_fold_{k}.png`: capital diario de cada estrategia en la prueba del fold (inicial = 1), con
    ambos benchmarks; escala log para comparar rendimientos relativos."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, curve in daily_curves.items():
            ax.plot(curve.index, curve.to_numpy(), **STRATEGY_STYLE.get(name, dict(label=name)))
        ax.axhline(1.0, color=INK, lw=0.6)
        ax.set_yscale("log")
        plain = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}")
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
        ax.set_title(f"Fold {fold}: capital en prueba ({test_dates[0]} a {test_dates[1]}), con costos de 0.125 % por lado")
        ax.set_ylabel("capital (inicial = 1, escala log)")
        ax.set_xlabel("fecha (UTC)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=len(daily_curves), fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def plot_drawdown_period(ohlcv, trades, equity, window, fold, path, pad_days=3, max_markers=60):
    """`max_drawdown_fold_{k}.png` (§3.9.2): precio y acciones del agente durante su mayor drawdown en prueba, desde unos
    días antes del máximo previo hasta la recuperación (o el fondo, si no se recupera).

    Cada operación se sombrea mientras está abierta (verde = largo, naranja = corto); con pocas operaciones también se
    marcan entradas (▲/▼) y salidas (×). Líneas verticales: máximo previo y fondo del capital. Abajo, el capital a mercado.
    """
    start = window["peak"] - pd.Timedelta(days=pad_days)
    end = (window["recovery"] or window["trough"]) + pd.Timedelta(days=pad_days)
    price = ohlcv.loc[start:end, "close"]
    inside = trades[(trades["exit_time"] >= start) & (trades["entry_time"] <= end)]
    with plt.rc_context(STYLE):
        fig, (ax, ax_eq) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        for _, trade in inside.iterrows():
            ax.axvspan(trade["entry_time"], trade["exit_time"], color=AQUA if trade["side"] == "largo" else ORANGE,
                       alpha=0.18, lw=0)
        ax.plot(price.index, price.to_numpy(), color=INK, lw=0.8)
        if len(inside) <= max_markers:
            for side, marker, color in (("largo", "^", AQUA), ("corto", "v", ORANGE)):
                entries = inside[inside["side"] == side]
                ax.scatter(entries["entry_time"], entries["entry_price"], marker=marker, color=color, s=30, zorder=3)
            ax.scatter(inside["exit_time"], inside["exit_price"], marker="x", color=BLUE, s=22, zorder=3)
        for axis in (ax, ax_eq):
            for moment in (window["peak"], window["trough"]):
                axis.axvline(moment, color=MUTED, lw=0.8, ls="--")
        handles = [Line2D([], [], color=INK, lw=0.8, label="cierre BTCUSDT"),
                   Patch(color=AQUA, alpha=0.35, label="en largo"), Patch(color=ORANGE, alpha=0.35, label="en corto"),
                   Line2D([], [], color=MUTED, ls="--", lw=0.8, label="máximo previo y fondo del capital")]
        ax.legend(handles=handles, loc="upper left", fontsize=7, ncol=4)
        n_long, n_short = int((inside["side"] == "largo").sum()), int((inside["side"] == "corto").sum())
        ax.set_title(f"Fold {fold}: mayor drawdown en prueba ({window['max_drawdown']:.1%}); "
                     f"{len(inside)} operaciones ({n_long} largos, {n_short} cortos)")
        ax.set_ylabel("USDT")
        curve = equity.loc[start:end]
        ax_eq.plot(curve.index, curve.to_numpy(), color=BLUE, lw=1.0)
        ax_eq.set_ylabel("capital del agente")
        ax_eq.set_xlabel("fecha (UTC)")
        ax_eq.set_title("Capital a mercado")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ====================================================================== Simulación visual del agente (GC)

SIMULATION_DAYS = 14
REASON_LABEL = {"horizonte": "horizonte", "stop": "stop-loss", "take_profit": "take-profit", "fin_datos": "fin de datos"}
REASON_COLOR = {"horizonte": MUTED, "stop": "#d63031", "take_profit": AQUA, "fin_datos": MUTED}


def busiest_window(trade_logs, days=SIMULATION_DAYS):
    """Criterio fijo del tramo de la simulación: la ventana de `days` días (empezando a medianoche UTC) con más
    operaciones del agente entre todos los folds de prueba; empates → el fold y la fecha más tempranos.
    Devuelve (fold, inicio, fin)."""
    best = None
    for k in sorted(trade_logs):
        log = trade_logs[k]
        if not len(log):
            continue
        entries = pd.to_datetime(log["entry_time"], utc=True)
        for start in pd.date_range(entries.min().floor("D"), entries.max().floor("D"), freq="D"):
            n = int(((entries >= start) & (entries < start + pd.Timedelta(days=days))).sum())
            if best is None or n > best[0]:
                best = (n, k, start)
    _, k, start = best
    return k, start, start + pd.Timedelta(days=days)


def write_agent_simulation_gif(ohlcv, trades, equity, btc_equity, fold, start, end, path, step_hours=2, fps=8):
    """`agent_simulation.gif`: el agente operando un tramo de prueba, barra por barra, como el pingüino de FrozenLake.

    Arriba, el precio hasta la hora actual, la posición abierta sombreada (verde = largo, naranja = corto) con sus
    niveles de stop y take-profit, las entradas (▲/▼) y las salidas con su razón. Abajo, el capital del agente contra
    comprar y mantener BTC, ambos en 1 al inicio del tramo. Solo lee la bitácora y las curvas de capital.
    """
    price = ohlcv.loc[start:end, "close"]
    trades = trades.copy()
    for col in ("decision_time", "entry_time", "exit_time"):
        trades[col] = pd.to_datetime(trades[col], utc=True)
    trades = trades[(trades["exit_time"] >= start) & (trades["entry_time"] <= end)]
    agent = equity.loc[start:end]
    agent = agent / agent.iloc[0]
    btc = btc_equity.loc[start:end]
    btc = btc / btc.iloc[0]
    times = price.index[::step_hours]
    pad = (price.max() - price.min()) * 0.08

    with plt.rc_context(STYLE):
        fig, (ax, ax_eq) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
        fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.08, hspace=0.3)

        def draw(i):
            now = times[i]
            ax.clear()
            ax_eq.clear()
            seen = price.loc[:now]
            ax.plot(seen.index, seen.to_numpy(), color=INK, lw=0.9)
            ax.set_xlim(price.index[0], price.index[-1])
            ax.set_ylim(price.min() - pad, price.max() + pad)
            status = "plano: el agente decide al cierre de cada barra (esperar, largo o corto)"
            for _, trade in trades[trades["entry_time"] <= now].iterrows():
                long_ = trade["side"] == "largo"
                color = AQUA if long_ else ORANGE
                closed = trade["exit_time"] <= now
                stop_at = trade["exit_time"] if closed else now
                ax.axvspan(trade["entry_time"], stop_at, color=color, alpha=0.15, lw=0)
                ax.scatter([trade["entry_time"]], [trade["entry_price"]], marker="^" if long_ else "v", color=color,
                           s=45, zorder=4)
                if closed:
                    reason = trade["exit_reason"]
                    ax.scatter([trade["exit_time"]], [trade["exit_price"]], marker="X", color=REASON_COLOR[reason],
                               s=40, zorder=4)
                else:
                    for level, style in ((trade["stop_level"], "#d63031"), (trade["take_level"], AQUA)):
                        ax.hlines(level, trade["entry_time"], now, colors=style, linestyles="--", lw=0.9)
                    status = (f"{trade['side']} desde {trade['entry_time']:%d-%b %H:%M} a {trade['entry_price']:,.0f}; "
                              f"stop {trade['stop_level']:,.0f} · take-profit {trade['take_level']:,.0f}")
            last_exit = trades[trades["exit_time"] <= now].tail(1)
            if len(last_exit):
                t = last_exit.iloc[0]
                net = t["net_pnl"] / t["equity_before"]
                status += (f"   |   última salida: {REASON_LABEL[t['exit_reason']]} "
                           f"({t['exit_time']:%d-%b %H:%M}), {net:+.2%} neto")
            ax.set_title(f"Fold {fold} · prueba · {now:%Y-%m-%d %H:%M} UTC", loc="left")
            ax.text(0.0, 1.16, "Agente DQN operando BTCUSDT (costos de 0.125 % por lado)", transform=ax.transAxes,
                    fontsize=13, fontweight="bold")
            ax.text(0.0, -0.09, status, transform=ax.transAxes, fontsize=8, color=INK)
            ax.set_ylabel("USDT")
            handles = [Line2D([], [], marker="^", ls="", color=AQUA, label="entra largo"),
                       Line2D([], [], marker="v", ls="", color=ORANGE, label="entra corto"),
                       Line2D([], [], marker="X", ls="", color="#d63031", label="sale por stop"),
                       Line2D([], [], marker="X", ls="", color=AQUA, label="sale por take-profit"),
                       Line2D([], [], marker="X", ls="", color=MUTED, label="sale por horizonte (24 h)"),
                       Patch(color=MUTED, alpha=0.25, label="en posición")]
            ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7, ncol=3)

            a, b = agent.loc[:now], btc.loc[:now]
            ax_eq.plot(a.index, a.to_numpy(), color=BLUE, lw=1.4, label=f"agente {a.iloc[-1] - 1:+.1%}")
            ax_eq.plot(b.index, b.to_numpy(), color=ORANGE, lw=1.0, label=f"comprar y mantener BTC {b.iloc[-1] - 1:+.1%}")
            ax_eq.axhline(1, color=INK, lw=0.5)
            lo, hi = min(agent.min(), btc.min()), max(agent.max(), btc.max())
            ax_eq.set_ylim(lo - 0.01, hi + 0.01)
            ax_eq.set_ylabel("capital (inicio = 1)")
            ax_eq.legend(loc="lower left", fontsize=8, ncol=2)
            ax_eq.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d-%b"))
            return []

        animation = FuncAnimation(fig, draw, frames=len(times), blit=False)
        animation.save(path, writer=PillowWriter(fps=fps), dpi=80)
        plt.close(fig)
