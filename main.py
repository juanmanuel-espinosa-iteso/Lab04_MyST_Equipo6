"""Punto de entrada único del pipeline del Lab04: python main.py

Orquesta, con semillas fijas, todas las etapas que producen los artefactos de `artifacts/` (§3.8 del enunciado):
datos y features → EDA → entrenamiento por fold → evaluación en prueba → sensibilidad. La lógica vive en `code/`;
aquí solo se definen los parámetros globales (semilla, rutas y folds) y el orden de las etapas.
"""

import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# `code/` (obligatoria, §2.2) se llama igual que un módulo de la biblioteca estándar: se asegura el paquete local.
sys.path.insert(0, str(ROOT))
if not str(getattr(sys.modules.get("code"), "__file__", "")).startswith(str(ROOT)):
    sys.modules.pop("code", None)

from code import data, env, evaluate, plots, reward, train  # noqa: E402

SEED = 42

DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "docs" / "figuras"
BTC_CSV = DATA / "Binance_BTCUSDT_1h.csv"
SP500_CSV = DATA / "sp500_daily.csv"  # solo benchmark (§3.7)
MODELS = ARTIFACTS / "models"  # no se versionan; se regeneran
# Procesos para la búsqueda (uno por corrida, torch con 1 hilo); LAB04_PROCESSES lo cambia
PROCESSES = int(os.environ.get("LAB04_PROCESSES", max(1, min(10, (os.cpu_count() or 2) - 1))))
# LAB04_REUSE_MODELS=1 reusa la búsqueda ya entrenada en artifacts/models/ si se hizo con los mismos parámetros (para
# iterar sin volver a entrenar ≈ 40 min); un clon limpio no tiene modelos y siempre entrena.
REUSE_MODELS = os.environ.get("LAB04_REUSE_MODELS") == "1"
SEARCH_CACHE = MODELS / "search.pkl"

# Walk-forward de ventana expansiva (§3.6): fechas inclusivas, horas en UTC.
FOLDS = {
    1: {"train": ("2017-08-17", "2021-12-31"), "validation": ("2022-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    2: {"train": ("2017-08-17", "2022-12-31"), "validation": ("2023-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    3: {"train": ("2017-08-17", "2023-12-31"), "validation": ("2024-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
    4: {"train": ("2017-08-17", "2024-12-31"), "validation": ("2025-01-01", "2025-12-31"), "test": ("2026-01-01", "2026-04-26")},
}


def _json_safe(obj):
    """Copia de `obj` apta para JSON estricto: NaN e infinitos pasan a `null` (JSON no tiene NaN)."""
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(obj, path):
    """Escribe `obj` como JSON válido y legible (UTF-8, sangría de 2); NaN e infinitos se escriben como `null`."""
    path.write_text(json.dumps(_json_safe(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run_data_stage():
    """Parte 01: carga y valida el CSV, calcula las features causales y describe los folds.

    Escribe `artifacts/data_report.json`, `artifacts/folds.json` y los estadísticos del normalizador de cada fold
    (`artifacts/normalizer_fold_{k}.csv`, ajustados solo con su train). Devuelve (ohlcv, features) sin normalizar;
    cada etapa posterior arma su fold con `data.build_fold_dataset`, que reproduce esos mismos estadísticos.
    """
    ohlcv, report = data.load_ohlcv(BTC_CSV)
    features = data.compute_features(ohlcv)
    write_json(report, ARTIFACTS / "data_report.json")
    write_json(data.describe_folds(ohlcv, features, FOLDS), ARTIFACTS / "folds.json")
    for k, dates in FOLDS.items():
        train = data.split_fold(features, dates)["train"]
        data.Normalizer.fit(train).to_frame(train).to_csv(ARTIFACTS / f"normalizer_fold_{k}.csv")
    print(f"[datos] {report['bars']:,} barras ({report['first_bar'][:16]} a {report['last_bar'][:16]}); "
          f"{report['duplicates_removed']} duplicados quitados; {report['missing_hours']:,} horas faltantes; "
          f"{len(data.FEATURES)} features en {len(data.FEATURE_FAMILIES)} familias")
    return ohlcv, features


def run_eda_stage(ohlcv, features):
    """Parte 02: regímenes, análisis exploratorio y relación de cada feature con el objetivo, solo en entrenamiento.

    Escribe `artifacts/regimes.csv`, `artifacts/feature_analysis_fold_{k}.csv`, `artifacts/feature_robustness_fold_{k}.csv`
    (robustez a la fase de la malla de muestreo) y `artifacts/btcusdt_eda.pdf` (y sus figuras
    en `docs/figuras/`). El estado se justifica con el train del fold 1; los demás folds solo revisan la estabilidad,
    porque su train incluye la validación y la prueba del fold 1.
    """
    regimes, analyses, robustness = {}, {}, {}
    for k, dates in FOLDS.items():
        train_ohlcv, train_features = evaluate.train_frames(ohlcv, features, dates)
        regimes[k] = evaluate.regime_table(evaluate.daily_close(train_ohlcv), evaluate.REGIME_THRESHOLD)
        analyses[k] = evaluate.feature_analysis(train_ohlcv, train_features, SEED)
        analyses[k].to_csv(ARTIFACTS / f"feature_analysis_fold_{k}.csv", float_format="%.6g")
        robustness[k] = evaluate.offset_robustness(train_ohlcv, train_features, SEED)
        robustness[k].to_csv(ARTIFACTS / f"feature_robustness_fold_{k}.csv", float_format="%.6g")
    pd.concat(regimes, names=["fold"]).to_csv(ARTIFACTS / "regimes.csv", float_format="%.6g")
    plots.write_eda_pdf(ARTIFACTS / "btcusdt_eda.pdf", FIGURES, ohlcv, features, FOLDS, regimes, analyses, robustness)
    print(f"[eda] regímenes por fold: {[len(t) for t in regimes.values()]}; "
          f"features con IM sobre el azar por fold: {[int((t.mi > t.mi_null_max_95).sum()) for t in analyses.values()]}")


def run_training_stage(ohlcv, features):
    """Parte 05: check_env, líneas base y entrenamiento por fold y semilla con búsqueda de hiperparámetros en validación.

    Escribe `artifacts/baselines.csv`, `artifacts/validation_results_fold_{k}.csv` (todas las configuraciones ×
    semillas × puntos de control), `artifacts/training_curves_fold_{k}.png`, `artifacts/run_metadata.json` y los
    modelos en `artifacts/models/`. Elige con el criterio (A) de `train.select_config`, solo con validación, y devuelve
    {fold: política `obs -> acción`} del modelo elegido de cada fold, congelado para tocar la prueba una vez (parte 06).
    """
    from stable_baselines3.common.env_checker import check_env

    start = time.perf_counter()
    folds = {k: data.build_fold_dataset(ohlcv, features, k, dates) for k, dates in FOLDS.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for fold_data in folds.values():
            check_env(env.make_env(fold_data, "train", reward.REWARDS[reward.DEFAULT_REWARD]), warn=True)
    baselines = pd.concat({k: train.run_baselines(fold_data, SEED + k) for k, fold_data in folds.items()}, names=["fold"])
    baselines.to_csv(ARTIFACTS / "baselines.csv", float_format="%.6g")

    jobs = [job for k, dates in FOLDS.items() for job in train.search_jobs(k, dates, BTC_CSV, MODELS)]
    saved = train.load_search(SEARCH_CACHE, jobs) if REUSE_MODELS else None
    if saved is None:
        results, curves = train.hyperparameter_search(jobs, processes=PROCESSES)
        train.save_search(SEARCH_CACHE, jobs, results, curves)
    else:
        results, curves = saved
        print(f"[entrenamiento] LAB04_REUSE_MODELS=1: se reusan los {len(jobs)} modelos de {MODELS} (misma búsqueda)")
    policies, chosen = {}, {}
    for k in FOLDS:
        fold_results = results[results["fold"] == k]
        fold_results.to_csv(ARTIFACTS / f"validation_results_fold_{k}.csv", index=False, float_format="%.6g")
        config_id, timesteps, seed = train.select_config(fold_results)
        model = train.DQN.load(train.model_path(MODELS, k, config_id, seed, timesteps), device="cpu")
        policies[k] = train.as_policy(model)
        runs = fold_results[fold_results["config_id"] == config_id]
        candidate = runs[runs["timesteps"] == timesteps]
        chosen[k] = {"config_id": config_id, **train.GRID[config_id], "timesteps": timesteps, "seed": seed,
                     "val_log_growth_median": float(candidate["val_log_growth"].median()),
                     "val_log_growth_by_seed": dict(zip(candidate["seed"].astype(int), candidate["val_log_growth"]))}
        plots.plot_training_curves({s: curves[(k, config_id, s)] for s in train.SEEDS}, runs, train.GRID[config_id],
                                   timesteps, k, ARTIFACTS / f"training_curves_fold_{k}.png")

    import gymnasium
    import stable_baselines3
    import torch

    write_json({
        "python": platform.python_version(), "stable_baselines3": stable_baselines3.__version__,
        "gymnasium": gymnasium.__version__, "torch": torch.__version__, "platform": platform.platform(),
        "global_seed": SEED, "seeds": list(train.SEEDS), "algorithm": "DQN + SemiMarkovReplayBuffer (γ^τ)",
        "reward": reward.DEFAULT_REWARD, "total_timesteps": train.TOTAL_TIMESTEPS, "checkpoints": list(train.CHECKPOINTS),
        "search_space": {k: [list(v) if isinstance(v, tuple) else v for v in vals]
                         for k, vals in train.SEARCH_SPACE.items()},
        "dqn_fixed": train.DQN_FIXED, "selection": "mediana entre semillas de val_log_growth; semilla mediana",
        "chosen": chosen, "runs": len(jobs), "processes": PROCESSES, "reused_search": saved is not None,
        "training_minutes": round((time.perf_counter() - start) / 60, 1),
    }, ARTIFACTS / "run_metadata.json")
    done = "reusadas" if saved is not None else f"en {(time.perf_counter() - start) / 60:.1f} min"
    print(f"[entrenamiento] {len(jobs)} corridas {done}; elegidas: "
          + "; ".join(f"fold {k} cfg {c['config_id']} {c['timesteps'] // 1000}k pasos "
                      f"(val {c['val_log_growth_median']:+.3f})" for k, c in chosen.items()))
    return policies


def run_evaluation_stage(ohlcv, features, agent_policies=None):
    """Parte 06: evaluación única en prueba, bitácora, benchmarks y métricas agregadas entre folds.

    `agent_policies` (fold -> `obs -> acción`) viene de la parte 05 con los hiperparámetros ya congelados. En cada prueba
    se evalúan además las líneas base y los dos benchmarks con el mismo costo. Escribe `metrics_summary.csv` y
    `equity_curves_test_fold_{k}.png`; con agente, también `trade_log_fold_{k}.csv`, `max_drawdown_fold_{k}.png` y
    `trade_analysis.json` (§3.9).
    """
    agent_policies = agent_policies or {}
    sp500 = data.load_sp500(SP500_CSV)
    results, raw_test, btc_daily = {}, {}, {}
    for k, dates in FOLDS.items():
        fold = data.build_fold_dataset(ohlcv, features, k, dates)
        policies = evaluate.baseline_policies(n_actions=3, seed=SEED + k)
        if k in agent_policies:
            policies = {evaluate.AGENT: agent_policies[k], **policies}
        results[k] = evaluate.evaluate_test_fold(fold, policies, sp500, dates["test"], SEED + k, env.COST_PER_SIDE)
        raw_test[k], btc_daily[k] = fold.raw_features["test"], results[k]["btc_buy_hold"]["daily"]
        shown = [evaluate.AGENT, "siempre_largo", "aleatorio", *evaluate.BENCHMARKS]
        plots.plot_equity_curves({n: results[k][n]["daily"] for n in shown if n in results[k]}, k, dates["test"],
                                 ARTIFACTS / f"equity_curves_test_fold_{k}.png")
        if evaluate.AGENT in results[k]:
            log = results[k][evaluate.AGENT]["trades"]
            state = raw_test[k].reindex(log["decision_time"]).reset_index(drop=True).add_prefix("state_")
            pd.concat([log, state], axis=1).to_csv(ARTIFACTS / f"trade_log_fold_{k}.csv", index=False)

    table = evaluate.metrics_table(results)
    table.to_csv(ARTIFACTS / "metrics_summary.csv", float_format="%.6g")
    if agent_policies:
        analysis = evaluate.trade_analysis(results, raw_test, btc_daily)
        for k, window in analysis["drawdowns"].items():
            agent = results[k][evaluate.AGENT]
            plots.plot_drawdown_period(ohlcv, agent["trades"], agent["equity"], window, k,
                                       ARTIFACTS / f"max_drawdown_fold_{k}.png")
        write_json(json.loads(json.dumps({**analysis, "worst_trades": analysis["worst_trades"].to_dict("records")},
                                         default=str)), ARTIFACTS / "trade_analysis.json")
    agg = table.loc["media"]
    print(f"[evaluación] Sharpe medio en prueba: " + "; ".join(f"{n} {agg.loc[n, 'sharpe']:.2f}" for n in agg.index)
          + ("" if agent_policies else " (sin agente: falta la parte 05)"))
    return results


def run_sensitivity_stage():
    """Parte 07: resolución intrabarra, retraso de una barra, ablación y segundo algoritmo."""


def main():
    """Corre el pipeline completo de punta a punta con la semilla global."""
    np.random.seed(SEED)
    ARTIFACTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ohlcv, features = run_data_stage()
    run_eda_stage(ohlcv, features)
    agent_policies = run_training_stage(ohlcv, features)
    run_evaluation_stage(ohlcv, features, agent_policies)
    run_sensitivity_stage()
    print(f"Pipeline terminado. Artefactos en {ARTIFACTS}")


if __name__ == "__main__":
    main()
