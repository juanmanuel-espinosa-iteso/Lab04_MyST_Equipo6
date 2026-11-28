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


def run_search(jobs, cache, stage):
    """Corre las corridas en paralelo y guarda la búsqueda junto a los modelos; con `LAB04_REUSE_MODELS=1` la reusa si
    se hizo con los mismos parámetros. Devuelve (resultados, curvas, la búsqueda reusada o None)."""
    saved = train.load_search(cache, jobs) if REUSE_MODELS else None
    if saved is None:
        results, curves = train.hyperparameter_search(jobs, processes=PROCESSES)
        train.save_search(cache, jobs, results, curves)
        return results, curves, None
    print(f"[{stage}] LAB04_REUSE_MODELS=1: se reusan los {len(jobs)} modelos de {cache.parent} (misma búsqueda)")
    return (*saved, saved)


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
    results, curves, saved = run_search(jobs, SEARCH_CACHE, "entrenamiento")
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


def run_evaluation_stage(ohlcv, features, agent_policies=None, agent_env_params=None):
    """Parte 06: evaluación única en prueba, bitácora, benchmarks y métricas agregadas entre folds.

    `agent_policies` (fold -> `obs -> acción`) viene de la parte 05 con los hiperparámetros ya congelados. En cada prueba
    se evalúan además las líneas base y los dos benchmarks con el mismo costo. `agent_env_params` (fold -> dict) son los
    parámetros del entorno congelados en la validación (p. ej. horizonte y barreras); sin ellos, los de `code/env.py`. Escribe `metrics_summary.csv` y
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
        results[k] = evaluate.evaluate_test_fold(fold, policies, sp500, dates["test"], SEED + k, env.COST_PER_SIDE,
                                                 (agent_env_params or {}).get(k))
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


def run_sensitivity_stage(ohlcv, features, agent_policies=None, agent_env_params=None):
    """Parte 07 (JME): sensibilidad del resultado en prueba a los supuestos del simulador.

    Por fold y política: resolución intrabarra contraria (`tp_first`), una barra de retraso en los datos, latencia de 2 s
    como costo extra y +5/+10/+25 pb de slippage por lado, cada uno contra el escenario base (§3.3, §3.10, pregunta 6).
    Se corre el agente congelado de la parte 05 (`agent_policies`, fold -> `obs -> acción`) y, como referencia, las
    líneas base siempre largo y aleatorio (operan mucho más, así que muestran el efecto de cada supuesto con más
    operaciones). Escribe `artifacts/sensitivity.csv` con los resultados por fold y la media y la desviación entre folds
    de cada Δ. La ablación y el segundo algoritmo (GC) van aparte.
    """
    rows = []
    for k, dates in FOLDS.items():
        fold = data.build_fold_dataset(ohlcv, features, k, dates)
        makers = {name: (lambda name=name, k=k: evaluate.baseline_policies(3, SEED + k)[name])
                  for name in ("siempre_largo", "aleatorio")}
        if agent_policies and k in agent_policies:
            makers = {evaluate.AGENT: (lambda policy=agent_policies[k]: policy), **makers}
        for name, make_policy in makers.items():
            table = evaluate.run_sensitivity(fold, make_policy, dates["test"], SEED + k, env.COST_PER_SIDE,
                                             (agent_env_params or {}).get(k))
            rows.append(table.assign(fold=k, strategy=name).reset_index())
    per_fold = pd.concat(rows).set_index(["fold", "strategy", "scenario"])
    deltas = per_fold[["delta_total_return", "delta_sharpe"]].groupby(level=["strategy", "scenario"])
    summary = pd.concat({"media": deltas.mean(), "desv": deltas.std()}, names=["fold"])
    pd.concat([per_fold, summary]).to_csv(ARTIFACTS / "sensitivity.csv", float_format="%.6g")
    mean = summary.loc["media", "delta_total_return"]
    shown = mean.xs("tp_first", level="scenario")
    print("[sensibilidad] Δ rendimiento total medio con take-profit primero: "
          + "; ".join(f"{n} {v:+.4f}" for n, v in shown.items())
          + f"; operaciones ambiguas: {int(per_fold['n_ambiguous'].xs('base', level='scenario').sum())}")


def run_simulation_stage(ohlcv, results):
    """Simulación visual del agente (GC, petición del profesor): `artifacts/agent_simulation.gif`.

    El tramo se elige con un criterio fijo, las 2 semanas de prueba con más operaciones del agente entre todos los
    folds (`plots.busiest_window`), y la animación solo lee la bitácora y las curvas de capital de la evaluación.
    """
    if not all(evaluate.AGENT in res for res in results.values()):
        return
    logs = {k: res[evaluate.AGENT]["trades"] for k, res in results.items()}
    k, start, end = plots.busiest_window(logs)
    agent, btc = results[k][evaluate.AGENT], results[k]["btc_buy_hold"]
    plots.write_agent_simulation_gif(ohlcv, agent["trades"], agent["equity"], btc["equity"], k, start, end,
                                     ARTIFACTS / "agent_simulation.gif")
    print(f"[simulación] agent_simulation.gif: fold {k}, {start:%Y-%m-%d} a {end:%Y-%m-%d}")


def chosen_models():
    """Configuración, pasos y semilla elegidos por fold en la parte 05 (de `run_metadata.json`)."""
    chosen = json.loads((ARTIFACTS / "run_metadata.json").read_text(encoding="utf-8"))["chosen"]
    return {int(k): c for k, c in chosen.items()}


def run_ablation_stage():
    """Parte 07 (GC), pregunta 7: ablación por familias de la observación.

    Reentrena la configuración elegida de cada fold, con sus pasos y sus 3 semillas, quitando una familia de features a
    la vez (o la cuenta) y la base sin quitar nada, que reproduce el modelo elegido. Escribe `artifacts/ablation.csv`
    (medianas entre semillas en validación y prueba, y la diferencia contra la base). Se reporta; no reelige el modelo.
    """
    chosen = chosen_models()
    jobs = [job for k, dates in FOLDS.items()
            for job in train.ablation_jobs(k, dates, BTC_CSV, MODELS / "ablation", chosen[k]["config_id"],
                                           chosen[k]["timesteps"])]
    results, _, _ = run_search(jobs, MODELS / "ablation" / "search.pkl", "ablación")
    table = train.ablation_table(results)
    table.to_csv(ARTIFACTS / "ablation.csv", float_format="%.6g")
    base = results[results["variant"] == "ninguna"]
    for k in FOLDS:  # la base reproduce la búsqueda (mismo modelo): verificación barata de la ablación
        expected = pd.read_csv(ARTIFACTS / f"validation_results_fold_{k}.csv")
        expected = expected[(expected["config_id"] == chosen[k]["config_id"])
                            & (expected["timesteps"] == chosen[k]["timesteps"])].set_index("seed")["val_log_growth"]
        got = base[base["fold"] == k].set_index("seed")["val_log_growth"]
        if not np.allclose(got.sort_index(), expected.sort_index(), atol=1e-5):
            raise RuntimeError(f"la base de la ablación no reproduce el modelo elegido del fold {k}")
    mean = table.loc["media", "delta_val"].drop("ninguna")
    print("[ablación] Δ crecimiento log en validación al quitar cada familia (media de folds): "
          + "; ".join(f"{v} {d:+.3f}" for v, d in mean.items()))


def run_second_algorithm_stage(ohlcv, features):
    """Parte 07 (GC), pregunta 7: segundo algoritmo (PPO) con la misma búsqueda en validación y el mismo criterio.

    Escribe `artifacts/validation_results_ppo.csv` y `artifacts/algorithm_comparison.csv`: por fold y algoritmo, el
    candidato elegido solo con validación y el resultado de su modelo congelado en prueba (una vez).
    """
    jobs = [job for k, dates in FOLDS.items()
            for job in train.search_jobs(k, dates, BTC_CSV, MODELS / "ppo", algo="ppo")]
    ppo_results, _, _ = run_search(jobs, MODELS / "ppo" / "search.pkl", "segundo algoritmo")
    ppo_results.to_csv(ARTIFACTS / "validation_results_ppo.csv", index=False, float_format="%.6g")
    rows = {}
    for k, dates in FOLDS.items():
        fold = data.build_fold_dataset(ohlcv, features, k, dates)
        searches = {"dqn": (pd.read_csv(ARTIFACTS / f"validation_results_fold_{k}.csv"), MODELS, train.DQN),
                    "ppo": (ppo_results[ppo_results["fold"] == k], MODELS / "ppo", train.PPO)}
        for algo, (results, models_dir, cls) in searches.items():
            summary = train.chosen_summary(results)
            path = train.model_path(models_dir, k, summary["config_id"], summary["seed"], summary["timesteps"])
            policy = train.as_policy(cls.load(path, device="cpu"))
            test = train.evaluate_policy(fold, "test", policy, SEED + k)
            rows[(k, algo)] = {**summary, **train.GRIDS[algo][summary["config_id"]],
                               **{f"test_{m}": v for m, v in test.items()}}
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis(["fold", "algo"])
    table["net_arch"] = table["net_arch"].astype(str)
    numeric = ["val_median", "val_min", "val_max", "candidates_above_zero", "test_log_growth", "test_sharpe",
               "test_n_trades"]
    mean = table[numeric].groupby(level="algo").mean()
    mean.index = pd.MultiIndex.from_product([["media"], mean.index], names=["fold", "algo"])
    pd.concat([table, mean]).to_csv(ARTIFACTS / "algorithm_comparison.csv", float_format="%.6g")
    print("[segundo algoritmo] crecimiento log medio en prueba: "
          + "; ".join(f"{a} {mean.loc[('media', a), 'test_log_growth']:+.3f}" for a in ("dqn", "ppo")))


def main():
    """Corre el pipeline completo de punta a punta con la semilla global."""
    np.random.seed(SEED)
    ARTIFACTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ohlcv, features = run_data_stage()
    run_eda_stage(ohlcv, features)
    agent_policies = run_training_stage(ohlcv, features)
    results = run_evaluation_stage(ohlcv, features, agent_policies)
    run_simulation_stage(ohlcv, results)
    run_sensitivity_stage(ohlcv, features, agent_policies)
    run_ablation_stage()
    run_second_algorithm_stage(ohlcv, features)
    print(f"Pipeline terminado. Artefactos en {ARTIFACTS}")


if __name__ == "__main__":
    main()
