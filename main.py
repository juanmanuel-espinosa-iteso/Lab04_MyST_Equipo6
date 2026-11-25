"""Punto de entrada único del pipeline del Lab04: python main.py

Orquesta, con semillas fijas, todas las etapas que producen los artefactos de `artifacts/` (§3.8 del enunciado):
datos y features → EDA → entrenamiento por fold → evaluación en prueba → sensibilidad. La lógica vive en `code/`;
aquí solo se definen los parámetros globales (semilla, rutas y folds) y el orden de las etapas.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# `code/` (obligatoria, §2.2) se llama igual que un módulo de la biblioteca estándar: se asegura el paquete local.
sys.path.insert(0, str(ROOT))
if not str(getattr(sys.modules.get("code"), "__file__", "")).startswith(str(ROOT)):
    sys.modules.pop("code", None)

from code import data  # noqa: E402

SEED = 42

DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "docs" / "figuras"
BTC_CSV = DATA / "Binance_BTCUSDT_1h.csv"

# Walk-forward de ventana expansiva (§3.6): fechas inclusivas, horas en UTC.
FOLDS = {
    1: {"train": ("2017-08-17", "2021-12-31"), "validation": ("2022-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    2: {"train": ("2017-08-17", "2022-12-31"), "validation": ("2023-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
    3: {"train": ("2017-08-17", "2023-12-31"), "validation": ("2024-01-01", "2024-12-31"), "test": ("2025-01-01", "2025-12-31")},
    4: {"train": ("2017-08-17", "2024-12-31"), "validation": ("2025-01-01", "2025-12-31"), "test": ("2026-01-01", "2026-04-26")},
}


def write_json(obj, path):
    """Escribe `obj` como JSON legible (UTF-8, sangría de 2)."""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


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


def run_eda_stage():
    """Parte 02: EDA, regímenes y análisis de features solo en entrenamiento (btcusdt_eda.pdf, feature_analysis)."""


def run_training_stage():
    """Parte 05: check_env, líneas base y entrenamiento por fold y semilla con búsqueda de hiperparámetros en validación."""


def run_evaluation_stage():
    """Parte 06: evaluación única en prueba, bitácora, benchmarks y métricas agregadas entre folds."""


def run_sensitivity_stage():
    """Parte 07: resolución intrabarra, retraso de una barra, ablación y segundo algoritmo."""


def main():
    """Corre el pipeline completo de punta a punta con la semilla global."""
    np.random.seed(SEED)
    ARTIFACTS.mkdir(exist_ok=True)
    run_data_stage()
    run_eda_stage()
    run_training_stage()
    run_evaluation_stage()
    run_sensitivity_stage()
    print(f"Pipeline terminado. Artefactos en {ARTIFACTS}")


if __name__ == "__main__":
    main()
