# Lab04_MyST_Equipo6 — Agente de Reinforcement Learning para Trading

Microestructuras y Sistemas de Trading · ITESO

## Integrantes

- Gonzalo Cano
- Juan Manuel Espinosa

## Nivel de alcance

Lab04 completo: formulación de Powell semi-Markov, entorno Gymnasium con gestor de posiciones (horizonte, stop-loss,
take-profit), agente de Stable-Baselines3 entrenado por fold con varias semillas, walk-forward de 4 folds, benchmarks
(BTC buy-and-hold y S&P 500), análisis de operaciones, sensibilidad, ablación por familias de variables y segundo
algoritmo.

## Descripción

*(En construcción.)* Agente de RL que opera BTCUSDT con datos horarios de Binance (2017-08-17 a 2026-04-26), con
costos de 0.125 % por lado.

## Instalación

Requiere **Python 3.12**.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducción

```bash
python main.py
```

Corre el pipeline completo y escribe todos los artefactos en `artifacts/`. **Tiempo:** ≈ 86 min en una Mac de 11 núcleos (5 de rendimiento y 6 de eficiencia): ≈ 41 min de DQN (192 corridas),
≈ 13 de ablación (96) y ≈ 33 de PPO (96), en 10 procesos (`LAB04_PROCESSES` cambia el número). Para volver a correr sin
reentrenar, `LAB04_REUSE_MODELS=1 python main.py` (≈ 3.5 min) reusa los modelos de `artifacts/models/` solo si cada
búsqueda se hizo con los mismos parámetros (entorno, features, datos, rejilla y semillas); un clon limpio siempre entrena. Los datos de BTCUSDT están versionados en
`data/Binance_BTCUSDT_1h.csv`; el S&P 500 (solo benchmark) se toma de `data/sp500_daily.csv`.

**Semilla aleatoria:** `SEED = 42`, fijada en `main.py` y aplicada al inicio del pipeline; de ella salen las semillas
de las líneas base, la evaluación y la sensibilidad (`SEED + k` en el fold k). Las tres semillas del entrenamiento son
fijas y explícitas, `SEEDS = (0, 1, 2)` en `code/train.py`. Toda función con aleatoriedad recibe su semilla como
argumento.

## Pruebas

```bash
pytest
```

## Uso de asistencia de IA

*(Se completa al cerrar el proyecto.)*

- **JME:** Claude Code (Anthropic) como asistente de programación y redacción: estructura del repositorio, guías de
  trabajo y resumen de las notas de clase.
- **GC:**
- **Cómo se verificó:**
