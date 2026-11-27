# 05 — Entrenamiento con Stable-Baselines3: líneas base, semillas e hiperparámetros

**Dueño:** Gonzalo Cano (GC) · **Revisor:** Juan Manuel Espinosa (JME) · **Notebook:** `notebooks/05_entrenamiento.ipynb` · **Módulo:** `code/train.py` (+ `code/plots.py` para las curvas) · **Depende de:** 01, 03, 04 · **Alimenta a:** 06, 07
**Leer antes:** `docs/notas_clase/02_rl_q_learning_gymnasium.md` (§3–5) · `docs/notas_clase/03_sb3_dqn_ppo.md` (§4–6: DQN, PPO, `Monitor`, evaluación y semillas) · `docs/Lab04.pdf` §3.5 y §3.6

## Objetivo

Entrenar **un modelo por fold** con varias semillas, elegir hiperparámetros **solo con validación**, congelar todo y
dejar listo el modelo elegido de cada fold para tocar prueba una sola vez.

## Parámetros de este módulo

Algoritmo, `TOTAL_TIMESTEPS`, rejilla de hiperparámetros, número de semillas, criterio de selección y número de
procesos al inicio de `code/train.py`.

## Contrato

- **Lee:** `build_fold_dataset(k)`, `TradingEnv`, `REWARDS`.
- **Escribe:** `artifacts/baselines.csv`, `artifacts/validation_results_fold_{k}.csv`,
  `artifacts/training_curves_fold_{k}.png`, `artifacts/run_metadata.json`, modelos en `artifacts/models/` (no se versionan).
- **Expone (propuesta):** `run_baselines(fold)`, `train_agent(fold, config, seed) -> model`,
  `hyperparameter_search(fold) -> DataFrame`, `select_config(results) -> config`.

## Loop de trabajo

```
REPETIR hasta cumplir el criterio de terminado:
  1. Toma el primer ítem sin [x].
  2. Implementa; corre primero con pocos pasos y un fold.
  3. Verifica: líneas base con sentido, curvas que suben, validación sin tocar prueba.
  4. Marca [x], registra la decisión con referencia a validation_results, commit.
```

## Checklist

- [x] `check_env` antes de cualquier entrenamiento (en el pipeline, no solo en la prueba).
- [x] **Líneas base** con costos en cada fold y partición: aleatorio, siempre largo, siempre plano → `baselines.csv`.
      Si el aleatorio gana dinero, parar y revisar el simulador.
- [x] Algoritmo principal con justificación (propuesta: DQN; Q-learning con red, off-policy, coherente con S18–S19).
- [x] **Replay buffer semi-Markov:** cada transición guarda (estado en la apertura, acción, recompensa al cierre, estado
      al cierre, duración real τ) y el objetivo usa γ^τ (subclase del buffer y del `train` de DQN). Si se usa un
      algoritmo sin esto, declarar la aproximación.
- [x] **Varias semillas** (mínimo 3) por configuración; dispersión entre semillas reportada.
- [x] **Búsqueda de hiperparámetros** en validación: rejilla chica y documentada (p. ej. tasa de aprendizaje, γ, tamaño de
      red, exploración y, si aplica, H/SL/TP) → `validation_results_fold_{k}.csv` con todas las configuraciones × semillas.
- [x] **Criterio de selección** fijado antes de ver resultados (p. ej. mediana entre semillas del Sharpe neto en
      validación) y documentado con referencia a esos archivos.
- [x] Si hay `VecNormalize`, estadísticos congelados en validación y prueba.
- [x] `training_curves_fold_{k}.png`: recompensa por episodio contra pasos, por semilla, con anotaciones de los cambios
      relevantes (fin de la exploración, mesetas, caídas).
- [x] `run_metadata.json`: versiones de SB3, Gymnasium, torch y Python; semillas; configuración elegida por fold; pasos y tiempo.
- [x] Presupuesto de cómputo: `python main.py` completo en ≤ ~1.5 h en la Mac M2 (procesos en paralelo por fold);
      tiempo real documentado en el README.

**Estado (2026-11-27, GC):** hecho en `feat/gc-entrenamiento`. `code/train.py` (DQN con `SemiMarkovReplayBuffer`,
rejilla, búsqueda en paralelo, criterio A y reuso de la búsqueda), `tests/test_train.py`, `notebooks/05_entrenamiento.ipynb`
y 8 filas en `docs/decisiones.md`. La misma configuración gana en los 4 folds; en validación casi ningún candidato le gana
a no operar. `python main.py` completo: ≈ 43 min (40.4 de entrenamiento con 10 procesos).

## Criterio de terminado

Cuatro modelos (uno por fold), elegidos solo con validación y con varias semillas; artefactos de §3.8 de esta parte
completos; prueba aún sin tocar.

## Trampas

- Elegir hiperparámetros o semilla viendo prueba (invalida el protocolo).
- Un solo modelo evaluado en los cuatro periodos de prueba (invalida el protocolo).
- Reportar la mejor semilla en vez de la dispersión.
- Bootstrap después de `terminated` o descuento γ por paso cuando la transición duró τ barras.

## Preguntas probables

- ¿Cómo eligieron los hiperparámetros y dónde está la evidencia? ¿Cuánto varían las semillas?
- ¿Cómo maneja el replay buffer la duración variable?
