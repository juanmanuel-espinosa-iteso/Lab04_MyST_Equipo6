# CLAUDE.md — Lab04 MyST Equipo 6

Agente de reinforcement learning (Stable-Baselines3) que opera BTCUSDT con datos horarios. El foco es el **diseño
del entorno**: formulación de Powell, estado, función de contribución, simulador semi-Markov realista y evaluación
walk-forward rigurosa. Materia: Microestructuras y Sistemas de Trading (ITESO). Enunciado original: `docs/Lab04.pdf`.

- **Equipo:** Gonzalo Cano (GC) y Juan Manuel Espinosa (JME).
- **Alcance:** Lab04 completo (incluye ablación por familias de variables y segundo algoritmo, pregunta 7).
- **Repositorio:** público, `juanmanuel-espinosa-iteso/Lab04_MyST_Equipo6`. Exposición: **lunes 30 de noviembre de
  2026**; se evalúa el último commit antes del **domingo 29 de noviembre, 23:59**.

## Cómo trabajar en este repo (leer primero)

1. Antes de implementar una parte, lee su guía en `docs/instrucciones/` y las notas de clase que indica en
   `docs/notas_clase/`. La implementación sigue la notación y el enfoque de clase.
2. Cada guía es un **loop de trabajo**: implementar → verificar → marcar checklist → commit, hasta cumplir el
   "Criterio de terminado". Marca `[x]` conforme se avance.
3. Respeta el **contrato de artefactos** (abajo): es lo que permite trabajar en paralelo.
4. Ambos integrantes deben poder explicar **cualquier línea** (si alguien no puede, la sección se califica como
   Deficiente). Prioriza claridad sobre astucia y explica el porqué en docstrings.
5. Toda decisión de diseño se registra en `docs/decisiones.md` con **elección, al menos una alternativa descartada
   y por qué, con referencia a datos u observaciones empíricas** (§3.2: las justificaciones genéricas no reciben crédito).

## Reglas duras (romperlas invalida resultados)

- **Sin look-ahead.** Cada feature se calcula de forma causal con información disponible en el momento de la
  decisión o antes. El agente decide con la barra t ya cerrada y la orden se llena en la apertura de t + 1.
- **Normalización por fold.** Medias, desviaciones, cuantiles y cualquier estadístico se ajustan **solo con el
  entrenamiento de cada fold** y se aplican sin cambios a validación y prueba; cada fold tiene su normalizador. Si se
  usa `VecNormalize`, se congela (`training=False`, `norm_reward=False`) en validación y prueba.
- **Análisis de datos solo en entrenamiento.** EDA, autocorrelación, estacionariedad e información mutua usan solo
  los folds de entrenamiento.
- **Walk-forward de 4 folds con ventana expansiva** (tabla abajo). **Cuatro modelos separados**, uno por fold.
  Validación sirve solo para elegir hiperparámetros; **prueba se toca exactamente una vez**, con todo congelado.
- **Varias semillas** por configuración; se reporta la dispersión entre semillas y media ± desviación entre folds.
- **Antes de entrenar:** `check_env` sin advertencias y líneas base triviales (aleatorio, siempre largo, siempre
  plano), todas con costos. Si el aleatorio gana dinero, el simulador tiene un error.
- **Costos de 0.125 % por lado** dentro de la contribución (entrada y salida), también en los benchmarks.
- **Resolución intrabarra:** si en una misma barra se tocan stop-loss y take-profit, por defecto el **stop se
  ejecuta primero** (conservador). Se cuantifica qué proporción de operaciones cambia y cuánto cambia el resultado
  bajo el supuesto contrario.
- **`terminated` solo para eventos reales** (ruina o límite duro de riesgo); el fin de datos es `truncated` (S18).
- **Semilla fija: `SEED = 42`** en `main.py`; las semillas de entrenamiento se derivan de ella y toda función con
  aleatoriedad la recibe como argumento.
- **Metadatos de cada corrida:** versiones de Stable-Baselines3, Gymnasium y torch, semilla, fold e
  hiperparámetros en `artifacts/run_metadata.json` (§2.2).
- **Datos externos de precio no permitidos.** Única excepción: el S&P 500 (`^GSPC`) para el benchmark, nunca como
  feature (ver `docs/decisiones.md`).
- **Sin archivos sueltos en la raíz:** solo `README.md`, `requirements.txt`, `.gitignore` y `main.py`.
- `python main.py` corre todo el pipeline de punta a punta, sin intervención, y produce todos los artefactos.

## Folds (§3.6)

| Fold | Entrenamiento | Validación | Prueba |
|---|---|---|---|
| 1 | 2017-08-17 a 2021-12-31 | 2022 | 2023 |
| 2 | 2017-08-17 a 2022-12-31 | 2023 | 2024 |
| 3 | 2017-08-17 a 2023-12-31 | 2024 | 2025 |
| 4 | 2017-08-17 a 2024-12-31 | 2025 | 2026-01-01 a 2026-04-26 |

## Parámetros (cada uno vive en el módulo que lo usa)

No hay archivo de configuración global: los parámetros globales van arriba de `main.py` y se pasan como argumentos;
los demás son constantes en mayúsculas al inicio de su módulo. Esta tabla es el índice único: si cambias o mueves un
valor, actualízala. Los marcados *por decidir* se fijan en su parte y se registran en `decisiones.md`.

| Parámetro | Valor | Dónde vive |
|---|---|---|
| Semilla | `SEED = 42`; semillas de entrenamiento derivadas (por decidir: cuántas, mínimo 3) | `main.py` |
| Rutas (`DATA`, `ARTIFACTS`, `FIGURES`) y fechas de los folds (`FOLDS`) | — | `main.py` |
| Archivo de datos | `data/Binance_BTCUSDT_1h.csv` (CryptoDataDownload, Binance, horario OHLCV) | `main.py` → `code/data.py` |
| Costo de transacción | 0.125 % por lado (`COST_PER_SIDE = 0.00125`) | `code/env.py` |
| Features y ventanas | por decidir (parte 01) | `code/data.py` |
| Acciones, horizonte H, stop-loss, take-profit | por decidir (parte 03) | `code/env.py` |
| Resolución intrabarra | stop primero (`INTRABAR = "stop_first"`) | `code/env.py` |
| Definición de episodio | por decidir (parte 03) | `code/env.py` |
| Contribución | por decidir (parte 04): terminal al cierre o por paso | `code/reward.py` |
| Algoritmo principal / segundo | por decidir (parte 05; propuesta DQN con descuento semi-Markov γ^τ / PPO) | `code/train.py` |
| Rejilla de hiperparámetros y criterio de selección | por decidir (parte 05) | `code/train.py` |
| Anualización de métricas | por decidir (parte 06): BTC opera 24/7, el S&P 500 en días hábiles | `code/evaluate.py` |

## Estructura (obligatoria por el enunciado + extras marcados con *)

```
main.py              pipeline completo: python main.py (semilla, rutas, folds)
code/                data.py · env.py · reward.py · train.py · evaluate.py · plots.py (§2.2)
artifacts/           todos los artefactos exigidos (§3.8); se versionan, salvo los modelos (artifacts/models/)
tests/               pytest: check_env, causalidad, operación de referencia, identidad contable (§3.11)
docs/                reporte.pdf, walkthrough.pdf, presentacion.pdf, Lab04.pdf, decisiones.md *, instrucciones/ *,
                     notas_clase/ *, figuras/ *
data/ *              Binance_BTCUSDT_1h.csv (versionado) y snapshot del S&P 500
notebooks/ *         un notebook de demostración por parte (solo importa code/)
project_management/ * charter.md, sprint_NN.md, postmortem.md
```

**Ojo con el nombre `code/`:** `code` también es un módulo de la biblioteca estándar. Con `python main.py` y `pytest`
desde la raíz gana el paquete local, pero en Jupyter el `code` estándar ya está importado y `from code import data`
falla. `code/__init__.py` queda vacío; los notebooks usan la solución registrada en `decisiones.md`.

## Contrato de artefactos (`artifacts/`)

Formato: `.csv` para tablas, `.png` para figuras, `.json` para escalares y metadatos, `.parquet` para series
intermedias. `{k}` = 1, 2, 3, 4. Los marcados con † son obligatorios por §3.8.

| Archivo | Productor | Contenido |
|---|---|---|
| `folds.json` | data | fechas de corte de cada fold y número de barras por partición |
| `data_report.json` | data | limpieza: filas leídas, duplicados, huecos horarios, orden, rango de fechas |
| `btcusdt_eda.pdf` † | eda (plots) | regímenes, distribución de rendimientos, autocorrelación (rendimientos y cuadrados), estacionariedad (ADF/KPSS), solo entrenamiento |
| `regimes.csv` | eda | ≥ 4 regímenes del entrenamiento: fechas, volatilidad, tendencia, drawdown (pregunta 1) |
| `feature_analysis_fold_{k}.csv` † | eda | por feature: información mutua y correlaciones con el objetivo, ADF, autocorrelación, solo train del fold |
| `baselines.csv` | train | aleatorio, siempre largo y siempre plano con costos, por fold y partición |
| `training_curves_fold_{k}.png` † | train (plots) | recompensa por episodio contra pasos, por semilla, con anotaciones |
| `validation_results_fold_{k}.csv` † | train | cada configuración × semilla: métricas en validación y la elegida |
| `run_metadata.json` | train | versiones de SB3, Gymnasium, torch, Python; semillas, hiperparámetros elegidos, pasos, tiempo de cómputo |
| `trade_log_fold_{k}.csv` † | evaluate | entrada y salida (marca de tiempo y precio), razón de salida (horizonte, stop, take-profit, cierre), tamaño, P&L bruto, costos, P&L neto, duración, estado en la entrada |
| `equity_curves_test_fold_{k}.png` † | evaluate (plots) | estrategia contra BTC buy-and-hold y S&P 500 en prueba |
| `metrics_summary.csv` † | evaluate | todas las métricas de §3.7 por fold y agregadas (media y desviación estándar) para estrategia y benchmarks |
| `trade_analysis.json` | evaluate | 3 peores operaciones de prueba, periodo del mayor drawdown, correlación con BTC (§3.9) |
| `max_drawdown_fold_{k}.png` | evaluate (plots) | acciones del agente y precio en el periodo del mayor drawdown (§3.9.2) |
| `sensitivity.csv` | sensitivity | resolución intrabarra (proporción afectada y cambio de resultado) y retraso de una barra en los datos (§3.3, §3.10, P6) |
| `ablation.csv` | sensitivity | desempeño quitando cada familia de variables (P7) |
| `algorithm_comparison.csv` | sensitivity | algoritmo principal contra el segundo (P7) |

Si cambias el esquema de un artefacto, actualiza esta tabla **y** avisa al otro integrante.

## Convenciones de código

- Python 3.12 (`.venv`); dependencias directas con versión fijada en `requirements.txt`.
- Identificadores en inglés (`compute_features`), docstrings y comentarios en español.
- Funciones puras y de responsabilidad única; `main.py` solo orquesta.
- Docstring en toda función pública: qué hace, parámetros, retorno y supuestos (causalidad, unidades).
- Sin código muerto ni duplicado. Los notebooks solo importan funciones de `code/` y muestran resultados.
- Figuras: título, ejes etiquetados, leyenda legible. Las del reporte se exportan a `docs/figuras/`.
- Correr `pytest` antes de cada push.

## Flujo de Git (commits balanceados 50/50)

- `main` siempre corre. Trabajo en ramas `feat/<iniciales>-<tema>` (ej. `feat/jme-entorno`, `feat/gc-recompensa`).
- Commits pequeños y descriptivos en español: `feat(env): gestor de posiciones con stop primero dentro de la barra`.
  Prefijos: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`.
- Cada integrante hace commits **desde su propia cuenta** (integrante sin commits = 0 en la parte técnica).
- Merge vía Pull Request revisado por el otro integrante. Revisar el balance con `git shortlog -sn --no-merges`.
- División del trabajo y dueños: `docs/instrucciones/00_flujo_de_trabajo.md`.

## Mapa de documentación

- `docs/instrucciones/`: una guía por parte con pasos, checklist y criterio de terminado.
  `99_checklist_entrega.md` es la revisión final contra la rúbrica.
- `docs/notas_clase/`: resumen de clase (Powell, S18 y S19). **Fuente de verdad para notación y enfoque.**
- `docs/decisiones.md`: bitácora de decisiones de diseño.
