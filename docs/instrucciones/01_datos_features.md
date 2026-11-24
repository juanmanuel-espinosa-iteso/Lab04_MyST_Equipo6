# 01 — Datos, features causales, folds y normalizador

**Dueño:** Juan Manuel Espinosa (JME) · **Revisor:** Gonzalo Cano (GC) · **Notebook:** `notebooks/01_datos_features.ipynb` · **Módulo:** `code/data.py` · **Depende de:** — · **Alimenta a:** 02, 03, 05, 06
**Leer antes:** `docs/notas_clase/02_rl_q_learning_gymnasium.md` (§6, diseñar bien la observación) · `docs/Lab04.pdf` §3.1

## Objetivo

Cargar el CSV horario de BTCUSDT, limpiarlo y validarlo, construir variables **causales** agrupadas en familias
(para la ablación de la parte 07), partir los 4 folds de §3.6 y ajustar un normalizador por fold solo con su
entrenamiento.

## Parámetros de este módulo

Ventanas de las features, lista de familias y método de normalización (z-score o cuantiles) al inicio de
`code/data.py`. Recibe la ruta del CSV y `FOLDS` desde `main.py`. Si cambian, actualizar la tabla de `CLAUDE.md`.

## Contrato

- **Lee:** `data/Binance_BTCUSDT_1h.csv` (primera línea = URL de CryptoDataDownload; orden descendente; columnas
  `Unix, Date, Symbol, Open, High, Low, Close, Volume BTC, Volume USDT, tradecount`).
- **Escribe:** `artifacts/folds.json`, `artifacts/data_report.json`.
- **Expone:** `load_ohlcv(path) -> (DataFrame, report)`, `compute_features(ohlcv) -> DataFrame`,
  `FEATURE_FAMILIES` / `FEATURES`, `forward_log_return(ohlcv, horizon)`, `split_fold(df, fold_dates)`,
  `Normalizer.fit(train)` / `.transform(df)`, `build_fold_dataset(ohlcv, features, fold, fold_dates) -> FoldData`
  (`.ohlcv[name]`, `.features[name]`, `.normalizer`), `describe_folds(...)`. `ohlcv` trae `gap_hours`.

## Loop de trabajo

```
REPETIR hasta cumplir el criterio de terminado:
  1. Toma el primer ítem sin [x] del checklist.
  2. Impleméntalo en code/data.py con docstring.
  3. Verifica: shape, rango de fechas, NaN, huecos; corre la prueba de causalidad si tocaste features.
  4. Marca [x], registra la decisión en docs/decisiones.md, commit.
```

## Checklist

- [x] Lectura del CSV: saltar la línea de la URL, ordenar ascendente, índice `datetime` en UTC, columnas en minúsculas.
- [x] Validación: duplicados, huecos horarios, precios > 0, `low ≤ open, close ≤ high`, índice ordenado. El enunciado
      dice 75,558 registros y el archivo trae 75,588 filas de datos: explicar la diferencia en `data_report.json`.
- [x] Tratamiento de huecos documentado (no inventar barras con precios futuros; nunca back-fill).
- [x] Features **causales**, cada una calculable con barras hasta t inclusive (la decisión se toma al cierre de t y
      se ejecuta en la apertura de t + 1). Familias propuestas para la ablación: rendimientos rezagados, volatilidad
      (realizada, Parkinson / Garman–Klass), tendencia/momento, volumen y calendario (hora y día en seno/coseno).
- [x] Componente de **creencia** del estado (§3.2): lo estimado con su incertidumbre (p. ej. volatilidad o tendencia
      estimada con su error estándar o una banda); se define junto con la parte 08.
- [x] Objetivo para el análisis de features: rendimiento futuro a horizonte H (lo usa la parte 02; **nunca** entra al estado).
- [x] Folds de §3.6 con fechas exactas en `main.py` y `artifacts/folds.json`; prueba 2026-01-01 a 2026-04-26 en el fold 4.
- [x] Calentamiento de las ventanas: las primeras barras con NaN se descartan dentro del entrenamiento, no se rellenan.
- [x] Normalizador ajustado **solo** con el entrenamiento de cada fold y aplicado sin cambios a validación y prueba;
      uno por fold, sus estadísticos guardados para el reporte.
- [x] `test_causality.py` (prueba obligatoria 2): la observación reconstruida con datos hasta el índice i es igual a la
      calculada sobre la serie completa.

**Estado (2026-11-24, JME):** terminado en la rama `feat/jme-datos`. El contenido final del estado se confirma con la evidencia de la parte 02.

## Criterio de terminado

`build_fold_dataset(k)` devuelve, para k = 1..4, entrenamiento, validación y prueba sin NaN, con features
normalizadas y OHLC crudo; `test_causality.py` pasa; `folds.json` y `data_report.json` existen.

## Trampas

- `rolling(...).mean()` está bien; `shift(-1)`, `center=True`, `bfill` o normalizar con toda la serie es look-ahead.
- Normalizar con estadísticos de validación o prueba (o reajustar el normalizador en cada partición).
- Usar el precio de cierre de t como precio de llenado de una decisión tomada con ese mismo cierre.
- Features en niveles de precio (no estacionarias): usar rendimientos y razones.

## Preguntas probables

- ¿Por qué cada feature es causal? ¿Dónde se ajusta el normalizador y por qué uno por fold?
- ¿Qué hicieron con los huecos y con las 30 filas de diferencia?
