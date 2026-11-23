# 99 — Checklist final de entrega (contra la rúbrica)

Correr este loop el **domingo 29 de noviembre**, antes de las 23:59, desde un clon limpio del repo.

```
REPETIR hasta que todo esté en [x]:
  1. git clone https://github.com/juanmanuel-espinosa-iteso/Lab04_MyST_Equipo6.git /tmp/check && cd /tmp/check
  2. python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
  3. python main.py && pytest
  4. Revisar la lista; arreglar lo que falle en el repo real; commit; volver a 1.
```

## A. Estructura y reproducibilidad (10)

- [ ] Repo público `Lab04_MyST_Equipo6` (se clona sin credenciales); profesor y Gonzalo como colaboradores.
- [ ] Estructura obligatoria de §2.2: `README.md`, `requirements.txt`, `.gitignore`, `main.py`, `code/{data,env,reward,train,evaluate,plots}.py`, `artifacts/`, `tests/`, `docs/{reporte,walkthrough,presentacion}.pdf`.
- [ ] `python main.py` corre sin errores desde un clon limpio y produce todos los artefactos (−10 si no); tiempo anotado en el README.
- [ ] `requirements.txt` con versiones fijadas (SB3, Gymnasium y torch incluidos) e instalación en un entorno nuevo de Python 3.12.
- [ ] Versiones de SB3, Gymnasium y torch en `artifacts/run_metadata.json`.
- [ ] Semilla fijada y documentada (`SEED = 42`, README).
- [ ] README: integrantes, nivel de alcance, instalación, comando exacto de reproducción, uso de IA.
- [ ] Ambos integrantes con commits propios y balanceados; historial descriptivo (`git shortlog -sn --no-merges`).

## B. Lógica del modelo (10)

- [ ] Formulación de Powell completa (estado físico / información / creencia, decisión, exógena, transición, contribución, objetivo).
- [ ] Semi-Markov correcto en el replay buffer (estado en la apertura, acción, recompensa al cierre, estado al cierre, duración).
- [ ] Resolución intrabarra declarada (stop primero) y cuantificada.
- [ ] Normalizador por fold solo con entrenamiento; features causales (prueba 2).
- [ ] Supuestos del modelo declarados explícitamente en el reporte.

## C. Resultados y análisis (10)

- [ ] Los 7 artefactos de §3.8 para los 4 folds (`btcusdt_eda.pdf`, `feature_analysis`, `training_curves`, `validation_results`, `equity_curves_test`, `trade_log`, `metrics_summary`).
- [ ] Líneas base (aleatorio, siempre largo, siempre plano) con costos; el aleatorio pierde.
- [ ] Varias semillas con dispersión reportada; 4 modelos separados; prueba tocada una sola vez.
- [ ] Métricas de §3.7 con media y desviación entre folds, contra ambos benchmarks con costos.
- [ ] §3.9 (peores operaciones, mayor drawdown, correlación), §3.10 (fallas de producción) y las 7 preguntas con cifras propias.
- [ ] Advertencia de interpretación y limitaciones explícitas.

## D. Código, pruebas y documentación (10)

- [ ] Código modular, funciones de responsabilidad única, nombres descriptivos, docstrings, sin código muerto ni duplicado.
- [ ] Las 4 pruebas obligatorias presentes y en verde.
- [ ] Uso de IA declarado en el README (ambos integrantes).

## Exposición (60)

- [ ] `docs/reporte.pdf`, `docs/walkthrough.pdf` y `docs/presentacion.pdf` en `docs/` antes de la exposición (−5 si no).
- [ ] ≤ 14 diapositivas sin contar portada y cierre; ensayo de 10 min con ambos exponiendo.
- [ ] Ambos repasan las "preguntas probables" de todas las guías y saben dónde está cada cosa en el código.
- [ ] Último commit antes del domingo 29, 23:59 (−20 si es tarde).
