# 07 — Sensibilidad, ablación, segundo algoritmo y fallas de producción

**Dueños:** JME (resolución intrabarra, retraso de una barra, latencia y ejecución) · GC (ablación por familias y segundo
algoritmo) · **Revisor:** cruzado · **Notebook:** `notebooks/07_sensibilidad.ipynb` · **Módulos:** `code/evaluate.py`, `code/train.py`, `code/env.py` · **Depende de:** 05, 06 · **Alimenta a:** 10
**Leer antes:** `docs/Lab04.pdf` §3.3 (punto sutil), §3.10, preguntas 6 y 7 y la advertencia de interpretación

## Objetivo

Medir qué tan frágil es el resultado ante los supuestos del simulador y responder con cifras propias qué tan lejos
está de la operación real. El simulador supone que el resto del sistema es gratuito, instantáneo y perfecto: hay que
declarar esa distancia y estimar su magnitud.

## Contrato

- **Escribe:** `artifacts/sensitivity.csv`, `artifacts/ablation.csv`, `artifacts/algorithm_comparison.csv`.

## Loop de trabajo

```
REPETIR hasta cumplir el criterio de terminado:
  1. Toma el primer ítem sin [x].
  2. Implementa el experimento reutilizando env/evaluate (sin copiar código).
  3. Verifica que solo cambie el supuesto que se estudia.
  4. Marca [x], anota la cifra en decisiones.md o en el borrador del reporte, commit.
```

## Checklist

- [x] **Intrabarra (§3.3):** proporción de operaciones con stop y take-profit en la misma barra y cambio del resultado
      con `tp_first` contra `stop_first` (mismo modelo, misma prueba).
- [x] **Retraso de una barra (pregunta 6 y latencia):** la observación llega con una barra de retraso; desempeño del
      mismo modelo.
- [x] **Latencia de 2 s:** argumento cuantitativo de si el precio de llenado supuesto (apertura de t + 1) sigue disponible
      (p. ej. con el rango típico de la barra horaria o el movimiento en minutos).
- [x] **Ejecución:** a qué precio llena el simulador y qué pasa al operar tamaño (profundidad del libro, slippage);
      sensibilidad a un costo extra por lado.
- [ ] **Capacidad y saturación:** si 50 fondos operan algo similar; quién está del otro lado y por qué.
- [ ] **Cambio de régimen:** qué pasa si los próximos 12 meses se parecen a 2018, a 2022 o a un régimen ausente; qué
      variables se degradarían más (con el análisis de features de la parte 02 y la ablación).
- [ ] **Ablación (pregunta 7):** reentrenar quitando cada familia de variables con la configuración elegida y varias semillas
      → `ablation.csv`.
- [ ] **Segundo algoritmo (pregunta 7):** p. ej. PPO con la misma búsqueda en validación → `algorithm_comparison.csv`.

**Estado (2026-11-27, JME):** maquinaria de los 4 escenarios de JME en la rama `feat/jme-sensibilidad`
(`run_sensitivity`, `latency_cost`, `run_sensitivity_stage` → `sensitivity.csv`, `tests/test_sensitivity.py`), probada con
las líneas base. Falta correrla con el agente de la 05 y escribir en el reporte las respuestas de capacidad/saturación y
cambio de régimen (texto, con el análisis de la parte 02). Ablación y segundo algoritmo: GC.

## Criterio de terminado

Cada pregunta de §3.10 y las preguntas 6 y 7 tienen una cifra propia y un párrafo; los tres CSV los genera `main.py`.

## Trampas

- Cambiar dos supuestos a la vez y atribuir el efecto a uno.
- Elegir la variante de la ablación viendo prueba (la ablación se reporta, no se usa para reelegir el modelo).

## Preguntas probables

- ¿Cuánto cambia el resultado si el take-profit se ejecuta primero? ¿Y con una barra de retraso?
- ¿Qué familia de variables aporta más? ¿Sobreviviría la ventaja con 50 fondos haciendo lo mismo?
