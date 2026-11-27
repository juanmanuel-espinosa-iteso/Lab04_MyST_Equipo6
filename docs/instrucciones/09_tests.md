# 09 — Pruebas (pytest)

**Dueños:** cada quien las de su parte (ver `00_flujo_de_trabajo.md`) · **Revisor:** cruzado · **Carpeta:** `tests/`
**Leer antes:** `docs/Lab04.pdf` §3.11 · `docs/notas_clase/03_sb3_dqn_ppo.md` (§3, validación en tres niveles)

## Objetivo

Al menos las cuatro pruebas obligatorias, que corran con `pytest` desde la raíz y pasen. Deben ser rápidas (datos
sintéticos o un tramo corto del CSV), sin entrenar modelos.

## Checklist

- [x] **1. `test_env_check.py`** (JME): el entorno pasa `check_env` **sin advertencias** (`warnings` como error).
- [x] **2. `test_causality.py`** (JME): reconstruir la observación con los datos hasta el índice i devuelve el mismo
      vector que el cálculo sobre la serie completa (varios i, incluido el normalizador del fold).
- [x] **3. `test_reference_trade.py`** (JME): trayectoria corta calculada a mano con razón de salida, duración y P&L
      conocidos, reproducida exactamente; un caso por razón de salida y el empate intrabarra.
- [ ] **4. `test_accounting.py`** (JME): capital final = capital inicial + Σ P&L neto de la bitácora.
- [ ] `test_reward.py` (GC): contribución de una operación conocida = cálculo a mano.
- [ ] Pruebas extra útiles (GC): folds sin traslape y en orden; normalizador ajustado solo con train; métricas contra valores a mano.
- [ ] `pytest` en verde desde un clon limpio.

## Criterio de terminado

`pytest` pasa completo en menos de un minuto y cada prueba obligatoria tiene un docstring que dice qué garantiza.
