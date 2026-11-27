"""Parte 04 — Función de contribución C(S, x, W) del entorno semi-Markov (`code/env.py`).

El entorno decide solo estando plano y paga la contribución **al cierre** de cada decisión (terminal): una espera dura
una barra y una operación dura de la decisión a la salida, τ horas de reloj. La contribución principal es el
rendimiento log neto de la operación, con todo el capital y 0.125 % por lado sobre el nocional de entrada y el de salida:

    C(S, x, W) = ln(1 + s · (P_sal − P_ent) / P_ent − c · (P_ent + P_sal) / P_ent),   s ∈ {+1, −1},  c = 0.00125
    C(S, esperar, W) = 0

Como todo el capital entra en cada operación, Σ C = ln(capital final / capital inicial): maximizar la suma esperada de
contribuciones es maximizar el crecimiento log del capital, lo mismo que se evalúa, sin coeficientes de moldeo. En una
operación de 24 h queda en el orden de 10⁻². Esperar vale 0: el descuento γ^τ ya cobra el tiempo plano.

Alternativas del registro, para compararlas con las mismas operaciones (`docs/decisiones.md`):
- `vol_scaled_return`: el log neto entre σ̂ · √H (σ̂ = volatilidad horaria en la decisión); Σ C deja de ser el capital.
- `gross_log_return`: sin costos; operar al azar no pierde, así que premia rotar.

Todas son funciones puras `TradeOutcome -> float`.
"""

import numpy as np

from code import env

VOL_FLOOR = 1e-8  # evita dividir entre 0 si la volatilidad estimada es nula


def net_log_return(outcome):
    """Contribución principal: ln(1 + rendimiento neto de costos) de la operación; 0 al esperar."""
    if outcome.side == 0:
        return 0.0
    return float(np.log1p(outcome.net_return))


def vol_scaled_return(outcome, horizon_hours=env.HORIZON_HOURS):
    """Alternativa B: log neto / (σ̂ · √H), la escala de las barreras; 0 al esperar."""
    if outcome.side == 0:
        return 0.0
    return net_log_return(outcome) / (max(outcome.entry_vol, VOL_FLOOR) * np.sqrt(horizon_hours))


def gross_log_return(outcome):
    """Alternativa C: ln(1 + rendimiento bruto), **sin costos**; 0 al esperar."""
    if outcome.side == 0:
        return 0.0
    return float(np.log1p(outcome.gross_return))


REWARDS = {
    "net_log_return": net_log_return,
    "vol_scaled_return": vol_scaled_return,
    "gross_log_return": gross_log_return,
}
DEFAULT_REWARD = "net_log_return"
