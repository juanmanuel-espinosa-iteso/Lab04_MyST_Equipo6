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
import pandas as pd

from code import env

VOL_FLOOR = 1e-8  # evita dividir entre 0 si la volatilidad estimada es nula


def net_log_return(outcome):
    """Contribución principal: ln(1 + rendimiento neto de costos) de la operación; 0 al esperar."""
    if outcome.side == 0:
        return 0.0
    return float(np.log1p(outcome.net_return))


def vol_scaled_return(outcome, horizon_hours=env.HORIZON_HOURS):
    """Alternativa B: log neto / (σ̂ · √H), la escala de las barreras; 0 al esperar.

    σ̂ = 0 se acota a `VOL_FLOOR` (queda finita pero enorme; en los datos σ̂ ≥ 0.0011); σ̂ no finita es un error.
    """
    if outcome.side == 0:
        return 0.0
    if not np.isfinite(outcome.entry_vol):
        raise ValueError(f"volatilidad en la entrada no finita: {outcome.entry_vol}")
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


# ---------------------------------------------------------------------- evidencia para elegir la contribución

POLICIES = ("aleatorio", "siempre_largo", "siempre_corto")


def collect_outcomes(fold_data, split, policy, seed):
    """`TradeOutcome` de cada decisión de un episodio de evaluación completo de la partición.

    `policy`: "aleatorio" (acción uniforme entre esperar, largo y corto con `np.random.default_rng(seed)`),
    "siempre_largo" o "siempre_corto". El entorno recibe una contribución que solo guarda el resultado, así que después
    se le puede aplicar cualquier contribución a **las mismas** operaciones.
    """
    outcomes = []

    def record(outcome):
        outcomes.append(outcome)
        return 0.0

    rng = np.random.default_rng(seed)
    fixed = {"siempre_largo": env.LONG, "siempre_corto": env.SHORT}
    trading_env = env.make_env(fold_data, split, record, mode="eval")
    trading_env.reset(seed=seed)
    done = False
    while not done:
        action = int(rng.integers(trading_env.action_space.n)) if policy == "aleatorio" else fixed[policy]
        _, _, terminated, truncated, _ = trading_env.step(action)
        done = terminated or truncated
    return outcomes


def reward_comparison(fold_data, split="train", seed=42):
    """Contribución media por decisión y por operación, y total, de cada política con cada contribución de `REWARDS`.

    Índice (policy, reward). Columnas: `decisions`, `trades`, `mean_per_decision`, `mean_per_trade`, `total`.
    """
    rows = {}
    for policy in POLICIES:
        outcomes = collect_outcomes(fold_data, split, policy, seed)
        trades = [o for o in outcomes if o.side != 0]
        for name, fn in REWARDS.items():
            values = np.array([fn(o) for o in outcomes])
            rows[(policy, name)] = {
                "decisions": len(outcomes), "trades": len(trades),
                "mean_per_decision": values.mean(),
                "mean_per_trade": np.mean([fn(o) for o in trades]) if trades else np.nan,
                "total": values.sum(),
            }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["policy", "reward"])


def per_bar_contributions(entry_price, closes, exit_price, side, cost=env.COST_PER_SIDE):
    """La misma operación pagada **por barra**: log del cambio del capital marcado a mercado en cada barra.

    `closes` son los cierres de las barras en posición antes de la salida. El costo de entrada cae en la primera barra
    y el de salida en la última, así que la suma es la contribución terminal; solo cambia **cuándo** se paga (y, con
    descuento por barra, cuánto pesa cada parte).
    """
    marks = 1 - cost + side * (np.asarray(closes, dtype=float) - entry_price) / entry_price
    final = 1 - cost * (entry_price + exit_price) / entry_price + side * (exit_price - entry_price) / entry_price
    equity = np.concatenate([[1.0], marks, [final]])
    return np.diff(np.log(equity))
