"""Parte 05 — Entrenamiento con Stable-Baselines3: DQN semi-Markov, líneas base, semillas e hiperparámetros.

La transición del entorno dura τ horas de reloj (1 al esperar, de 1 a H o más al operar), así que el objetivo de DQN es

    y = C + (1 − terminado) · γ^τ · max_x' Q̄(S', x'),

con γ por hora. `SemiMarkovReplayBuffer` guarda τ = `info["duration"]` de cada transición y entrega γ^τ como el
descuento por muestra que el `train` de DQN de SB3 ya usa (`replay_data.discounts`). La ruina (`terminated`) corta el
bootstrap; el fin de la ventana (`truncated`) no, porque el mercado sigue.
"""

import numpy as np
import pandas as pd
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from code import env, reward


class SemiMarkovReplayBuffer(ReplayBuffer):
    """Replay buffer de SB3 que además guarda la duración τ de cada transición y devuelve γ^τ como descuento."""

    def __init__(self, *args, gamma=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.durations = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        """Como `ReplayBuffer.add`, más τ de `info["duration"]` (obligatoria: sin ella no hay γ^τ)."""
        self.durations[self.pos] = np.array([info["duration"] for info in infos], dtype=np.float32)
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        """Muestras de `ReplayBuffer` con `discounts = γ^τ` (mismo índice de entorno que el resto de la muestra)."""
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env),
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            (self.gamma ** self.durations[batch_inds, env_indices]).reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


# ---------------------------------------------------------------------- recorridos, puntaje y líneas base

SPLITS = ("train", "validation", "test")
DAYS_PER_YEAR = 365  # BTC opera 24/7


def rollout(trading_env, policy, seed):
    """Un recorrido completo de evaluación con `policy(obs) -> acción`; devuelve (capital horario, bitácora)."""
    obs, _ = trading_env.reset(seed=seed)
    done = False
    while not done:
        obs, _, terminated, truncated, _ = trading_env.step(policy(obs))
        done = terminated or truncated
    return trading_env.equity_series(), trading_env.trade_log()


def score(equity):
    """Puntaje de un recorrido: `log_growth` = ln(capital final / inicial), el criterio de selección (A), y el Sharpe
    diario neto (informativo; 365 días, tasa 0), con el primer día medido desde el capital inicial."""
    daily = equity.resample("D").last().ffill()
    daily = pd.concat([pd.Series([env.INITIAL_EQUITY], index=[daily.index[0] - pd.Timedelta(days=1)]), daily])
    r = daily.pct_change().dropna()
    sd = r.std()
    return {"log_growth": float(np.log(equity.iloc[-1] / env.INITIAL_EQUITY)),
            "sharpe": float(r.mean() / sd * np.sqrt(DAYS_PER_YEAR)) if sd > 0 else np.nan}


def baseline_policies(seed):
    """Líneas base triviales (§3.5): aleatorio (generador propio), siempre largo y siempre plano."""
    rng = np.random.default_rng(seed)
    return {"aleatorio": lambda obs: int(rng.integers(3)),
            "siempre_largo": lambda obs: env.LONG,
            "siempre_plano": lambda obs: env.WAIT}


def evaluate_policy(fold_data, split, policy, seed, reward_name=reward.DEFAULT_REWARD):
    """Puntaje y número de operaciones de `policy` en un recorrido completo de la partición."""
    trading_env = env.make_env(fold_data, split, reward.REWARDS[reward_name], mode="eval")
    equity, log = rollout(trading_env, policy, seed)
    return {**score(equity), "n_trades": len(log)}


def run_baselines(fold_data, seed):
    """Líneas base con costos en train, validación y prueba del fold → filas (split, policy) de `baselines.csv`.

    Tocar la prueba con políticas triviales no fija nada del agente (no tienen parámetros)."""
    rows = {}
    for split in SPLITS:
        for name, policy in baseline_policies(seed).items():
            rows[(split, name)] = evaluate_policy(fold_data, split, policy, seed)
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["split", "policy"])
