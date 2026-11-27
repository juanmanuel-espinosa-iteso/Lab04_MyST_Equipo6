"""Parte 05 — Entrenamiento con Stable-Baselines3: DQN semi-Markov, líneas base, semillas e hiperparámetros.

La transición del entorno dura τ horas de reloj (1 al esperar, de 1 a H o más al operar), así que el objetivo de DQN es

    y = C + (1 − terminado) · γ^τ · max_x' Q̄(S', x'),

con γ por hora. `SemiMarkovReplayBuffer` guarda τ = `info["duration"]` de cada transición y entrega γ^τ como el
descuento por muestra que el `train` de DQN de SB3 ya usa (`replay_data.discounts`). La ruina (`terminated`) corta el
bootstrap; el fin de la ventana (`truncated`) no, porque el mercado sigue.
"""

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


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
