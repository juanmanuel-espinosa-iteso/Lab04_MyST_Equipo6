"""Pruebas de la parte 05: el replay buffer descuenta con γ^τ (semi-Markov) y la búsqueda solo usa validación."""

import numpy as np
import pytest
from gymnasium import spaces

from code import train

OBS = spaces.Box(-5, 5, shape=(2,), dtype=np.float32)
ACT = spaces.Discrete(3)


def filled_buffer(durations, dones=None, truncated=None, gamma=0.99):
    """Buffer con una transición por duración (un solo entorno)."""
    buffer = train.SemiMarkovReplayBuffer(16, OBS, ACT, device="cpu", gamma=gamma, handle_timeout_termination=True)
    dones = dones or [False] * len(durations)
    truncated = truncated or [False] * len(durations)
    for i, (tau, done, trunc) in enumerate(zip(durations, dones, truncated)):
        obs = np.full((1, 2), i, dtype=np.float32)
        buffer.add(obs, obs + 1, np.array([[1]]), np.array([0.01 * i]), np.array([done]),
                   [{"duration": tau, "TimeLimit.truncated": trunc}])
    return buffer


def test_discount_is_gamma_to_the_duration():
    """Cada transición se descuenta con γ^τ: 1 h de espera → γ; una operación de 24 h → γ^24."""
    buffer = filled_buffer([1.0, 24.0, 7.0], gamma=0.99)
    samples = buffer._get_samples(np.array([0, 1, 2]))
    np.testing.assert_allclose(samples.discounts.numpy().ravel(), 0.99 ** np.array([1.0, 24.0, 7.0]), rtol=1e-6)
    np.testing.assert_allclose(samples.rewards.numpy().ravel(), [0.0, 0.01, 0.02])
    np.testing.assert_allclose(samples.observations.numpy()[:, 0], [0, 1, 2])


def test_terminated_does_not_bootstrap_but_truncated_does():
    """Ruina (`terminated`) corta el bootstrap; el fin de ventana (`truncated`) no."""
    buffer = filled_buffer([3.0, 5.0], dones=[True, True], truncated=[False, True])
    samples = buffer._get_samples(np.array([0, 1]))
    np.testing.assert_array_equal(samples.dones.numpy().ravel(), [1.0, 0.0])


def test_missing_duration_is_an_error():
    """Sin `info["duration"]` no hay γ^τ: se falla en vez de descontar como si fuera 1 h."""
    buffer = train.SemiMarkovReplayBuffer(4, OBS, ACT, device="cpu", gamma=0.99)
    with pytest.raises(KeyError):
        obs = np.zeros((1, 2), dtype=np.float32)
        buffer.add(obs, obs, np.array([[0]]), np.array([0.0]), np.array([False]), [{}])
