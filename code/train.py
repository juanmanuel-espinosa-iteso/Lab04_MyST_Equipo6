"""Parte 05 — Entrenamiento con Stable-Baselines3: DQN semi-Markov, líneas base, semillas e hiperparámetros.

La transición del entorno dura τ horas de reloj (1 al esperar, de 1 a H o más al operar), así que el objetivo de DQN es

    y = C + (1 − terminado) · γ^τ · max_x' Q̄(S', x'),

con γ por hora. `SemiMarkovReplayBuffer` guarda τ = `info["duration"]` de cada transición y entrega γ^τ como el
descuento por muestra que el `train` de DQN de SB3 ya usa (`replay_data.discounts`). La ruina (`terminated`) corta el
bootstrap; el fin de la ventana (`truncated`) no, porque el mercado sigue.

Parámetros del módulo (abajo): rejilla `GRID`, `SEEDS`, `TOTAL_TIMESTEPS`, hiperparámetros fijos de DQN y criterio de
selección `SELECTION_METRIC` (mediana entre semillas del crecimiento log en validación, fijado antes de ver resultados).
"""

import itertools
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from code import data, env, reward


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


# ---------------------------------------------------------------------- rejilla y entrenamiento de una corrida

TOTAL_TIMESTEPS = 300_000  # decisiones por corrida (≈ 10 recorridos del train del fold 1)
SEEDS = (0, 1, 2)
SELECTION_METRIC = "log_growth"  # criterio (A): mediana entre semillas en validación
SEARCH_SPACE = {
    "learning_rate": (1e-4, 5e-4),
    "gamma": (0.99, 0.995),  # por hora: vida media de ≈ 69 h y ≈ 138 h
    "net_arch": ((64, 64), (128, 128)),
    "exploration_fraction": (0.1, 0.3),
}
GRID = [dict(zip(SEARCH_SPACE, values)) for values in itertools.product(*SEARCH_SPACE.values())]
DQN_FIXED = dict(buffer_size=100_000, learning_starts=5_000, batch_size=64, train_freq=4, gradient_steps=1,
                 target_update_interval=2_000, exploration_final_eps=0.05)


def make_dqn(trading_env, config, seed):
    """DQN de SB3 con el buffer semi-Markov (γ^τ) y la configuración `config` de la rejilla."""
    return DQN("MlpPolicy", trading_env, seed=seed, verbose=0, device="cpu",
               learning_rate=config["learning_rate"], gamma=config["gamma"],
               exploration_fraction=config["exploration_fraction"],
               policy_kwargs=dict(net_arch=list(config["net_arch"])),
               replay_buffer_class=SemiMarkovReplayBuffer, replay_buffer_kwargs=dict(gamma=config["gamma"]),
               **DQN_FIXED)


def train_agent(fold_data, config, seed, total_timesteps=TOTAL_TIMESTEPS, reward_name=reward.DEFAULT_REWARD):
    """Entrena un DQN en episodios aleatorios del train del fold. Devuelve (modelo, curva de entrenamiento).

    La curva tiene, por episodio, los pasos acumulados al terminarlo y su recompensa total (= ln del capital del
    episodio, con la contribución principal).
    """
    torch.set_num_threads(1)
    monitored = Monitor(env.make_env(fold_data, "train", reward.REWARDS[reward_name]))
    model = make_dqn(monitored, config, seed)
    model.learn(total_timesteps=total_timesteps)
    curve = pd.DataFrame({"timesteps": np.cumsum(monitored.get_episode_lengths()),
                          "episode_reward": monitored.get_episode_rewards()})
    return model, curve


def as_policy(model):
    """Política determinista `obs -> acción` de un modelo de SB3 (la que se evalúa en validación y prueba)."""
    return lambda obs: int(model.predict(obs, deterministic=True)[0])


# ---------------------------------------------------------------------- búsqueda en validación y selección

_CACHE = {}  # por proceso: datos y folds ya construidos


def _fold_data(csv_path, fold, dates):
    """`FoldData` del fold, construido una vez por proceso (el normalizador se ajusta solo con su train)."""
    if "base" not in _CACHE:
        ohlcv, _ = data.load_ohlcv(csv_path)
        _CACHE["base"] = (ohlcv, data.compute_features(ohlcv))
    if fold not in _CACHE:
        _CACHE[fold] = data.build_fold_dataset(*_CACHE["base"], fold, dates)
    return _CACHE[fold]


def model_path(models_dir, fold, config_id, seed):
    """Ruta del modelo de una corrida: `models_dir/fold_{k}/cfg{id}_seed{s}.zip`."""
    return Path(models_dir) / f"fold_{fold}" / f"cfg{config_id:02d}_seed{seed}.zip"


def search_jobs(fold, dates, csv_path, models_dir, config_ids=None, seeds=SEEDS, total_timesteps=TOTAL_TIMESTEPS):
    """Corridas (configuración × semilla) de la búsqueda de un fold."""
    config_ids = range(len(GRID)) if config_ids is None else config_ids
    return [dict(fold=fold, dates=dates, csv_path=str(csv_path), models_dir=str(models_dir), config_id=c, seed=s,
                 total_timesteps=total_timesteps) for c in config_ids for s in seeds]


def run_job(job):
    """Entrena una corrida, guarda el modelo y la puntúa en train y validación (nunca en prueba)."""
    start = time.perf_counter()
    fold_data = _fold_data(job["csv_path"], job["fold"], job["dates"])
    config = GRID[job["config_id"]]
    model, curve = train_agent(fold_data, config, job["seed"], job["total_timesteps"])
    path = model_path(job["models_dir"], job["fold"], job["config_id"], job["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    policy = as_policy(model)
    row = {"fold": job["fold"], "config_id": job["config_id"], "seed": job["seed"],
           **{k: str(v) if k == "net_arch" else v for k, v in config.items()}}
    for split, prefix in (("train", "train"), ("validation", "val")):
        row.update({f"{prefix}_{k}": v for k, v in evaluate_policy(fold_data, split, policy, job["seed"]).items()})
    row["minutes"] = (time.perf_counter() - start) / 60
    return row, curve


def hyperparameter_search(jobs, processes=1):
    """Corre las corridas (en paralelo si `processes` > 1). Devuelve (resultados, {(config_id, seed): curva})."""
    if processes > 1:
        with multiprocessing.get_context("spawn").Pool(processes) as pool:
            outputs = pool.map(run_job, jobs, chunksize=1)
    else:
        outputs = [run_job(job) for job in jobs]
    results = pd.DataFrame([row for row, _ in outputs]).sort_values(["fold", "config_id", "seed"], ignore_index=True)
    curves = {(row["config_id"], row["seed"]): curve for row, curve in outputs}
    return results, curves


def select_config(results, metric=SELECTION_METRIC):
    """Criterio (A), fijado antes de ver resultados: la configuración con mayor **mediana entre semillas** de
    `val_{metric}` (empates → menor índice) y, de ella, la semilla mediana (no la mejor). Devuelve (config_id, seed)."""
    column = f"val_{metric}"
    medians = results.groupby("config_id")[column].median()
    config_id = int(medians[medians == medians.max()].index.min())
    runs = results[results["config_id"] == config_id].sort_values([column, "seed"])
    return config_id, int(runs["seed"].iloc[(len(runs) - 1) // 2])
