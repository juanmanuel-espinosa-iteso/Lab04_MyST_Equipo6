"""Parte 05 — Entrenamiento con Stable-Baselines3: DQN semi-Markov, líneas base, semillas e hiperparámetros.

La transición del entorno dura τ horas de reloj (1 al esperar, de 1 a H o más al operar), así que el objetivo de DQN es

    y = C + (1 − terminado) · γ^τ · max_x' Q̄(S', x'),

con γ por hora. `SemiMarkovReplayBuffer` guarda τ = `info["duration"]` de cada transición y entrega γ^τ como el
descuento por muestra que el `train` de DQN de SB3 ya usa (`replay_data.discounts`). La ruina (`terminated`) corta el
bootstrap; el fin de la ventana (`truncated`) no, porque el mercado sigue.

Parámetros del módulo (abajo): rejilla `GRID`, `SEEDS`, `TOTAL_TIMESTEPS`, hiperparámetros fijos de DQN y criterio de
selección `SELECTION_METRIC` (mediana entre semillas del crecimiento log en validación, fijado antes de ver resultados).
Los pasos de entrenamiento también se eligen en validación: cada corrida se puntúa en los puntos de control
`CHECKPOINTS` (parada temprana), porque con 300k pasos el agente memoriza el train.
"""

import dataclasses
import itertools
import multiprocessing
import pickle
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
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
CHECKPOINTS = (50_000, 100_000, 150_000, 200_000, 250_000, 300_000)  # se puntúa en validación en cada uno
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


# Segundo algoritmo (pregunta 7): PPO de SB3, on-policy. Descuenta con γ por decisión (su GAE no admite τ variable):
# es una aproximación declarada frente al γ^τ del DQN. Rejilla de 8 (red fija en [64, 64]) para caber en el presupuesto.
PPO_SEARCH_SPACE = {
    "learning_rate": (1e-4, 3e-4),
    "gamma": (0.99, 0.995),
    "ent_coef": (0.0, 0.01),
}
PPO_GRID = [dict(zip(PPO_SEARCH_SPACE, values), net_arch=(64, 64))
            for values in itertools.product(*PPO_SEARCH_SPACE.values())]
PPO_FIXED = dict(n_steps=2048, batch_size=64, n_epochs=10, gae_lambda=0.95, clip_range=0.2)
GRIDS = {"dqn": GRID, "ppo": PPO_GRID}


def make_ppo(trading_env, config, seed):
    """PPO de SB3 con la configuración `config` de `PPO_GRID`."""
    return PPO("MlpPolicy", trading_env, seed=seed, verbose=0, device="cpu", learning_rate=config["learning_rate"],
               gamma=config["gamma"], ent_coef=config["ent_coef"], policy_kwargs=dict(net_arch=list(config["net_arch"])),
               **PPO_FIXED)


class _Checkpoints(BaseCallback):
    """Llama `on_checkpoint(model, pasos)` al llegar a cada punto de control y detiene el entrenamiento después del
    último (los calendarios, como el de exploración ε, siguen siendo los del total de pasos)."""

    def __init__(self, checkpoints, on_checkpoint):
        super().__init__()
        self.pending, self.on_checkpoint = sorted(checkpoints), on_checkpoint

    def _on_step(self):
        while self.pending and self.num_timesteps >= self.pending[0]:
            self.on_checkpoint(self.model, self.pending.pop(0))
        return bool(self.pending)


# ---------------------------------------------------------------------- variantes de la observación (ablación)

N_ACCOUNT = len(env.ACCOUNT_FEATURES)  # las 3 variables de la cuenta van al final de la observación
ABLATION_VARIANTS = ["ninguna", *data.FEATURE_FAMILIES, "cuenta"]


class MaskAccount(gym.ObservationWrapper):
    """Pone en 0 las variables de la cuenta de la observación (ablación de la familia `cuenta`)."""

    def observation(self, observation):
        return _mask_account(observation)


def _mask_account(observation):
    masked = np.array(observation, dtype=np.float32, copy=True)
    masked[..., -N_ACCOUNT:] = 0.0
    return masked


def ablate(fold_data, variant):
    """`FoldData` sin las features de la familia `variant` en todas las particiones (la observación se achica).

    Solo cambia la observación: `raw_features` queda igual, así que las barreras siguen usando la volatilidad estimada.
    `ninguna` y `cuenta` (que se quita con `MaskAccount`) no cambian los datos.
    """
    if variant in ("ninguna", "cuenta"):
        return fold_data
    if variant not in data.FEATURE_FAMILIES:
        raise ValueError(f"variante desconocida: {variant}")
    drop = data.FEATURE_FAMILIES[variant]
    return dataclasses.replace(fold_data, features={s: f.drop(columns=drop) for s, f in fold_data.features.items()})


def make_train_env(fold_data, variant="ninguna", reward_name=reward.DEFAULT_REWARD):
    """Entorno de entrenamiento (episodios aleatorios del train) de la variante de la observación."""
    trading_env = env.make_env(ablate(fold_data, variant), "train", reward.REWARDS[reward_name])
    return MaskAccount(trading_env) if variant == "cuenta" else trading_env


def make_model(algo, trading_env, config, seed):
    """Modelo sin entrenar del algoritmo `algo` ("dqn" o "ppo") con una configuración de su rejilla."""
    return {"dqn": make_dqn, "ppo": make_ppo}[algo](trading_env, config, seed)


def train_agent(fold_data, config, seed, total_timesteps=TOTAL_TIMESTEPS, reward_name=reward.DEFAULT_REWARD,
                checkpoints=(), on_checkpoint=None, algo="dqn", variant="ninguna"):
    """Entrena un agente en episodios aleatorios del train del fold. Devuelve (modelo, curva de entrenamiento).

    La curva tiene, por episodio, los pasos acumulados al terminarlo y su recompensa total (= ln del capital del
    episodio, con la contribución principal). `on_checkpoint(model, pasos)` se llama en cada punto de `checkpoints`.
    `algo` es "dqn" (parte 05) o "ppo" (segundo algoritmo); `variant`, la familia que se quita (ablación).
    """
    torch.set_num_threads(1)
    monitored = Monitor(make_train_env(fold_data, variant, reward_name))
    model = make_model(algo, monitored, config, seed)
    callback = _Checkpoints(checkpoints, on_checkpoint) if on_checkpoint else None
    model.learn(total_timesteps=total_timesteps, callback=callback)
    curve = pd.DataFrame({"timesteps": np.cumsum(monitored.get_episode_lengths()),
                          "episode_reward": monitored.get_episode_rewards()})
    return model, curve


def as_policy(model, mask_account=False):
    """Política determinista `obs -> acción` de un modelo de SB3 (la que se evalúa en validación y prueba).

    DQN: la acción de mayor Q̄, llamando directo a la red Q; PPO: la moda de la distribución de la política. Es lo mismo
    que `model.predict(obs, deterministic=True)` (prueba `test_fast_policy_matches_sb3_predict`), sin la preparación por
    observación de SB3, así que los recorridos completos son varias veces más rápidos. Con `mask_account`, la cuenta se
    pone en 0 antes de decidir (ablación `cuenta`).
    """
    model.policy.set_training_mode(False)
    if hasattr(model, "q_net"):
        def decide(obs_t):
            return model.q_net(obs_t).argmax(dim=1)
    else:
        def decide(obs_t):
            return model.policy.get_distribution(obs_t).mode()

    def policy(obs):
        if mask_account:
            obs = _mask_account(obs)
        with torch.no_grad():
            action = decide(torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1))
        return int(action[0])

    return policy


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


def model_path(models_dir, fold, config_id, seed, timesteps):
    """Ruta del modelo de una corrida en un punto de control: `models_dir/fold_{k}/cfg{id}_seed{s}_t{pasos}.zip`."""
    return Path(models_dir) / f"fold_{fold}" / f"cfg{config_id:02d}_seed{seed}_t{timesteps}.zip"


def search_jobs(fold, dates, csv_path, models_dir, config_ids=None, seeds=SEEDS, total_timesteps=TOTAL_TIMESTEPS,
                checkpoints=CHECKPOINTS, algo="dqn"):
    """Corridas (configuración × semilla) de la búsqueda de un fold con el algoritmo `algo`; solo train y validación."""
    config_ids = range(len(GRIDS[algo])) if config_ids is None else config_ids
    return [dict(fold=fold, dates=dates, csv_path=str(csv_path), models_dir=str(models_dir), config_id=c, seed=s,
                 total_timesteps=total_timesteps, checkpoints=tuple(checkpoints), algo=algo, variant="ninguna",
                 score_test=False)
            for c in config_ids for s in seeds]


def ablation_jobs(fold, dates, csv_path, models_dir, config_id, timesteps, seeds=SEEDS, variants=None,
                  total_timesteps=TOTAL_TIMESTEPS):
    """Corridas de la ablación de un fold: la configuración elegida de DQN, con sus pasos, por variante × semilla.

    Se entrena con el calendario de `total_timesteps` (como en la búsqueda) y se detiene en `timesteps`, así que la
    variante `ninguna` reproduce el modelo elegido. Se puntúan también en prueba porque la ablación **se reporta**
    (pregunta 7); no se usa para reelegir el modelo.
    """
    variants = ABLATION_VARIANTS if variants is None else variants
    return [dict(fold=fold, dates=dates, csv_path=str(csv_path), models_dir=str(Path(models_dir) / v),
                 config_id=config_id, seed=s, total_timesteps=total_timesteps, checkpoints=(timesteps,), algo="dqn",
                 variant=v, score_test=True)
            for v in variants for s in seeds]


def run_job(job):
    """Entrena una corrida y, en cada punto de control, guarda el modelo y lo puntúa en validación (y en train solo en el
    último punto; en prueba solo si `score_test`, en la ablación). Devuelve (filas, curva de entrenamiento)."""
    start = time.perf_counter()
    algo, variant = job.get("algo", "dqn"), job.get("variant", "ninguna")
    fold_data = _fold_data(job["csv_path"], job["fold"], job["dates"])
    observed = ablate(fold_data, variant)
    config = GRIDS[algo][job["config_id"]]
    base = {"fold": job["fold"], "algo": algo, "variant": variant, "config_id": job["config_id"], "seed": job["seed"],
            **{k: str(v) if k == "net_arch" else v for k, v in config.items()}}
    rows = []

    def checkpoint(model, timesteps):
        path = model_path(job["models_dir"], job["fold"], job["config_id"], job["seed"], timesteps)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        policy, row = as_policy(model, mask_account=variant == "cuenta"), {**base, "timesteps": timesteps}
        # La validación decide (todos los puntos); el train es informativo (sobreajuste): solo en el último punto.
        last = timesteps == max(job["checkpoints"])
        splits = (("validation", "val"),) + ((("train", "train"),) if last else ())
        splits += (("test", "test"),) if last and job.get("score_test") else ()
        for split, prefix in splits:
            row.update({f"{prefix}_{k}": v for k, v in evaluate_policy(observed, split, policy, job["seed"]).items()})
        rows.append(row)

    _, curve = train_agent(fold_data, config, job["seed"], job["total_timesteps"], checkpoints=job["checkpoints"],
                           on_checkpoint=checkpoint, algo=algo, variant=variant)
    for row in rows:
        row["minutes"] = (time.perf_counter() - start) / 60
    return rows, curve


def hyperparameter_search(jobs, processes=1):
    """Corre las corridas (en paralelo si `processes` > 1). Devuelve (resultados, {(fold, config_id, seed): curva})."""
    if processes > 1:
        with multiprocessing.get_context("spawn").Pool(processes) as pool:
            outputs = pool.map(run_job, jobs, chunksize=1)
    else:
        outputs = [run_job(job) for job in jobs]
    results = pd.DataFrame([row for rows, _ in outputs for row in rows])
    results = results.sort_values(["fold", "config_id", "timesteps", "seed"], ignore_index=True)
    curves = {(job["fold"], job["config_id"], job["seed"]) if job.get("variant", "ninguna") == "ninguna"
              else (job["fold"], job["variant"], job["seed"]): curve for job, (_, curve) in zip(jobs, outputs)}
    return results, curves


def select_config(results, metric=SELECTION_METRIC):
    """Criterio (A), fijado antes de ver resultados: el candidato (configuración, pasos) con mayor **mediana entre
    semillas** de `val_{metric}` (empates → menos pasos y luego menor índice) y, de él, la semilla mediana (no la
    mejor). Devuelve (config_id, timesteps, seed)."""
    column = f"val_{metric}"
    medians = results.groupby(["config_id", "timesteps"])[column].median()
    best = medians[medians == medians.max()].reset_index().sort_values(["timesteps", "config_id"]).iloc[0]
    config_id, timesteps = int(best["config_id"]), int(best["timesteps"])
    runs = results[(results["config_id"] == config_id) & (results["timesteps"] == timesteps)]
    runs = runs.sort_values([column, "seed"])
    return config_id, timesteps, int(runs["seed"].iloc[(len(runs) - 1) // 2])


# ---------------------------------------------------------------------- reusar una búsqueda ya entrenada

ENV_PARAMETERS = ("COST_PER_SIDE", "HORIZON_HOURS", "K_STOP", "K_TAKE", "INTRABAR", "ALLOW_SHORT", "EPISODE_BARS",
                  "RUIN_FRACTION", "INITIAL_EQUITY", "OBS_CLIP", "ACCOUNT_FEATURES")


def _signature(jobs):
    """Lo que define una búsqueda: corridas, rejilla, hiperparámetros fijos, recompensa, parámetros del entorno, features
    y el archivo de datos (su hash). Si cambia cualquiera, los modelos guardados ya no corresponden."""
    import hashlib

    keys = ("fold", "dates", "config_id", "seed", "total_timesteps", "checkpoints", "algo", "variant", "score_test")
    csv_files = sorted({job["csv_path"] for job in jobs})
    return {"jobs": [tuple(str(job[k]) for k in keys) for job in jobs], "grid": str(GRIDS), "dqn_fixed": str(DQN_FIXED),
            "ppo_fixed": str(PPO_FIXED),
            "reward": reward.DEFAULT_REWARD, "env": {name: str(getattr(env, name)) for name in ENV_PARAMETERS},
            "features": list(data.FEATURES),
            "data": {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in csv_files}}


def save_search(path, jobs, results, curves):
    """Guarda resultados y curvas de una búsqueda junto a los modelos (no se versiona) para poder reusarla."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(pickle.dumps({"signature": _signature(jobs), "results": results, "curves": curves}))


def load_search(path, jobs):
    """(resultados, curvas) de una búsqueda guardada **solo si** se hizo con las mismas corridas y parámetros; si no
    existe o cambió algo, None (hay que entrenar)."""
    path = Path(path)
    if not path.exists():
        return None
    saved = pickle.loads(path.read_bytes())
    if saved["signature"] != _signature(jobs):
        return None
    return saved["results"], saved["curves"]


# ---------------------------------------------------------------------- tablas de la pregunta 7

def ablation_table(results):
    """`ablation.csv`: por fold y variante, la mediana entre semillas del crecimiento log en validación y en prueba, el
    Sharpe y las operaciones en prueba, y la diferencia contra la base (`ninguna`); al final, la media entre folds.

    Un `delta_val` negativo dice que quitar la familia empeora la validación: la familia aporta."""
    cols = ["val_log_growth", "test_log_growth", "test_sharpe", "test_n_trades"]
    table = results.groupby(["fold", "variant"])[cols].median()
    folds = table.index.get_level_values("fold")
    base = table.xs("ninguna", level="variant")
    table["delta_val"] = table["val_log_growth"] - base["val_log_growth"].reindex(folds).to_numpy()
    table["delta_test"] = table["test_log_growth"] - base["test_log_growth"].reindex(folds).to_numpy()
    variants = [v for v in ABLATION_VARIANTS if v in table.index.get_level_values("variant")]
    table = table.reindex(pd.MultiIndex.from_product([sorted(set(folds)), variants], names=["fold", "variant"]))
    mean = table.groupby(level="variant").mean().reindex(variants)
    mean.index = pd.MultiIndex.from_product([["media"], variants], names=["fold", "variant"])
    return pd.concat([table, mean])
