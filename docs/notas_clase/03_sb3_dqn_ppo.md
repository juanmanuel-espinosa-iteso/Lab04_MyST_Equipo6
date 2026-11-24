# S19 — Deep RL en la práctica: entorno propio, DQN y PPO con Stable-Baselines3

Resumen del notebook de clase de la sesión 19 (*Gymnasium, SB3, DQN y PPO*). El notebook original no se versiona;
aquí solo está su contenido.

**Para qué sirve en el Lab04:** es la receta de trabajo con Stable-Baselines3 (SB3) y la plantilla del entorno de
trading. La idea central de la sesión: **el algoritmo ya está escrito y probado; lo que define el resultado es el
entorno**, y eso es lo que se evalúa en el Lab04.

## 1. SB3 es el sklearn del RL

| sklearn | SB3 |
|---|---|
| `X, y` (los datos) | `env` (el entorno Gymnasium): **esto lo escribe el equipo** |
| `model = LinearRegression()` | `model = DQN("MlpPolicy", env)` |
| `model.fit(X, y)` | `model.learn(total_timesteps=...)` |
| `model.predict(X)` | `model.predict(obs)` |
| `model.score(X, y)` | `evaluate_policy(model, env)` |
| `LogisticRegression()` | `PPO("MlpPolicy", env)` (cambiar de algoritmo es cambiar una palabra) |

## 2. Las cuatro piezas de un entorno

Todo entorno, sea un juego o un mercado, tiene las mismas piezas:

| Pieza | Qué define | En FrozenLake | En trading (Lab04) |
|---|---|---|---|
| `__init__` | `action_space` y `observation_space` | 4 direcciones; la celda | largo / plano / corto / cerrar; features + posición |
| `reset` | estado inicial del episodio | volver a la celda 0 | inicio de la ventana de datos |
| `step(a)` | la **transición** y la **recompensa** | moverse (con resbalón), +1 en la meta | ejecutar la orden, P&L menos costos |
| `terminated` / `truncated` | cuándo y **por qué** termina | hoyo o meta / 100 pasos | ruina / fin de datos |

**FrozenLake escrito a mano (`MiFrozenLake`):** mapa 4×4, acciones `Discrete(4)` (0 ←, 1 ↓, 2 →, 3 ↑), recompensa 1 al
llegar a la meta menos un `costo_paso` opcional; con resbalón, la dirección real es la elegida o una de sus dos
perpendiculares con probabilidad 1/3 cada una. `terminated` si cae en un hoyo o llega a la meta; `truncated` si se
alcanza `max_pasos` sin terminar.

**Decisión de diseño de la clase:** la observación es un vector `Box` (one-hot de 16 posiciones), no un entero, porque
en trading la observación siempre es un vector de reales.

**Tres detalles que se repiten en cualquier entorno, incluido el de trading:**
1. `super().reset(seed=seed)` crea `self.np_random`. Si se usa `np.random` o `random` por fuera, la semilla no
   controla el entorno. Toda aleatoriedad del entorno (p. ej. el inicio aleatorio del episodio) sale de `self.np_random`.
2. `terminated` y `truncated` son distintos: no hay bootstrap en `terminated` y sí en `truncated` (S18).
3. `info` sirve para diagnosticar (en `MiFrozenLake`: la celda y la dirección real del movimiento). El agente nunca lo usa.

## 3. Validar el entorno antes de entrenar

SB3 entrena sin quejarse un agente perfecto para un simulador roto. Se valida en tres niveles, **los tres exigidos
por el Lab04**:

1. **`check_env`** (`from stable_baselines3.common.env_checker import check_env`; `check_env(env, warn=True)`): que se
   cumpla el contrato (tipos, formas, límites y semillas), sin advertencias.
2. **Operación de referencia:** una trayectoria calculada a mano que el entorno reproduce **exactamente**. En clase:
   sin resbalón, la ruta ↓ ↓ → ↓ → → recorre las celdas 0 → 4 → 8 → 9 → 13 → 14 → 15, llega a la meta en 6 pasos con
   recompensa 1.0, `terminated=True` y `truncated=False`.
3. **Líneas base triviales:** cuánto gana quien no piensa. El agente entrenado tiene que superarlas con claridad. En
   trading: si el agente aleatorio gana dinero, el simulador tiene un error.

## 4. DQN

DQN aproxima $Q(s,a)$ con una red neuronal y aprende minimizando

$$\big(r + \gamma \max_{a'} Q_{\bar\theta}(s',a') - Q_\theta(s,a)\big)^2$$

(Mnih et al., 2015): es el Q-learning de la S18 con una red en lugar de tabla, donde $Q_{\bar\theta}$ es la red objetivo.

```python
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor

env_dqn = Monitor(MiFrozenLake())          # registra la recompensa y la longitud de cada episodio
modelo_dqn = DQN("MlpPolicy", env_dqn, seed=SEED,
                 learning_rate=1e-3, gamma=0.99,
                 buffer_size=20_000, learning_starts=500,
                 policy_kwargs=dict(net_arch=[64, 64]))
modelo_dqn.learn(total_timesteps=60_000)
accion, _ = modelo_dqn.predict(obs, deterministic=True)   # deterministic=True: la mejor acción, sin explorar
```

## 5. PPO

PPO no aprende $Q$: optimiza la **política** directamente y limita cuánto cambia en cada actualización con un objetivo
recortado (Schulman et al., 2017):

$$L^{\text{CLIP}} = \mathbb{E}\Big[\min\big(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t\big)\Big], \qquad \rho_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$$

```python
modelo_ppo = PPO("MlpPolicy", Monitor(MiFrozenLake()), seed=SEED,
                 learning_rate=1e-3, gamma=0.99,
                 n_steps=2048)                       # pasos que juega con la política actual antes de actualizar
modelo_ppo.learn(total_timesteps=60_000)
```

| | DQN | PPO |
|---|---|---|
| Aprende | $Q(s,a)$ | $\pi(a \mid s)$ directamente |
| Datos | off-policy (replay buffer, reutiliza experiencia vieja) | on-policy (usa cada lote y lo tira) |
| Acciones | solo discretas | discretas o continuas |
| Exploración | ε-greedy | política estocástica (`ent_coef`) |

## 6. Evaluar, comparar y guardar

- **Curva de aprendizaje:** recompensa por episodio (media móvil) contra pasos de entrenamiento, con
  `Monitor.get_episode_rewards()` y `np.cumsum(Monitor.get_episode_lengths())`. Es la figura que el Lab04 pide por
  fold (`training_curves_fold_{k}.png`). En clase se comparó contra la $\pi^\star$ de la S17 (≈ 74 % de éxito).
- **Evaluación final:** determinista y con un entorno **nuevo**, no el de entrenamiento. `evaluate_policy` es el
  `model.score` de SB3. En FrozenLake, la política óptima empuja contra las paredes para no resbalar a un hoyo.
- **Guardar y cargar** (como `joblib.dump`): `model.save(...)` y `DQN.load(...)`; el modelo cargado debe dar
  exactamente las mismas acciones.
- **Una semilla es una muestra de tamaño uno:** el mismo código con tres semillas da resultados distintos; se reporta
  la dispersión.

## 7. Del lago congelado al mercado: plantilla para el Lab04

Nada de lo que se usa de SB3 cambia en trading (`check_env`, `Monitor`, `DQN`/`PPO`, `learn`, `predict`,
`evaluate_policy`, `save`). **Lo que cambia es la clase del entorno:**

| FrozenLake | Trading (BTCUSDT horario) |
|---|---|
| `Box` one-hot de la celda | `Box` con features del mercado **hasta t** + estado físico (posición, precio de entrada, tiempo en posición) |
| `Discrete(4)` direcciones | `Discrete(k)`: plano / largo / corto / cerrar (y qué hacer con acciones inválidas) |
| resbalón con `self.np_random` | el mercado: la siguiente barra de los datos; inicio aleatorio con `self.np_random` |
| +1 en la meta − `costo_paso` | P&L **menos 0.125 % por lado** sobre el cambio de posición |
| `terminated`: hoyo o meta | ruina o violación de un límite de riesgo |
| `truncated`: 100 pasos | se acabó la ventana de datos (**nunca** `terminated`) |
| operación de referencia ↓↓→↓→→ | una operación calculada a mano: entrada, salida, duración y P&L exactos |
| líneas base: aleatorio, siempre → | aleatorio, siempre largo, siempre plano, todas con costos |

Esqueleto de la clase (en el Lab04 vive en `code/env.py`, no en `src/env.py`):

```python
class TradingEnv(gym.Env):
    def __init__(self, df, costo=0.00125, ...):
        super().__init__()
        self.action_space = spaces.Discrete(3)          # 0 plano, 1 largo, 2 corto
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_features + 3,), dtype=np.float32)

    def _obs(self):
        # features calculadas SOLO con datos hasta self.t + [posición, precio de entrada, tiempo en posición]
        ...

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = ...                                    # inicio (aleatorio con self.np_random)
        self.posicion, self.equity = 0, 1.0
        return self._obs(), {}

    def step(self, action):
        # 1) ejecutar la orden al precio que realmente se podría obtener
        # 2) pnl = posicion * rendimiento de t a t+1 - costo * |cambio de posición|
        # 3) avanzar el tiempo
        terminated = ...                                # ruina / límite de riesgo
        truncated = ...                                 # fin de datos
        return self._obs(), reward, terminated, truncated, {"equity": self.equity}
```

**Trampa nueva con observaciones continuas:** si se usa `VecNormalize`, los estadísticos de normalización se calculan
**solo** en entrenamiento y se congelan en validación y prueba (`training=False`, copiar `obs_rms`). Si no, es
look-ahead introducido por una librería.

**Diferencia con el Lab04:** la plantilla es de un paso por barra; el Lab04 pide la versión semi-Markov, donde cada
`step` va de la apertura al cierre de una posición y la duración real τ entra al descuento (ver la guía 03 y la 05).

## Para llevar

1. SB3 es el sklearn del RL: `DQN`/`PPO` → `learn` → `predict` → `evaluate_policy`.
2. El trabajo real es el entorno: espacios, `reset`, `step`, recompensa y por qué termina el episodio.
3. Validar antes de entrenar: `check_env`, operación de referencia y líneas base.
4. Evaluación determinista, en datos que no se usaron para entrenar, y con varias semillas.

**Ejercicios de la sesión:** (a) entrenar con `costo_paso=0.01` y ver si cambian la política y los pasos promedio por
episodio; (b) cambiar la observación a un `Box` de 2 valores (fila/3, columna/3) y comparar el aprendizaje (es el
problema de diseñar features); (c) escribir la prueba de la operación de referencia del entorno de trading **antes**
de escribir el entorno.
