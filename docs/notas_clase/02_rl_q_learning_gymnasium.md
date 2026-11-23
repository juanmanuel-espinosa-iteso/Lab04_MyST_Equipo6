# S18 — RL sin modelo: Q-learning y Gymnasium

Resumen de la sesión 18 del módulo de Reinforcement Learning (diapositivas *Model-Free RL: Q-Learning & Gymnasium*
y notebook de la sesión). Material original del Prof. Luis Felipe Gómez Estrada, adaptado por el Prof. Luis Carlos
Alvarado Garnica. Referencias de la sesión: Sutton y Barto (2018), caps. 5–7; Watkins y Dayan (1992); documentación
de Gymnasium.

**Para qué sirve en el Lab04:** es la fuente de notación para Q-learning, el contrato `reset`/`step`, la diferencia
`terminated`/`truncated` y la validación del entorno antes del agente (líneas base, `check_env`). La formulación de
Powell está en `01_sequential_decision_analytics.md`.

## 1. El obstáculo: aprender sin modelo

Value iteration (S17) necesita el modelo de transición, que en un mercado no tenemos:

$$V(s) \leftarrow \max_a \underbrace{\sum_{s'} P(s' \mid s,a)}_{\text{desconocido}} \big[R(s,a) + \gamma V(s')\big]$$

- **Idea:** reemplazar la esperanza por **muestras**. No se puede calcular $\sum_{s'}P(s'\mid s,a)f(s')$, pero sí
  observar transiciones reales $(s,a,r,s')$ y promediar sobre ellas. Es integración Monte Carlo aplicada a
  programación dinámica, y resuelve directamente la tercera maldición de Powell (el espacio de resultados).
- **Precio:** las muestras son ruidosas y se necesitan muchas. La **eficiencia muestral** es la preocupación
  práctica central de todo lo que sigue.

## 2. Monte Carlo contra diferencias temporales (TD)

**Monte Carlo:** se juega un episodio completo, se observa el retorno real $G_t$ y se mueve la estimación hacia él:

$$V(s_t) \leftarrow V(s_t) + \alpha\big[G_t - V(s_t)\big]$$

- Insesgado ($G_t$ es una muestra genuina de $\mathbb{E}[G_t]$), pero de varianza alta (suma de muchas recompensas aleatorias).
- **Requiere que los episodios terminen.** Un mercado continuo no tiene fin natural de episodio: hay que imponerlo.
- No hace bootstrap: no propaga errores propios, pero tampoco reutiliza lo que ya sabe.

**TD(0):** no se espera al final del episodio; se usa la relación de Bellman $V(s_t)=\mathbb{E}[r_{t+1}+\gamma V(s_{t+1})]$
y se sustituye la cola por la estimación actual:

$$V(s_t) \leftarrow V(s_t) + \alpha\big[\underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{objetivo TD}} - V(s_t)\big]$$

El corchete es el **error TD** $\delta_t$.

| | Monte Carlo | TD(0) |
|---|---|---|
| Objetivo | $G_t$ (retorno real) | $r_{t+1}+\gamma V(s_{t+1})$ |
| Sesgo | Ninguno | Sesgado (usa su propia estimación) |
| Varianza | Alta | Baja |
| Actualiza | Al final del episodio | En cada paso |
| Necesita terminación | Sí | No |

**Bootstrapping (el trade-off central).** Bootstrapping = actualizar una estimación con otra estimación (TD lo hace,
MC no).
- *Por qué TD suele ganar:* menor varianza, actualización en línea y aprovecha la estructura de Markov (cada
  transición enseña algo a todos los estados que llevan a ella). En datos financieros ruidosos, donde $G_t$ sobre
  250 pasos varía muchísimo, la reducción de varianza es decisiva.
- *Por qué puede fallar:* el sesgo se acumula; si $V(s_{t+1})$ está mal, el error entra en $V(s_t)$ y en sus
  predecesores. Junto con aproximación de funciones y aprendizaje off-policy forma la **tríada mortal** (S19).
- TD(λ) interpola: λ = 0 es TD(0) y λ = 1 es Monte Carlo.

## 3. Control: SARSA y Q-learning

**SARSA (on-policy):**

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\big[r_{t+1} + \gamma\, Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)\big]$$

donde $a_{t+1}$ es la acción que el agente **realmente tomó** (el nombre viene de la quíntupla
$s_t,a_t,r_{t+1},s_{t+1},a_{t+1}$). Aprende el valor de la política que sigue, **incluida su exploración**: si explora
al azar 10 % del tiempo, sus valores incluyen el costo de esos errores y aprende una política cautelosa.

**Q-learning (off-policy, Watkins y Dayan, 1992):**

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\big[\underbrace{r_{t+1} + \gamma \max_{a'} Q(s_{t+1},a')}_{\text{objetivo TD}} - Q(s_t,a_t)\big]$$

- Un símbolo de diferencia: SARSA usa $Q(s_{t+1},a_{t+1})$; Q-learning usa $\max_{a'}Q(s_{t+1},a')$. Q-learning
  evalúa la política **greedy** mientras se **comporta** con ε-greedy.
- Sin modelo (no requiere $P$ ni $R$). Off-policy: aprende $Q^\star$ sin importar cómo se recolectaron los datos,
  lo que hace legítimo el *experience replay* (S19). Converge a $Q^\star$ con probabilidad 1 si cada par
  $(s,a)$ se visita infinitas veces y la tasa de aprendizaje decae adecuadamente.
- **Exploración ε-greedy:** con probabilidad ε una acción al azar; si no, $\arg\max_a Q(s,a)$. ε decae
  multiplicativamente hasta un mínimo.

**Intuición del acantilado (cliff walking):** Q-learning aprende el camino óptimo pegado al borde y cae seguido
mientras explora (mejor política, peor desempeño en línea); SARSA aprende un camino más seguro. **Traducción a
trading:** si se explora en un simulador y solo se despliega la política final, Q-learning (se quiere $Q^\star$);
si el agente aprende operando dinero real, la cautela de SARSA es una ventaja, porque caer del acantilado cuesta P&L.

**Efecto de α y γ** (experimento del notebook en FrozenLake):
- α chica: estable pero lenta; α grande: rápida pero ruidosa (una sola muestra puede engañar mucho).
- γ chica: agente miope que no "ve" la meta desde lejos.
- La curva de entrenamiento queda por debajo de la política final porque mientras entrena sigue explorando con ε.
  Las diferencias con $Q^\star$ aparecen en estados poco visitados.

## 4. Implementación tabular y el bug de `terminated`

```python
for ep in range(episodes):
    s, _ = env.reset()
    done, total = False, 0.0
    while not done:
        a = rng.integers(n_actions) if rng.random() < eps else int(Q[s].argmax())
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        target = r + (0.0 if terminated else gamma * Q[s2].max())
        Q[s, a] += alpha * (target - Q[s, a])
        s, total = s2, total + r
    eps = max(eps_min, eps * decay)
```

**Al terminar hay que quitar el bootstrap.** Olvidarlo es el bug más común de RL tabular y corrompe en silencio
todas las estimaciones. En el notebook, con inicialización optimista $Q_0 = 1$, el bug "filtra" el valor de los
estados terminales hacia atrás y aparecen valores mayores a 1, imposibles. El código corre y parece que aprende.

## 5. El contrato de Gymnasium

| Método | Devuelve |
|---|---|
| `env.reset(seed=...)` | `obs, info` |
| `env.step(action)` | `obs, reward, terminated, truncated, info` |

- `obs`: la nueva observación. `reward`: recompensa escalar de la transición.
- `terminated`: el episodio acabó por una razón de la **tarea** (quiebra, meta). El futuro vale 0: **no** se hace bootstrap.
- `truncated`: acabó por un **límite de tiempo** (fin de datos, máximo de pasos). El futuro existe: **sí** se hace bootstrap.
- `info`: diagnósticos; **el agente nunca lo usa** (en FrozenLake, `info["prob"]` sería espiar el modelo).

El mismo código de agente corre sobre cualquier entorno que cumpla el contrato.

**Objetivo correcto:**

$$\text{objetivo} = \begin{cases} r & \text{si terminated} \\ r + \gamma \max_{a'} Q(s',a') & \text{si truncated o en curso} \end{cases}$$

**Por qué importa en trading:** si el fin de la serie de precios se marca como `terminated`, se le enseña al agente
que el mundo se acaba con valor futuro cero en la última barra de cada episodio, y aprende a liquidar cerca del
final por razones que no tienen nada que ver con el mercado. En un entorno de trading, `terminated` debe significar
algo real: capital por debajo de un umbral de ruina o la violación de un límite duro de riesgo. El fin de datos es `truncated`.

## 6. Un entorno de trading mínimo

`TradingEnv(gym.Env)` de la clase (precios sintéticos, semilla 42):
- **Acciones:** `Discrete(3)`: 0 corto, 1 fuera, 2 largo → posición $\in\{-1,0,+1\}$.
- **Observación:** `Box(-inf, inf)` con los últimos `window` rendimientos (hasta $t-1$) más la posición actual.
- **Recompensa:** $\text{pnl}_t = \text{pos}_t\,(p_{t+1}-p_t) - c\,\lvert\Delta \text{pos}\rvert\, p_t$, dividida entre el capital inicial (escala libre).
- `terminated` = ruina (equity < 50 % del inicial); `truncated` = se acabaron los datos.

**Tres cosas que verificar en un entorno propio:**
1. La observación en $t$ usa solo barras hasta $t-1$.
2. La recompensa usa $p_{t+1}-p_t$ con la posición ya tomada **antes** del movimiento.
3. El costo se cobra sobre el **cambio** de posición, no sobre la posición.

**Diseñar bien el entorno:**
- *Observación:* normalizar todo (rendimientos, no precios; z-scores con estadísticos móviles hacia atrás).
  Incluir la posición: sin ella el problema no es de Markov. Considerar P&L no realizado y tiempo en posición.
- *Episodios:* aleatorizar el índice de inicio en cada `reset` para que el agente no memorice una trayectoria.
  Episodios suficientemente largos para contener varias operaciones y suficientemente cortos para dar muchas
  actualizaciones de gradiente.
- *Escala de la recompensa:* dividir entre el capital inicial o entre la volatilidad móvil. Recompensas del orden
  de $10^3$ desestabilizan el entrenamiento neuronal; del orden de $10^{-2}$ están bien.

**Probar el entorno antes que el agente:**
1. `gymnasium.utils.env_checker.check_env` revisa el contrato (tipos, espacios, semillas).
2. Correr un agente aleatorio y uno de comprar y mantener. El retorno del episodio de comprar y mantener debe
   ser igual al retorno del activo en esa ventana (menos el costo de entrar), con tolerancia de punto flotante.
   Si no lo es, el entorno está mal, y ningún agente lo arregla.

## 7. Por qué la tabla no escala a trading

Un estado modesto (5 rendimientos rezagados, RSI, ATR, z-score de volumen y posición), discretizado en 10 bins
por variable, da $10^8$ estados × 3 acciones = $3\times10^8$ entradas. Para visitar cada par una sola vez harían
falta más barras de las que existen.
1. **Memoria:** la tabla no cabe.
2. **Generalización:** dos estados de mercado casi idénticos son entradas independientes; lo que el agente aprende
   en una no le enseña nada sobre la otra.

La solución es una función $Q_\theta(s,a)$ que interpole entre estados similares: la red neuronal de DQN (S19).

## Para llevar

1. Sin modelo = cambiar la esperanza de Bellman por muestras $(s,a,r,s')$.
2. Q-learning usa $\max$ → off-policy → aprende $Q^\star$ aunque explore.
3. SARSA usa la acción tomada → on-policy → cauteloso.
4. `terminated` sin bootstrap; `truncated` con bootstrap.
5. Validar el entorno (aleatorio y comprar y mantener) antes de entrenar.
