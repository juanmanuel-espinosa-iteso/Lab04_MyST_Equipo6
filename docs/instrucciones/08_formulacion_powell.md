# 08 — Formulación como problema de decisión secuencial (Powell)

**Dueños:** JME (estado, decisión, información exógena, transición y gestor de posiciones) · GC (contribución, objetivo
y política) · **Revisor:** cruzado · **Entregable:** sección del reporte (`docs/reporte.pdf`) y `docs/powell.md`
**Leer antes:** `docs/notas_clase/01_sequential_decision_analytics.md` (notación) · `docs/Lab04.pdf` §3.2

## Objetivo

Formulación **explícita y completa** con la notación estándar de clase (las formulaciones vagas o parciales tienen una
penalización considerable en el criterio de lógica del modelo). Cada elemento se enlaza con la función que lo implementa.

## Checklist

- [ ] **Estado S_t** con la descomposición explícita:
  - [ ] física R_t: posición, precio de entrada, tiempo en posición, efectivo;
  - [ ] información I_t: lo observado del mercado (features de la parte 01);
  - [ ] creencia B_t: lo estimado, con su incertidumbre.
- [ ] **Decisión x_t:** espacio de acciones y tratamiento de acciones inválidas.
- [ ] **Información exógena W_{t+1}:** la trayectoria OHLCV hasta el cierre de la posición.
- [ ] **Función de transición S^M(S_t, x_t, W_{t+1}):** incluye la lógica de cierre (horizonte, stop, take-profit, cierre
      explícito), la resolución intrabarra y la duración τ (semi-Markov).
- [ ] **Contribución C(S_t, x_t, W_{t+1})** con costos, por paso o terminal.
- [ ] **Objetivo:** max sobre políticas de E[Σ γ^{t_n} C(...)] con el descuento por duración real.
- [ ] **Política:** clase de política de Powell (aproximación de la función de valor con DQN) y cómo se busca.
- [ ] Para cada decisión de diseño (contenido del estado, espacio de decisión, episodio, parámetros del periodo de tenencia,
      contribución): elección, al menos una alternativa considerada y por qué se rechazó **con referencia a datos u
      observaciones empíricas** (tabla desde `decisiones.md`).
- [ ] Definición de episodio explícita y qué se pierde con ella (el mercado es continuo).
- [ ] Ejemplo numérico de una transición completa (puede ser la operación de referencia de la prueba 3).

## Criterio de terminado

`docs/powell.md` completo con el nombre de la función junto a cada elemento y la sección copiada al reporte.

## Preguntas probables

- ¿Cuál es su estado físico, de información y de creencia? ¿Dónde está la incertidumbre de la creencia?
- ¿Qué clase de política de Powell es su agente?
