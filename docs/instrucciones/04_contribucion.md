# 04 — Función de contribución

**Dueño:** Gonzalo Cano (GC) · **Revisor:** Juan Manuel Espinosa (JME) · **Notebook:** `notebooks/04_contribucion.ipynb` · **Módulo:** `code/reward.py` · **Depende de:** 03 · **Alimenta a:** 05, 08
**Leer antes:** `docs/notas_clase/02_rl_q_learning_gymnasium.md` (§6, escala de la recompensa) · `docs/notas_clase/01_sequential_decision_analytics.md` · `docs/Lab04.pdf` §3.4 y pregunta 3

## Objetivo

Definir la contribución C(S, x, W) que recibe el agente al cerrar cada decisión, con los costos de 0.125 % por lado
dentro, justificar si es **terminal** (pagada al cierre) o **por paso** (acumulada durante la posición) en términos de
asignación de crédito, y discutir sus modos de falla.

## Contrato

- **Lee:** datos de la operación cerrada que entrega el entorno (precios de entrada y salida, lado, tamaño, duración,
  trayectoria si hace falta).
- **Expone (propuesta):** funciones puras `net_log_return(...)`, `vol_scaled_return(...)`, etc., y un registro
  `REWARDS: dict[str, Callable]` para elegir por nombre desde `train.py` y comparar variantes.

## Loop de trabajo

```
REPETIR hasta cumplir el criterio de terminado:
  1. Toma el primer ítem sin [x].
  2. Implementa la función pura con docstring (fórmula en notación de clase).
  3. Verifica con números a mano y con el agente aleatorio (recompensa media < 0 con costos).
  4. Marca [x], registra la decisión, commit.
```

## Checklist

- [x] Contribución elegida, con fórmula explícita y unidades (p. ej. rendimiento log neto de costos de la operación,
      escalado por la volatilidad estimada en la entrada para quedar en orden 10⁻²).
- [x] Costos de 0.125 % por lado dentro de la contribución (entrada y salida).
- [x] **Terminal vs por paso** (pregunta 3): elección, justificación en asignación de crédito y qué comportamiento
      indeseado habría inducido la otra opción (con un experimento corto si es posible).
- [x] Recompensa de "esperar" definida (cero, costo de oportunidad o penalización) y justificada.
- [x] Modos de falla discutidos, uno por trampa de §3.4: P&L bruto premia riesgo apalancado; Sharpe por episodio no es
      markoviano; rendimientos log por paso ignoran el drawdown; coeficientes de moldeo son hiperparámetros (si se usan,
      se ajustan en validación y se justifican).
- [x] Al menos una alternativa comparada y descartada con datos (p. ej. P&L bruto o rendimiento sin escalar).
- [x] `test_reward.py`: la contribución de una operación conocida coincide con el cálculo a mano.

## Criterio de terminado

La contribución está fijada, documentada en `decisiones.md` con alternativa y evidencia, y el agente aleatorio
tiene contribución esperada negativa con costos en entrenamiento.

## Trampas

- Recompensa en dólares (orden 10³): desestabiliza la red (S18).
- Pagar la recompensa por paso con un descuento por barra mientras el agente decide por operación: inconsistente con γ^τ.
- Olvidar el costo de salida cuando la posición cierra por stop o take-profit.

## Preguntas probables

- ¿Su contribución es por paso o terminal? ¿Qué habría hecho el agente con la otra?
- ¿Por qué no Sharpe por episodio? ¿Su recompensa castiga el drawdown?
