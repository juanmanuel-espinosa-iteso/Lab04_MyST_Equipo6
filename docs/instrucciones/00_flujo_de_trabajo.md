# 00 — Flujo de trabajo y división del proyecto

## Cómo usar estas guías

Cada archivo `NN_*.md` es una parte del proyecto y funciona como un **prompt con loop**: se le puede pasar a
Claude (o seguir a mano) tal cual. Todas tienen la misma forma:

1. **Encabezado**: dueño, módulo, dependencias, notas de clase a leer antes.
2. **Contrato**: qué lee y qué escribe (archivos de `artifacts/`). No se rompe sin avisar.
3. **Loop de trabajo**: implementar → verificar → marcar → commit.
4. **Checklist**: cada requisito del enunciado que toca esa parte. Nada se entrega sin `[x]`.
5. **Criterio de terminado**: cuándo se sale del loop.
6. **Trampas** y **preguntas probables de la exposición**.

Prompt sugerido para Claude:

> Lee `docs/instrucciones/CLAUDE.md`, `docs/instrucciones/NN_xxx.md` y las notas de clase que indica. Ejecuta el
> loop de trabajo hasta cumplir el criterio de terminado. Marca el checklist conforme avances y registra en
> `docs/decisiones.md` cada decisión de diseño con su alternativa descartada y la evidencia. Explícame cada función
> al terminarla.

## Orden y dependencias

```
01 datos y features ──► 02 EDA y análisis de features
        │
        └──► 03 entorno ──► 04 contribución ──► 05 entrenamiento ──► 06 evaluación ──► 07 sensibilidad y producción
                  └──────────────── 08 formulación de Powell (reporte) ────────────────┘
09 pruebas (crecen con cada parte) · 10 reporte, walkthrough y presentación · 11 gestión del proyecto
```

Para trabajar en paralelo sin esperar al otro: se acuerda primero el **contrato de artefactos**
(`docs/instrucciones/CLAUDE.md`) y quien va adelante deja un artefacto "dummy" con el esquema correcto.
La interfaz crítica es la de 01 → 03: `build_fold_dataset(fold)` devuelve OHLC y la matriz de features
normalizada por partición; con eso 03 y 04 avanzan aunque las features cambien después.

## División del trabajo (50/50)

Propuesta del 2026-11-23. JME = Juan Manuel Espinosa, GC = Gonzalo Cano. La asignación se copia en los roles de `charter.md`.

| Parte | Dueño | Revisor (PR) | Notebook de demostración | Pruebas a su cargo |
|---|---|---|---|---|
| 01 Datos, features causales, folds y normalizador | JME | GC | `01_datos_features.ipynb` | `test_causality.py` |
| 02 EDA, regímenes y análisis de features | GC | JME | `02_eda.ipynb` | — |
| 03 Entorno semi-Markov y gestor de posiciones | JME | GC | `03_entorno.ipynb` | `test_env_check.py`, `test_reference_trade.py` |
| 04 Función de contribución | GC | JME | `04_contribucion.ipynb` | `test_reward.py` |
| 05 Entrenamiento SB3, líneas base, semillas e hiperparámetros | GC | JME | `05_entrenamiento.ipynb` | — |
| 06 Evaluación: métricas, benchmarks, bitácora, análisis de operaciones | JME | GC | `06_evaluacion.ipynb` | `test_accounting.py` |
| 07 Sensibilidad y fallas de producción | JME: intrabarra y retraso · GC: ablación y segundo algoritmo | cruzado | `07_sensibilidad.ipynb` | — |
| 08 Formulación de Powell (reporte) | JME: estado, transición y gestor de posiciones · GC: contribución, objetivo y política | cruzado | — | — |
| 09 Pruebas | cada quien las de su parte (columna de la derecha) | cruzado | — | — |
| 10 Reporte, walkthrough y presentación | ambos, mitad y mitad (cada quien lo de sus partes) | — | — | — |
| 11 Gestión del proyecto | JME: charter y sprint 01 · ambos: cierre de sprint y post-mortem | — | — | — |

**Notebooks (`notebooks/`):** uno por parte, para estudiar para la exposición y mostrar avance. Solo importan
funciones de `code/` y muestran resultados; **no contienen lógica propia**.

## Reglas de Git

- Ramas `feat/gc-<tema>` (Gonzalo) y `feat/jme-<tema>` (Juan Manuel). `main` siempre corre.
- PR obligatorio revisado por el otro integrante. El revisor debe poder explicar el código que aprueba.
- Commits pequeños, en español, con prefijo: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`.
- Revisar el balance cada día: `git shortlog -sn --no-merges`.
- Cada quien commitea con su propia cuenta (`git config user.name` / `user.email` correctos).
