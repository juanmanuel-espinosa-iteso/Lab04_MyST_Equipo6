# 02 — EDA, regímenes y análisis de features

**Dueño:** Gonzalo Cano (GC) · **Revisor:** Juan Manuel Espinosa (JME) · **Notebook:** `notebooks/02_eda.ipynb` · **Módulos:** `code/plots.py` (figuras, PDF), `code/evaluate.py` (estadísticos) · **Depende de:** 01 · **Alimenta a:** 03 (contenido del estado), 08, 10
**Leer antes:** `docs/Lab04.pdf` §3.2 (justificación del estado), §3.8 y preguntas 1 y 2

## Objetivo

Caracterizar los datos de entrenamiento para **justificar con evidencia** el contenido del estado de información:
regímenes, distribución de rendimientos, autocorrelación, estacionariedad y relación de cada feature con el objetivo,
todo calculado **solo sobre folds de entrenamiento**.

## Contrato

- **Lee:** `build_fold_dataset(k)` y `FEATURE_FAMILIES` (parte 01).
- **Escribe:** `artifacts/btcusdt_eda.pdf`, `artifacts/regimes.csv`, `artifacts/feature_analysis_fold_{1,2,3,4}.csv`.

## Loop de trabajo

```
REPETIR hasta cumplir el criterio de terminado:
  1. Toma el primer ítem sin [x].
  2. Implementa el cálculo (evaluate.py) y la figura (plots.py).
  3. Verifica que solo use entrenamiento y que las cifras salgan de los artefactos.
  4. Marca [x], registra en decisiones.md qué feature entra o sale y por qué, commit.
```

## Checklist

- [ ] **Regímenes (pregunta 1):** al menos 4 regímenes distintos en el entrenamiento por rango de fechas
      (p. ej. burbuja 2017, invierno 2018, recuperación 2019–2020, covid, alza 2020–2021, caída 2022), con
      volatilidad anualizada, tendencia (rendimiento) y drawdown de cada uno → `regimes.csv`. Criterio reproducible
      (p. ej. drawdown y volatilidad móvil), no a ojo.
- [ ] Distribución de rendimientos horarios y diarios: histograma contra la normal, asimetría, curtosis, Jarque–Bera.
- [ ] Autocorrelación (ACF/PACF) de rendimientos y de rendimientos al cuadrado/absolutos (agrupamiento de volatilidad),
      con Ljung–Box.
- [ ] Estacionariedad: ADF y KPSS de log-precio, rendimientos y cada feature, en el entrenamiento de cada fold.
- [ ] `feature_analysis_fold_{k}.csv`: por feature, información mutua con el objetivo, correlación de Pearson y Spearman,
      ADF p-value y autocorrelación de rezago 1; solo con el train del fold k.
- [ ] `btcusdt_eda.pdf` con todas las figuras (título, ejes, leyenda) y una página de resumen.
- [ ] **Pregunta 2:** párrafo y cifras que justifican el estado elegido y nombran la alternativa descartada
      (p. ej. ventana de precios crudos como en la clase vs features estacionarias) → `decisiones.md` y reporte.

## Criterio de terminado

`main.py` genera el PDF, `regimes.csv` y los 4 CSV de features; cada feature del estado tiene una línea de
evidencia en `decisiones.md`.

## Trampas

- Calcular cualquier estadístico con validación o prueba.
- Información mutua con muestras traslapadas: declarar que el objetivo a H horas se traslapa y que eso infla la significancia.
- Concluir "no hay autocorrelación en rendimientos" y aun así meter 24 rezagos al estado sin justificarlo.

## Preguntas probables

- ¿Qué régimen domina el entrenamiento del fold 1 y por qué importa para la prueba de 2023?
- ¿Qué feature tiene más información mutua y es estable entre folds?
