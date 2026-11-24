# Bitácora de decisiones de diseño

Registrar aquí toda decisión que no venga fijada por el enunciado o las notas de clase. §3.2 del enunciado exige,
para cada decisión de la formulación (contenido del estado, espacio de decisión, definición del episodio,
parámetros del periodo de tenencia y función de contribución): **la elección, al menos una alternativa considerada y
por qué se rechazó, con referencia a los datos o a observaciones empíricas**. Las justificaciones genéricas no
reciben crédito. Sirve para el reporte y para responder "¿por qué hicieron X?" en la exposición.

| Fecha | Parte | Decisión | Alternativa descartada | Justificación | Autor |
|---|---|---|---|---|---|
| 2026-11-23 | Alcance | Lab04 completo, incluida la ablación por familias de variables y el segundo algoritmo (pregunta 7) | Omitir la pregunta 7 | El enunciado no excluye ninguna sección | Equipo |
| 2026-11-23 | Datos | El CSV de Canvas (`Binance_BTCUSDT_1h.csv`) se versiona en `data/` y `main.py` lo lee de ahí | Pedir que se descargue de Canvas antes de correr | El profesor debe poder correr `python main.py` desde un clon sin pasos manuales (−10 si el comando documentado no corre); el archivo pesa 8.3 MB | JME |
| 2026-11-23 | Benchmarks | El S&P 500 se obtiene con yfinance (`^GSPC`, cierre diario) **solo para el benchmark** de §3.7, nunca como feature del estado; se guarda un snapshot en `data/sp500_daily.csv` y solo se descarga si falta | Proxy con un ETF (SPY); no reportar el benchmark | §3.1 prohíbe datos externos de precio para el modelo, pero §3.7 exige el benchmark del S&P 500 y no hay otra forma de calcularlo; el snapshot evita depender de Yahoo el día de la evaluación | Equipo |
| 2026-11-23 | Organización | Un notebook de demostración por parte en `notebooks/`, que solo importa funciones de `code/` | Notebooks con lógica propia | La rúbrica penaliza código en notebooks y duplicado; el pipeline evaluado es `main.py` (igual que en el Lab03) | Equipo |
| 2026-11-23 | Organización | Los notebooks importan el paquete con `sys.path.insert(0, '..')` y `sys.modules.pop('code', None)` antes de `import code` | Renombrar la carpeta `code/` | `code/` es obligatoria (§2.2), pero `code` también es un módulo de la biblioteca estándar y IPython lo importa al arrancar: en un kernel de Jupyter `import code` devolvía el estándar (verificado el 2026-11-23). Desde la raíz (`python main.py`, `pytest`) gana el paquete local sin ajustes | JME |
| 2026-11-23 | Entregables | Se agrega una **simulación visual** del agente operando (`artifacts/agent_simulation.gif`): animación barra por barra con precio, decisiones, entradas y salidas con su razón, stop/take-profit, posición y capital, construida solo con la bitácora y la curva de capital | Solo figuras estáticas | Petición del profesor en clase (2026-11-23), por analogía con la animación del pingüino en FrozenLake de la S18–S19; ayuda a explicar en la exposición cómo decide el agente | Profesor / Equipo |
| 2026-11-23 | Organización | Sin dashboard | Dashboard en Streamlit como en el Lab03 | El enunciado del Lab04 no lo pide (§2.1: cuatro entregables) | Equipo |
