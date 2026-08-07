# Importar librerías 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Descarga de datos
df = yf.download("AAPL", start="2018-01-01", end="2026-01-01", auto_adjust=True, progress=False)
df["Retorno"] = df["Close"].pct_change(fill_method=None)
df = df.dropna()

# Calcular las medias móviles
df["SMA50"] = df["Close"].rolling(window=50).mean()
df["SMA200"] = df["Close"].rolling(window=200).mean()

# Ver rdo parcial
print(df[["Close", "SMA50", "SMA200"]].tail(10))


#-----------------------------------------------------------------------------
# GENERAR SEÑAL --------------------------------------------------------------
#-----------------------------------------------------------------------------

# Señal = 1 (en Mercado) | si es = 0 (fuera del mercado)

df["Señal"] = 0
df.loc[df["SMA50"]>df["SMA200"], "Señal"] = 1

# Detectar cambios de señal 
df["Posicion"] = df["Señal"].diff()

# Posición +1 = Golden Cross (entrada)
# Posición -1 = Death Cross (salida)

entradas = df[df["Posicion"] == 1]
salidas = df[df["Posicion"] == -1]

# Ver rdos parciales
print(f"Señal de entradas: {len(entradas)}")
print(f"Señal de salidas: {len(salidas)}")
print(entradas[["Close","SMA50","SMA200"]].tail(3))


#-----------------------------------------------------------------------------
# PLOT DE GRÁFICO ------------------------------------------------------------
#-----------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

# Precio y medias móviles
ax.plot(df["Close"], color='blue', linewidth=0.8, label="AAPL", alpha=0.9)
ax.plot(df["SMA50"], color='green', linewidth=1.5, label="SMA 50")
ax.plot(df["SMA200"], color='red', linewidth=1.5, label="SMA 200")

# Señales de entrada
ax.scatter(entradas.index, entradas["Close"], marker="^", color='#1D9E75', s=120, zorder=5,
           label="Entrada - GC")

# Señales de salida
ax.scatter(salidas.index, salidas["Close"], marker="v", color='#C0392B', s=120, zorder=5,
           label="Salida - DC")


# plots
ax.set_title("AAPL - Estrategia Golden Cross")
ax.legend();ax.grid(True, alpha=0.3)
plt.tight_layout();plt.show()


#-----------------------------------------------------------------------------
# ESTRATEGIA VS B&H ----------------------------------------------------------
#-----------------------------------------------------------------------------

# Retorno diario de la estrategia
df["Ret_est"] = df["Retorno"] * df["Señal"].shift(1)

# Retorno acumulado
df["Acum_est"] = ( 1 + df["Ret_est"]).cumprod()
df["Acum_bnh"] = ( 1 + df["Retorno"]).cumprod()

# Resultados finales
ret_est = df["Acum_est"].iloc[-1] - 1
ret_bnh = df["Acum_bnh"].iloc[-1] - 1

print(f"Retorno estrategia Golden Cross: {ret_est:.2%}")
print(f"Retorno Buy & Hold AAPL: {ret_bnh:.2%}")
print(f"Diferencia: {ret_est - ret_bnh:.2%}")












































