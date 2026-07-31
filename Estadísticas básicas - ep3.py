# =============================================================================
# QUANTER-STRIKE — Episodio 3
# Estadística que sí necesitás 
# =============================================================================

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


# =============================================================================
# 1. DESCARGA DE DATOS
# =============================================================================

df = yf.download("AAPL", start="2020-01-01", end="2024-01-01",
                 auto_adjust=True, progress=False)

df["Retorno"] = df["Close"].pct_change(fill_method=None)
df = df.dropna()

print(df[["Close", "Retorno"]].head(10))
print(f"\nTotal de días: {len(df)}")


# =============================================================================
# 2. MEDIA Y DESVIACIÓN ESTÁNDAR
# =============================================================================

media = float(df["Retorno"].mean())
sigma = float(df["Retorno"].std())

print(f"Media diaria:          {media:.4%}")
print(f"Media anualizada:      {media * 252:.2%}")
print(f"Volatilidad diaria:    {sigma:.4%}")
print(f"Volatilidad anual:     {sigma * np.sqrt(252):.2%}")
print(f"Sharpe aproximado:     {(media / sigma) * np.sqrt(252):.2f}")


# =============================================================================
# 3. ¿LOS RETORNOS SON NORMALES? — Skewness y Kurtosis
# =============================================================================

skewness = df["Retorno"].skew()
kurtosis = df["Retorno"].kurt()
jb_stat, jb_p = stats.jarque_bera(df["Retorno"].values)

print(f"Skewness:            {skewness:.4f}  (normal = 0)")
print(f"Kurtosis:            {kurtosis:.4f}  (normal = 0)")
print(f"Jarque-Bera p-valor: {jb_p:.6f}")

if jb_p < 0.05:
    print("→ RECHAZAMOS normalidad. Los retornos NO son normales.")
else:
    print("→ No podemos rechazar normalidad.")


# =============================================================================
# 4. FAT TAILS — ¿Con qué frecuencia ocurren los extremos?
# =============================================================================

print(f"{'Nivel':<10} {'Normal predice':>16} {'Realidad AAPL':>15} {'Veces más':>10}")
print("-" * 56)

for n in [1, 2, 3, 4]:
    umbral      = n * sigma
    prob_normal = 2 * (1 - stats.norm.cdf(n)) * 100
    eventos     = len(df[abs(df["Retorno"]) > umbral])
    prob_real   = eventos / len(df) * 100
    ratio       = prob_real / prob_normal if prob_normal > 0 else 0
    print(f"  ±{n}σ ({umbral:.2%})   {prob_normal:>12.2f}%   {prob_real:>12.2f}%   {ratio:>8.1f}x")

print(f"\n── 5 peores días ──────────────────────────────────")
for fecha, ret_val in df["Retorno"].nsmallest(5).items():
    print(f"  {str(fecha.date())}   {ret_val:+.2%}   ({ret_val/sigma:.1f}σ)")


# =============================================================================
# 5. CORRELACIÓN — ¿Cómo se relacionan los activos entre sí?
# =============================================================================

tickers  = ["AAPL", "MSFT", "GLD", "BTC-USD"]
precios  = yf.download(tickers, start="2020-01-01", end="2024-01-01",
                       auto_adjust=True, progress=False)["Close"]
retornos = precios.pct_change(fill_method=None).dropna()
corr     = retornos.corr().round(3)

print(corr)
print("\n→ Cercano a  1: se mueven juntos")
print("→ Cercano a  0: independientes")
print("→ Cercano a -1: se mueven en sentido opuesto")


# =============================================================================
# 6. FUNCIÓN REUTILIZABLE — Análisis completo en una línea
# =============================================================================

def analisis(ticker, start="2020-01-01", end="2024-01-01"):
    data    = yf.download(ticker, start=start, end=end,
                          auto_adjust=True, progress=False)
    ret     = data["Close"].pct_change(fill_method=None).dropna()
    ret     = ret.squeeze()
    media_f = float(ret.mean())
    sigma_f = float(ret.std())
    cum     = (1 + ret).cumprod()
    max_dd  = float(((cum - cum.cummax()) / cum.cummax()).min())
    _, jb_p = stats.jarque_bera(ret.values)

    print(f"\n── {ticker} ──────────────────────────────────────")
    print(f"  Media anualizada:      {media_f * 252:.2%}")
    print(f"  Volatilidad anual:     {sigma_f * np.sqrt(252):.2%}")
    print(f"  Sharpe:                {(media_f / sigma_f) * np.sqrt(252):.2f}")
    print(f"  Skewness:              {ret.skew():.3f}")
    print(f"  Kurtosis:              {ret.kurt():.3f}")
    print(f"  VaR 95% diario:        {np.percentile(ret, 5):.2%}")
    print(f"  Max Drawdown:          {max_dd:.2%}")
    print(f"  ¿Distribución normal?: {'NO' if jb_p < 0.05 else 'Posiblemente'}")


# Probalo con cualquier ticker
analisis("AAPL")
analisis("GLD")
analisis("BTC-USD")