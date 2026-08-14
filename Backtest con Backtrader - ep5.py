# Importación de Librerías
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

# Error en caso de que MT5 no esté abierto
if not mt5.initialize():
    print(f"Error MT5: {mt5.last_error()}"); quit()
    
# Parámetro de descarga
SIMBOLO = "AAPL"
TEMPORALIDAD = mt5.TIMEFRAME_H1
FECHA_INICIO = datetime(2020, 1, 1)
FECHA_FIN = datetime(2026, 1, 1)

rates = mt5.copy_rates_range(SIMBOLO, TEMPORALIDAD, FECHA_INICIO, FECHA_FIN)
mt5.shutdown() #Cierre post descarga


# Convertir a DataFrame
df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")
df = df.set_index("time")
df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","tick_volume":"Volume"})

df["Retorno"] = df["Close"].pct_change(fill_method=None)
df = df.dropna()

print(f"Velas: {len(df)} | {df.index[0].date()} >> {df.index[-1].date()}")

# -----------------------------------------------------------------------------
# GENERAR SEÑALES CON BACKTRADER ----------------------------------------------
# -----------------------------------------------------------------------------

import backtrader as bt

class GoldenCross(bt.Strategy):
    params = (("ventana_rapida", 50),("ventana_lenta", 200),)
    
    def __init__(self):
        self.sma_r = bt.indicators.SMA(self.data.close, period=self.p.ventana_rapida)
        self.sma_l = bt.indicators.SMA(self.data.close, period=self.p.ventana_lenta)
        # Cruce alcista = +1 | Cruce bajista = -1
        self.cruce = bt.indicators.CrossOver(self.sma_r, self.sma_l)
        
    def next(self):
        if self.cruce > 0 and not self.position :
            self.buy() # GC Entrada
        elif self.cruce < 0 and self.position:
            self.sell() # DC Salida
            
    def notify_trade(self, trade):
        if trade.isclosed:
            print(f" PnL neto: {trade.pnlcomm:.2f}")


# -----------------------------------------------------------------------------
# BT ANALYZER -----------------------------------------------------------------
# -----------------------------------------------------------------------------

# Covenrtir DF a BT
data_bt = bt.feeds.PandasData(dataname=df, datetime=None, 
                              open="Open", high="High", low="Low", close="Close",
                              volume="Volume", openinterest=-1)

# Config cerebro
cerebro = bt.Cerebro()
cerebro.adddata(data_bt)
cerebro.addstrategy(GoldenCross, ventana_rapida=50, ventana_lenta=200)

# Capital y comisiones
capital_inicial = 10_000
cerebro.broker.set_cash(capital_inicial)
cerebro.broker.setcommission(commission=0.001)
cerebro.addsizer(bt.sizers.AllInSizer, percents=95)

# Analizadores automáticos
cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name="Sharpe")
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="DrawDown")
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="Trades")


resultados = cerebro.run()
print(f"Capital Final: ${cerebro.broker.getvalue():.2f}")

# -----------------------------------------------------------------------------
# MÉTRICAS --------------------------------------------------------------------
# -----------------------------------------------------------------------------

start = resultados[0]
capital_final = cerebro.broker.getvalue()

# Extraer de los analizadores
sharpe_val = start.analyzers.Sharpe.get_analysis().get("sharperatio", None)
dd = start.analyzers.DrawDown.get_analysis()
trades = start.analyzers.Trades.get_analysis()

# Calcular métricas derivadas
retorno = (capital_final - capital_inicial)/ capital_inicial
total_tr = trades.get("total",{}).get("closed",0)
won = trades.get("won",{}).get("total",0)
winrate = won / total_tr if total_tr > 0 else 0
avg_won = trades.get("won",{}).get("pnl",{}).get("average",0)
avg_lost = trades.get("lost",{}).get("pnl",{}).get("average",0)
rrr = abs(avg_won/avg_lost) if avg_lost != 0 else 0


print(f"Retorno total: {retorno:.2%}")
print(f"Sharpe ratio: {sharpe_val:.3f}")
print(f"Drawdown Máximo: {dd.max.drawdown/100:.2%}")
print(f"Operaciones: {total_tr}")
print(f"Win Rate: {winrate:.1%}")
print(f"RRB: {rrr:.2f}")

# -----------------------------------------------------------------------------
# IN-SAMPLE vs OUT-OF-SAMPLE --------------------------------------------------
# -----------------------------------------------------------------------------

def run_backtest(df_subset, ventana_r=50, ventana_l=200,
                 capital=capital_inicial, comision=0.001):
    feed = bt.feeds.PandasData(dataname=df_subset, datetime=None,
                               open="Open", high="High", low="Low",
                               close="Close", volume="Volume", openinterest=-1)
    cer = bt.Cerebro()
    cer.adddata(feed)
    cer.addstrategy(GoldenCross, ventana_rapida=ventana_r, ventana_lenta=ventana_l)
    cer.broker.set_cash(capital)
    cer.broker.setcommission(commission=comision)
    cer.addsizer(bt.sizers.AllInSizer, percents=95)
    cer.addanalyzer(bt.analyzers.SharpeRatio_A, _name="Sharpe")
    cer.addanalyzer(bt.analyzers.DrawDown,      _name="DrawDown")
    cer.addanalyzer(bt.analyzers.TradeAnalyzer, _name="Trades")

    res = cer.run()
    st  = res[0]
    ret = (cer.broker.getvalue() - capital) / capital
    sh  = st.analyzers.Sharpe.get_analysis().get("sharperatio", 0) or 0
    dd  = st.analyzers.DrawDown.get_analysis().max.drawdown / 100
    tr  = st.analyzers.Trades.get_analysis()
    total = tr.get("total", {}).get("closed", 0)
    won   = tr.get("won",   {}).get("total",  0)
    wr    = won / total if total > 0 else 0
    return {"retorno": ret, "sharpe": sh, "max_dd": dd,
            "trades": total, "winrate": wr}


# Separar In-Sample / Out-of-Sample
tasa_de_split = 0.70
split  = int(len(df) * tasa_de_split)
df_is  = df.iloc[:split].copy()   # para diseñar
df_oos = df.iloc[split:].copy()   # para validar

print(f"\nIS:  {df_is.index[0].date()} >> {df_is.index[-1].date()}  ({len(df_is)} velas)")
print(f"OOS: {df_oos.index[0].date()} >> {df_oos.index[-1].date()}  ({len(df_oos)} velas)\n")

# Correr backtest en cada período
for nombre, datos in [("IN-SAMPLE   ", df_is), ("OUT-OF-SAMPLE", df_oos)]:
    m = run_backtest(datos)
    print(f"{nombre} | Retorno {m['retorno']:>8.2%} | Sharpe {m['sharpe']:>6.2f} | "
          f"DD {m['max_dd']:>7.2%} | Trades {m['trades']:>3} | WR {m['winrate']:>5.1%}")

print("\n▸ Si Sharpe IS >> Sharpe OOS → overfitting.")
print("▸ Si ambos son similares → la señal puede ser real.")
















































































