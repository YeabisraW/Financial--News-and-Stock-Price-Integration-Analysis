# ---------------------------------------------------------
# FULL QUANTITATIVE & TECHNICAL ANALYSIS SCRIPT
# - Load stock data
# - Compute technical indicators (TA-Lib)
# - Financial metrics (PyNance)
# - Visualization of Close, SMA, RSI, MACD
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import talib
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load & Prepare Data
# ---------------------------------------------------------
file_path = "META.csv"  
print(f"Loading CSV: {file_path}")

df = pd.read_csv(file_path, parse_dates=["Date"])
df.set_index("Date", inplace=True)

# Ensure numeric consistency
cols = ["Open", "High", "Low", "Close", "Volume"]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)

print(f"CSV loaded: {df.shape[0]} rows")
print("Data cleaned and prepared.\n")

# ---------------------------------------------------------
# 2. Technical Indicators (TA-Lib)
# ---------------------------------------------------------
print("Calculating TA-Lib indicators...")

df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)

df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
    df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
)

df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
    df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2
)

print("TA-Lib indicators completed.\n")

# ---------------------------------------------------------
# 3. Financial Metrics (Manual — No PyNance)
# ---------------------------------------------------------
print("Calculating financial metrics...")

# Daily returns
df['DailyReturn'] = df['Close'].pct_change()

# Annualized volatility
df['Volatility'] = df['DailyReturn'].rolling(20).std() * np.sqrt(252)

# Rolling Sharpe ratio (20-day)
risk_free_rate = 0.02 / 252  # approx daily risk-free rate
df['Sharpe_20'] = (df['DailyReturn'].rolling(20).mean() - risk_free_rate) / \
                   df['DailyReturn'].rolling(20).std()

print("Financial metrics computed.\n")

# ---------------------------------------------------------
# 4. Save Results
# ---------------------------------------------------------
os.makedirs("results", exist_ok=True)

output_csv = "results/technical_analysis_META.csv"
df.to_csv(output_csv)

print(f"Technical analysis saved to: {output_csv}\n")

# ---------------------------------------------------------
# 5. Visualizations
# ---------------------------------------------------------

# ----- Plot 1: Close Price + SMAs -----
plt.figure(figsize=(14,7))
plt.plot(df.index, df['Close'], label='Close')
plt.plot(df.index, df['SMA_20'], label='SMA 20')
plt.plot(df.index, df['SMA_50'], label='SMA 50')
plt.title("Close Price with SMA 20 & 50")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("results/plot_close_sma_META.png")
plt.close()

# ----- Plot 2: RSI -----
plt.figure(figsize=(14,5))
plt.plot(df.index, df['RSI'], label="RSI", color="purple")
plt.axhline(30, color='red', linestyle='--')
plt.axhline(70, color='green', linestyle='--')
plt.title("RSI (14)")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.tight_layout()
plt.savefig("results/plot_rsi_META.png")
plt.close()

# ----- Plot 3: MACD -----
plt.figure(figsize=(14,6))
plt.plot(df.index, df['MACD'], label="MACD")
plt.plot(df.index, df['MACD_signal'], label="Signal")
plt.bar(df.index, df['MACD_hist'], alpha=0.4, label="Histogram")
plt.title("MACD Indicator")
plt.xlabel("Date")
plt.ylabel("MACD")
plt.legend()
plt.tight_layout()
plt.savefig("results/plot_macd_META.png")
plt.close()

print("All plots saved in 'results/' folder.")
print("Technical analysis complete!")