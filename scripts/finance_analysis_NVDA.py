# ---------------------------
# Quantitative / Technical Analysis
# ---------------------------

import pandas as pd
import talib
import os
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load CSV
# ---------------------------
file_path = "NVDA.csv"  # Change this to your CSV path
print(f"Loading CSV: {file_path}")
df = pd.read_csv(file_path, parse_dates=["Date"])
df.set_index("Date", inplace=True)
print(f"CSV loaded: {df.shape[0]} rows")

# ---------------------------
# 2. Prepare data
# ---------------------------
cols = ["Open", "High", "Low", "Close", "Volume"]
df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)
print("Data cleaned and numeric columns ensured.")

# ---------------------------
# 3. Compute Technical Indicators
# ---------------------------
print("Calculating technical indicators...")

# Simple Moving Averages
df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)

# RSI
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

# MACD
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
    df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
)

# Bollinger Bands
df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
    df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
)

print("Technical indicators calculated.")

# ---------------------------
# 4. Save results
# ---------------------------
os.makedirs("results", exist_ok=True)
output_csv = "results/technical_analysis_NVDA.csv"
df.to_csv(output_csv)
print(f"Technical analysis saved to {output_csv}")

# ---------------------------
# 5. Optional: Plot Closing Price with SMA
# ---------------------------
plt.figure(figsize=(14,7))
plt.plot(df.index, df['Close'], label='Close', color='blue')
plt.plot(df.index, df['SMA_20'], label='SMA 20', color='orange')
plt.plot(df.index, df['SMA_50'], label='SMA 50', color='green')
plt.title("Stock Close Price with SMA 20 & 50")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("results/close_price_sma.png")
plt.close()
print("Closing price plot with SMA saved: results/close_price_sma.png")