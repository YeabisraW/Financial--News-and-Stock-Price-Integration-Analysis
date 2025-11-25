# scripts/pynance_integration.py
from pynance import MarketData
import pandas as pd
import matplotlib.pyplot as plt

def fetch_stock_data(ticker, start_date, end_date):
    md = MarketData([ticker])
    df = md.load_data(start=start_date, end=end_date)
    df['return'] = df['close'].pct_change()
    df['volatility_30d'] = df['return'].rolling(window=30).std() * (252 ** 0.5)
    return df

if __name__ == "__main__":
    ticker = "AAPL"
    df = fetch_stock_data(ticker, "2020-01-01", "2023-01-01")
    print(df.head())

    # Plot volatility
    df['volatility_30d'].plot(title=f"30-day annualized volatility ({ticker})")
    plt.show()
