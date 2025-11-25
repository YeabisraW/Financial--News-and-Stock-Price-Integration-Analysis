# ---------------------------
# Task 3: Correlation between News and Stock Movement
# ---------------------------

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1. Load and align datasets
# ---------------------------
print("Loading datasets...")
news = pd.read_csv("raw_analyst_ratings.csv")
stock = pd.read_csv("GOOG.csv", parse_dates=['Date'])

# Convert 'date' column to datetime
news['date'] = pd.to_datetime(news['date'], errors='coerce')
news = news.dropna(subset=['date'])  # drop rows with invalid dates
news['date'] = news['date'].dt.date
stock['Date'] = stock['Date'].dt.date

# Aggregate multiple news per day
daily_news = news.groupby('date').agg({'headline': ' '.join}).reset_index()
print(f"News aggregated: {daily_news.shape[0]} days")

# ---------------------------
# 2. Sentiment analysis
# ---------------------------
print("Performing sentiment analysis...")
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

daily_news['sentiment'] = daily_news['headline'].apply(get_sentiment)

# ---------------------------
# 3. Calculate daily stock returns
# ---------------------------
stock['daily_return'] = stock['Close'].pct_change()
stock = stock.dropna(subset=['daily_return'])

# ---------------------------
# 4. Merge datasets by date
# ---------------------------
merged = pd.merge(stock, daily_news, left_on='Date', right_on='date', how='inner')
print(f"Merged dataset: {merged.shape[0]} rows")

# ---------------------------
# 5. Correlation analysis
# ---------------------------
corr = merged['sentiment'].corr(merged['daily_return'])
print(f"Correlation between daily sentiment and stock return: {corr:.4f}")

# ---------------------------
# 6. Save results
# ---------------------------
os.makedirs("results", exist_ok=True)
merged_csv = "results/sentiment_stock_correlation_GOOG.csv"
merged.to_csv(merged_csv, index=False)
print(f"Merged data with sentiment and returns saved: {merged_csv}")

# ---------------------------
# 7. Visualization
# ---------------------------
# Scatter plot: Sentiment vs Daily Return
plt.figure(figsize=(10,6))
plt.scatter(merged['sentiment'], merged['daily_return'], alpha=0.5)
plt.xlabel("Daily News Sentiment")
plt.ylabel("Daily Stock Return")
plt.title(f"Correlation: Sentiment vs Stock Return ({corr:.4f})")
plt.tight_layout()
plt.savefig("results/sentiment_vs_stock_return_GOOG.png")
plt.close()
print("Scatter plot saved: results/sentiment_vs_stock_return_GOOG.png")

# Time series plot: Sentiment and Stock Returns
plt.figure(figsize=(12,6))
plt.plot(merged['Date'], merged['sentiment'], label='Daily Sentiment', color='blue')
plt.plot(merged['Date'], merged['daily_return'], label='Daily Stock Return', color='orange')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Daily News Sentiment and Stock Returns")
plt.legend()
plt.tight_layout()
plt.savefig("results/sentiment_and_returns_timeseries_GOOG.png")
plt.close()
print("Time series plot saved: results/sentiment_and_returns_timeseries_GOOG.png")

print("Task 3 complete! Check the 'results/' folder for CSV and plots.")
