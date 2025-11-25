# ---------------------------
# Time Series Analysis of Article Publications
# ---------------------------

import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1. Load CSV
# ---------------------------
print("Loading CSV...")
df = pd.read_csv("raw_analyst_ratings.csv")
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # ensure datetime format
print(f"CSV loaded: {len(df)} rows")

# ---------------------------
# 2. Daily and Monthly Counts
# ---------------------------
daily_counts = df.groupby(df['date'].dt.date).size()
monthly_counts = df.groupby(df['date'].dt.to_period('M')).size()

# ---------------------------
# 3. Detect spikes in daily publications
# ---------------------------
threshold = daily_counts.mean() + 2*daily_counts.std()
spike_days = daily_counts[daily_counts > threshold]

# Save spike days to CSV
os.makedirs("results", exist_ok=True)
spike_days.to_csv("results/spike_days.csv", header=['count'])
print("Spike days CSV saved: results/spike_days.csv")

# ---------------------------
# 4. Analyze publishing times (hour of day)
# ---------------------------
df['hour'] = df['date'].dt.hour
hourly_counts = df.groupby('hour').size()

# ---------------------------
# 5. Plotting
# ---------------------------
# Daily publication trends
plt.figure(figsize=(12,5))
daily_counts.plot()
plt.title("Articles Published Per Day")
plt.xlabel("Date")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.savefig("results/daily_article_counts.png")
plt.close()

# Monthly publication trends
plt.figure(figsize=(12,5))
monthly_counts.plot()
plt.title("Articles Published Per Month")
plt.xlabel("Month")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.savefig("results/monthly_article_counts.png")
plt.close()

# Hourly publication counts
plt.figure(figsize=(10,5))
hourly_counts.plot(kind='bar')
plt.title("Articles Published by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Articles")
plt.tight_layout()
plt.savefig("results/hourly_article_counts.png")
plt.close()

print("Time series analysis complete! Check the 'results/' folder for CSVs and plots.")
print(f"Detected spike days:\n{spike_days}")
