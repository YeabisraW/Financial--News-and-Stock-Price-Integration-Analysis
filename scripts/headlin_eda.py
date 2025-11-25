import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1. Load CSV
# ---------------------------
df = pd.read_csv("raw_analyst_ratings.csv")

# ---------------------------
# 2. Basic text length statistics
# ---------------------------
df["char_length"] = df["headline"].astype(str).str.len()
df["word_length"] = df["headline"].astype(str).str.split().str.len()

# ---------------------------
# 3. Publisher frequency
# ---------------------------
publisher_counts = df["publisher"].value_counts()

# ---------------------------
# 4. Date trends
# ---------------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")
daily_counts = df.groupby(df["date"].dt.date).size()

# ---------------------------
# 5. Make results folder
# ---------------------------
os.makedirs("results", exist_ok=True)

# Save statistics
df.to_csv("results/headlines_with_lengths.csv", index=False)
publisher_counts.to_csv("results/publisher_counts.csv")
daily_counts.to_csv("results/daily_article_counts.csv")

# ---------------------------
# 6. Plots
# ---------------------------

# Headline length histogram
plt.figure()
plt.hist(df["char_length"], bins=30)
plt.title("Headline Character Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/headline_char_hist.png")
plt.close()

# Publisher bar chart
plt.figure(figsize=(8, 6))
publisher_counts.plot(kind="bar")
plt.title("Number of Articles per Publisher")
plt.xlabel("Publisher")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/publisher_counts.png")
plt.close()

# Daily frequency line chart
plt.figure(figsize=(10, 5))
daily_counts.plot()
plt.title("Articles Published per Day")
plt.xlabel("Date")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/daily_counts.png")
plt.close()

print("EDA Complete! Check the 'results/' folder.")
print(df["publisher"].head())
print(df["date"].head())
print(publisher_counts)
print(daily_counts)
