# ---------------------------
# Publisher Analysis
# ---------------------------

import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# ---------------------------
# 1. Load CSV
# ---------------------------
df = pd.read_csv("raw_analyst_ratings.csv")
print(f"CSV loaded: {len(df)} rows")

# ---------------------------
# 2. Count articles per publisher
# ---------------------------
publisher_counts = df['publisher'].value_counts()
os.makedirs("results", exist_ok=True)
publisher_counts.to_csv("results/publisher_counts.csv")
print("Publisher counts saved: results/publisher_counts.csv")

# ---------------------------
# 3. Identify publishers with email addresses
# ---------------------------
df['is_email'] = df['publisher'].str.contains(r'@', na=False)
df['domain'] = df['publisher'].str.extract(r'@([^\s]+)')

domain_counts = df[df['is_email']]['domain'].value_counts()
domain_counts.to_csv("results/publisher_domains.csv")
print("Publisher domains saved: results/publisher_domains.csv")

# ---------------------------
# 4. Plot top publishers
# ---------------------------
top_publishers = publisher_counts.head(20)
plt.figure(figsize=(12,6))
top_publishers.plot(kind='bar', color='skyblue')
plt.title("Top 20 Publishers by Article Count")
plt.ylabel("Number of Articles")
plt.xlabel("Publisher")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("results/top_publishers_bar.png")
plt.close()
print("Top publishers plot saved: results/top_publishers_bar.png")

print("Publisher analysis complete! Check the 'results/' folder for CSVs and plot.")
