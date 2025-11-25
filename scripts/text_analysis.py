# ---------------------------
# Text Analysis / Topic Modeling
# ---------------------------

import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# ---------------------------
# 1. Load CSV
# ---------------------------
print("Loading CSV...")
df = pd.read_csv("raw_analyst_ratings.csv")
print(f"CSV loaded: {len(df)} rows")

# ---------------------------
# 2. Clean text
# ---------------------------
print("Cleaning text...")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation/numbers
    text = ' '.join([w for w in text.split() if w not in stop_words])  # remove stopwords
    return text

df['clean_headline'] = df['headline'].apply(clean_text)
print("Text cleaning complete!")

# ---------------------------
# 3. Keyword extraction (single words)
# ---------------------------
print("Extracting top keywords...")
all_words = ' '.join(df['clean_headline']).split()
word_counts = Counter(all_words)
top_words = word_counts.most_common(20)

# Save keywords to CSV
os.makedirs("results", exist_ok=True)
pd.DataFrame(top_words, columns=['word', 'count']).to_csv("results/top_keywords.csv", index=False)
print("Top keywords CSV saved: results/top_keywords.csv")

# ---------------------------
# 4. Phrase extraction (bigrams & trigrams)
# ---------------------------
print("Extracting top phrases (bigrams & trigrams)...")
vectorizer = CountVectorizer(ngram_range=(2,3), stop_words='english', min_df=2)
X = vectorizer.fit_transform(df['clean_headline'])
sum_words = X.sum(axis=0)
phrases_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
phrases_freq = sorted(phrases_freq, key=lambda x: x[1], reverse=True)
top_phrases = phrases_freq[:20]

# Save phrases to CSV
pd.DataFrame(top_phrases, columns=['phrase', 'count']).to_csv("results/top_phrases.csv", index=False)
print("Top phrases CSV saved: results/top_phrases.csv")

# ---------------------------
# 5. Topic modeling (LDA)
# ---------------------------
print("Starting topic modeling (LDA)...")
lda_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
X_lda = lda_vectorizer.fit_transform(df['clean_headline'])

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X_lda)
print("LDA topic modeling complete!")

topics = {}
for i, topic in enumerate(lda.components_):
    top_words_topic = [lda_vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-10:]]
    topics[f"Topic {i+1}"] = top_words_topic

pd.DataFrame.from_dict(topics, orient='index').to_csv("results/topics.csv")
print("Topics CSV saved: results/topics.csv")

# ---------------------------
# 6. Plots
# ---------------------------
print("Generating WordCloud...")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['clean_headline']))
plt.figure(figsize=(15,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig("results/wordcloud.png")
plt.close()
print("WordCloud saved: results/wordcloud.png")

print("Generating top keywords bar chart...")
words, counts = zip(*top_words)
plt.figure(figsize=(12,6))
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/top_keywords_bar.png")
plt.close()
print("Top keywords bar chart saved: results/top_keywords_bar.png")

print("Text analysis complete! Check the 'results/' folder for CSVs and plots.")
