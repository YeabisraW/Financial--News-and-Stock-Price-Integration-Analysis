# Financial News and Stock Price Analysis

## Project Overview
This project integrates financial news and stock price data to perform exploratory data analysis, text analysis, time series analysis, publisher analysis, and quantitative stock analysis.  
The goal is to extract insights such as common keywords, publication trends, top publishers, and technical indicators to support trading decisions.

---

## Scripts Overview
All scripts are located in the `scripts/` folder.

| Script | Purpose | Output |
|--------|---------|--------|
| `headlin_eda.py` | Exploratory Data Analysis (EDA) | CSV summaries, basic plots in `results/` |
| `text_analysis.py` | Text analysis and topic modeling | `results/top_keywords.csv`, `results/top_phrases.csv`, `results/topics.csv`, wordcloud & bar plots |
| `time_series_analysis.py` | Analyze publication frequency over time | `results/publication_counts.csv`, `results/publication_counts.png` |
| `publisher_analysis.py` | Publisher contribution and domain analysis | `results/publisher_counts.csv`, `results/publisher_domains.csv`, `results/top_publishers_bar.png` |
---

## Required Libraries
The project requires the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `nltk`
- `scikit-learn`
- `wordcloud`
- `TA-Lib` (`talib`)

### Installing Libraries
```bash
pip install pandas numpy matplotlib nltk scikit-learn wordcloud
# TA-Lib installation (Windows example)
pip install TA-Lib

