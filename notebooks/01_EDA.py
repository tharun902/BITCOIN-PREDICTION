"""
01_EDA.py – Exploratory Data Analysis for Bitcoin Price Prediction
Run this script or convert to Jupyter notebook:
    jupyter nbconvert --to notebook --execute 01_EDA.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import load_data, add_technical_features

# Style
sns.set_theme(style="darkgrid", palette="Oranges")
plt.rcParams["figure.figsize"] = (12, 5)

df = load_data()
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# 1. Price over time
fig, ax = plt.subplots()
ax.plot(df["Date"], df["Close"], color="#F7931A", linewidth=1.2)
ax.set_title("Bitcoin Closing Price (2015–2025)")
ax.set_ylabel("USD")
ax.set_xlabel("Date")
fig.tight_layout()
fig.savefig(ROOT / "images" / "price_trend.png", dpi=120)
plt.close()

# 2. Volume
fig, ax = plt.subplots()
ax.bar(df["Date"], df["Volume"], color="#F7931A", alpha=0.7, width=1)
ax.set_title("Daily Trading Volume")
ax.set_ylabel("Volume")
fig.tight_layout()
fig.savefig(ROOT / "images" / "volume.png", dpi=120)
plt.close()

# 3. Returns distribution
returns = df["Close"].pct_change().dropna() * 100
fig, ax = plt.subplots()
sns.histplot(returns, bins=80, kde=True, color="#F7931A", ax=ax)
ax.set_title("Distribution of Daily Returns (%)")
ax.set_xlabel("Daily Return %")
fig.tight_layout()
fig.savefig(ROOT / "images" / "returns_dist.png", dpi=120)
plt.close()

# 4. Correlation heatmap
df_feat = add_technical_features(df)
cols = ["Close", "Volume", "RSI_14", "MACD", "Volatility_14", "SMA_50", "SMA_200"]
cols = [c for c in cols if c in df_feat.columns]
corr = df_feat[cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="Oranges", ax=ax)
ax.set_title("Feature Correlation Heatmap")
fig.tight_layout()
fig.savefig(ROOT / "images" / "correlation.png", dpi=120)
plt.close()

print("EDA plots saved to images/")
print("Done.")
