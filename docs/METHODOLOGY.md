# Methodology

## 1. Data Acquisition
- Source: Yahoo Finance via `yfinance` library (`BTC-USD`)
- Period: 2015-01-01 to latest available date
- Frequency: Daily OHLCV
- Alternative Kaggle sources that can be substituted:
  - Zielak – Bitcoin Historical Data (1-min, very large)
  - CoinMarketCap historical CSVs

## 2. Preprocessing
1. Parse dates and sort chronologically
2. Handle missing values (forward fill)
3. Feature Engineering (see below)
4. Time-series split (70% train / 15% val / 15% test) – **no shuffling**
5. Feature scaling (MinMaxScaler fitted only on training set)

## 3. Feature Engineering
- Returns & Log-returns
- Simple & Exponential Moving Averages (7, 14, 21, 50, 100, 200)
- RSI (14)
- MACD + Signal + Histogram
- Bollinger Bands (20, 2σ) + Bandwidth
- ATR (14)
- Rolling volatility
- Volume change & SMA
- Price range features
- Lagged Close and Returns (1–7 days)

## 4. Models Implemented
| Model | Category | Hyperparameters |
|-------|----------|----------------|
| Linear Regression | Baseline | – |
| Ridge / Lasso | Regularized Linear | α |
| Random Forest | Bagging Ensemble | n_estimators, max_depth |
| Gradient Boosting | Boosting | lr, depth, n_estimators |
| XGBoost | Advanced Boosting | lr, depth, subsample |
| LSTM (2-layer) | Deep Learning | units, dropout, lookback |

## 5. Evaluation Metrics
- RMSE, MAE, MAPE, R²
- Directional Accuracy

## 6. Model Improvement Techniques
- RandomizedSearchCV for tree-based models
- Early stopping + ReduceLROnPlateau for LSTM
- Feature importance analysis
- Walk-forward style validation mindset

## 7. Application Layer
- Streamlit multi-page dashboard
- Interactive Plotly charts
- Live next-day prediction
- CSV download of forecasts

## 8. Deployment
- Streamlit Cloud (recommended)
- Requirements pinned in `requirements.txt`
- All heavy computation done offline; app only loads artifacts
