# Problem Statement – Bitcoin Price Prediction

## 1. Background

Bitcoin (BTC) is the first and largest cryptocurrency by market capitalization. Its price is known for extreme volatility, influenced by:

- Market sentiment and social media
- Macroeconomic factors (interest rates, inflation)
- Regulatory news
- On-chain metrics and adoption
- Liquidity and exchange flows

Traditional linear models often fail to capture the non-linear, regime-switching nature of crypto markets.

## 2. Problem Definition

**Primary Task:**  
Predict the **next-day closing price** of Bitcoin (BTC-USD) using historical OHLCV data and engineered technical indicators.

**Secondary Tasks:**
- Multi-day forecast (illustrative)
- Directional accuracy (up/down)
- Comparison of classical ML vs Deep Learning (LSTM)

## 3. Objectives

1. Collect and preprocess high-quality daily Bitcoin price data.
2. Perform thorough Exploratory Data Analysis.
3. Engineer meaningful technical features (SMA, RSI, MACD, Bollinger, lags, volatility).
4. Implement and compare multiple ML algorithms.
5. Evaluate models using RMSE, MAE, MAPE, R² and Directional Accuracy.
6. Improve models via hyperparameter tuning and architecture search.
7. Build an interactive Streamlit web application.
8. Prepare a clean GitHub repository and deploy the app.

## 4. Scope & Limitations

- **In Scope:** Daily price prediction, technical indicators, classical ML + LSTM, Streamlit UI.
- **Out of Scope:** High-frequency trading, real-time order-book, live trading execution, fundamental on-chain data (for simplicity).

## 5. Success Criteria

- Test-set MAPE < 3%
- Directional Accuracy > 52%
- Fully reproducible pipeline
- Working Streamlit dashboard deployable on Streamlit Cloud

## 6. Ethical & Legal Note

This project is strictly for **educational and academic purposes**.  
It does **not** constitute financial advice. Cryptocurrency investments carry a high risk of capital loss.
