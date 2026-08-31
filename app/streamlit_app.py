"""
Bitcoin Price Prediction - Streamlit Dashboard
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json

st.set_page_config(
    page_title="₿ Bitcoin Price Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #F7931A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #F7931A;
    }
    .stMetric > label {
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_raw_data():
    path = ROOT / "data" / "bitcoin_historical.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_resource
def load_artifacts():
    models_dir = ROOT / "models"
    artifacts = {}
    if (models_dir / "best_model.joblib").exists():
        artifacts["model"] = joblib.load(models_dir / "best_model.joblib")
    if (models_dir / "scaler.joblib").exists():
        artifacts["scaler"] = joblib.load(models_dir / "scaler.joblib")
    if (models_dir / "feature_cols.joblib").exists():
        artifacts["feature_cols"] = joblib.load(models_dir / "feature_cols.joblib")
    if (models_dir / "metrics.json").exists():
        with open(models_dir / "metrics.json") as f:
            artifacts["metrics"] = json.load(f)
    return artifacts


def add_features_for_prediction(df):
    """Lightweight feature engineering for live prediction."""
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    for w in [7, 14, 21, 50, 100, 200]:
        df[f"SMA_{w}"] = df["Close"].rolling(w).mean()
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    df["Volatility_14"] = df["Return"].rolling(14).std()
    df["Volatility_30"] = df["Return"].rolling(30).std()
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_SMA_7"] = df["Volume"].rolling(7).mean()
    df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["OC_Range"] = (df["Close"] - df["Open"]) / df["Open"]

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger
    df["BB_Mid"] = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_High"] = df["BB_Mid"] + 2 * std20
    df["BB_Low"] = df["BB_Mid"] - 2 * std20
    df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / (df["BB_Mid"] + 1e-10)

    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    for lag in [1, 2, 3, 5, 7]:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)

    return df


# ---------------- SIDEBAR ----------------
st.sidebar.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=80)
st.sidebar.title("₿ Controls")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "📊 EDA & Charts", "🤖 Model Performance", "🔮 Predict", "ℹ️ About"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This is an **educational** project.\n\n"
    "Not financial advice. Crypto trading involves high risk."
)

# ---------------- LOAD DATA ----------------
df = load_raw_data()
artifacts = load_artifacts()

# ---------------- PAGES ----------------
if page == "🏠 Overview":
    st.markdown('<p class="main-header">₿ Bitcoin Price Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">End-to-End Machine Learning Project | Streamlit Dashboard</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    change = latest["Close"] - prev["Close"]
    pct = (change / prev["Close"]) * 100

    col1.metric("Latest Close", f"${latest['Close']:,.2f}", f"{change:+,.2f} ({pct:+.2f}%)")
    col2.metric("All-Time High (in data)", f"${df['High'].max():,.2f}")
    col3.metric("Data Points", f"{len(df):,}")
    col4.metric("Period", f"{df['Date'].min().date()} → {df['Date'].max().date()}")

    st.markdown("### 📈 Price History")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        mode="lines", name="Close",
        line=dict(color="#F7931A", width=2)
    ))
    fig.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🎯 Project Goals")
    st.markdown("""
    - Forecast next-day Bitcoin closing price using classical ML & Deep Learning
    - Compare Linear Regression, Random Forest, XGBoost and LSTM
    - Interactive dashboard for EDA, evaluation and live prediction
    - Fully reproducible pipeline ready for GitHub & Streamlit Cloud deployment
    """)

elif page == "📊 EDA & Charts":
    st.header("📊 Exploratory Data Analysis")

    # Candlestick
    st.subheader("Candlestick Chart (last 365 days)")
    recent = df.tail(365)
    fig = go.Figure(data=[go.Candlestick(
        x=recent["Date"],
        open=recent["Open"], high=recent["High"],
        low=recent["Low"], close=recent["Close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    )])
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Volume
    st.subheader("Trading Volume")
    fig2 = px.bar(recent, x="Date", y="Volume", template="plotly_dark")
    fig2.update_traces(marker_color="#F7931A")
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)

    # Returns distribution
    st.subheader("Daily Returns Distribution")
    returns = df["Close"].pct_change().dropna() * 100
    fig3 = px.histogram(returns, nbins=80, template="plotly_dark",
                        labels={"value": "Daily Return (%)"})
    fig3.update_traces(marker_color="#F7931A")
    fig3.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        st.metric("Mean Daily Return", f"{returns.mean():.3f}%")
        st.metric("Std Dev (Volatility)", f"{returns.std():.3f}%")
    with colB:
        st.metric("Best Day", f"+{returns.max():.2f}%")
        st.metric("Worst Day", f"{returns.min():.2f}%")

    # Correlation of basic features
    st.subheader("Feature Correlation (OHLCV)")
    corr = df[["Open", "High", "Low", "Close", "Volume"]].corr()
    fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="Oranges",
                     template="plotly_dark", aspect="auto")
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

elif page == "🤖 Model Performance":
    st.header("🤖 Model Evaluation")

    if "metrics" not in artifacts:
        st.warning("Models not trained yet. Run `python run_pipeline.py` first.")
        st.code("python run_pipeline.py", language="bash")
    else:
        metrics = artifacts["metrics"]
        best_name = metrics.get("best_model_name", "Unknown")
        st.success(f"**Best Model:** `{best_name}`")

        # Summary table
        rows = []
        for k, v in metrics.items():
            if isinstance(v, dict) and "RMSE" in v:
                rows.append({
                    "Model": k,
                    "RMSE": round(v["RMSE"], 2),
                    "MAE": round(v["MAE"], 2),
                    "MAPE (%)": round(v["MAPE"], 2),
                    "R²": round(v["R2"], 4),
                    "Dir. Acc (%)": round(v.get("Directional_Accuracy", 0), 1),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if "BEST_TEST" in metrics:
            st.subheader("🏆 Best Model – Test Set Metrics")
            t = metrics["BEST_TEST"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{t['RMSE']:,.2f}")
            c2.metric("MAE", f"{t['MAE']:,.2f}")
            c3.metric("MAPE", f"{t['MAPE']:.2f}%")
            c4.metric("R²", f"{t['R2']:.4f}")

        st.markdown("""
        ### Evaluation Notes
        - **Time-series split** used (no future leakage)
        - Metrics computed on held-out test set
        - Directional Accuracy measures correct up/down prediction
        """)

elif page == "🔮 Predict":
    st.header("🔮 Next-Day Price Prediction")

    if "model" not in artifacts or "scaler" not in artifacts:
        st.error("Model artifacts missing. Please train the models first:")
        st.code("python run_pipeline.py")
    else:
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        feature_cols = artifacts.get("feature_cols", [])

        # Prepare latest features
        df_feat = add_features_for_prediction(df)
        df_feat = df_feat.dropna().reset_index(drop=True)

        # Align columns
        available = [c for c in feature_cols if c in df_feat.columns]
        if len(available) < len(feature_cols) * 0.7:
            st.warning("Some engineered features missing – prediction may be less accurate.")
            # Use intersection
            feature_cols = available

        last_row = df_feat.iloc[[-1]][feature_cols]
        X_last = scaler.transform(last_row)
        pred = model.predict(X_last)[0]
        current = df_feat.iloc[-1]["Close"]
        change = pred - current
        pct = (change / current) * 100

        st.markdown("### 📅 Latest Available Data")
        st.write(f"**Date:** {df_feat.iloc[-1]['Date'].date()}")
        st.write(f"**Current Close:** ${current:,.2f}")

        st.markdown("### 🎯 Predicted Next-Day Close")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Price", f"${pred:,.2f}")
        col2.metric("Expected Change", f"${change:+,.2f}")
        col3.metric("Expected %", f"{pct:+.2f}%")

        if pct > 0.5:
            st.success("Model leans **BULLISH** for the next day.")
        elif pct < -0.5:
            st.error("Model leans **BEARISH** for the next day.")
        else:
            st.info("Model expects a relatively **SIDEWAYS** move.")

        # Simple multi-step naive forecast (recursive)
        st.subheader("Approximate 7-Day Outlook (Recursive)")
        st.caption("Educational illustration only – not a reliable multi-step forecast.")
        future_prices = [current]
        temp_df = df_feat.copy()
        for i in range(7):
            row = temp_df.iloc[[-1]][feature_cols]
            X = scaler.transform(row)
            p = model.predict(X)[0]
            future_prices.append(p)
            # Append a synthetic row (very rough)
            new_row = temp_df.iloc[-1].copy()
            new_row["Close"] = p
            new_row["Open"] = p
            new_row["High"] = p * 1.01
            new_row["Low"] = p * 0.99
            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
            temp_df = add_features_for_prediction(temp_df)

        days = list(range(8))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=future_prices, mode="lines+markers",
                                 name="Forecast", line=dict(color="#F7931A", width=3)))
        fig.update_layout(template="plotly_dark", height=350,
                          xaxis_title="Days ahead", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        # Download button
        pred_df = pd.DataFrame({
            "Day": days,
            "Predicted_Close": future_prices
        })
        st.download_button(
            "Download Forecast CSV",
            pred_df.to_csv(index=False),
            file_name="btc_forecast.csv",
            mime="text/csv"
        )

else:  # About
    st.header("ℹ️ About This Project")
    st.markdown("""
    ### Bitcoin Price Prediction – Complete Academic Project

    This repository covers **all evaluation criteria** required for a typical final-year / ML course project:

    | Component | Marks |
    |-----------|-------|
    | Problem Identification | 10 |
    | Dataset & Preprocessing | 15 |
    | EDA & Visualization | 10 |
    | ML Algorithm Implementation | 20 |
    | Model Evaluation | 10 |
    | Model Improvement | 10 |
    | Application / UI | 10 |
    | GitHub Repository | 5 |
    | Deployment | 5 |
    | Presentation & Viva | 5 |
    | **Total** | **100** |

    **Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, TensorFlow/Keras, Streamlit, Plotly, yfinance

    **Dataset:** Daily BTC-USD OHLCV from Yahoo Finance (2015 – 2025)

    **How to deploy on Streamlit Cloud:**
    1. Push the entire folder to a public GitHub repository
    2. Visit https://share.streamlit.io
    3. Select the repo and set main file to `app/streamlit_app.py`
    4. Deploy!

    ---
    **Disclaimer:** Educational use only. Not financial advice.
    """)
