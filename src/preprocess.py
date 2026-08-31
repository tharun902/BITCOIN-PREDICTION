"""
Data Loading, Cleaning & Feature Engineering for Bitcoin Price Prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pathlib import Path
import joblib

# Optional technical indicators
try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False


def load_data(filepath: str = None) -> pd.DataFrame:
    """Load Bitcoin historical data from CSV or download via yfinance."""
    if filepath is None:
        filepath = Path(__file__).parent.parent / "data" / "bitcoin_historical.csv"

    filepath = Path(filepath)
    if filepath.exists():
        df = pd.read_csv(filepath)
    else:
        # Fallback: download fresh data
        import yfinance as yf
        df = yf.download("BTC-USD", start="2015-01-01", progress=False)
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Close"])
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators and lag features."""
    df = df.copy()

    # Basic returns
    df["Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Moving Averages
    for window in [7, 14, 21, 50, 100, 200]:
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()

    # Volatility
    df["Volatility_14"] = df["Return"].rolling(window=14).std()
    df["Volatility_30"] = df["Return"].rolling(window=30).std()

    # Volume features
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_SMA_7"] = df["Volume"].rolling(7).mean()

    # Price range
    df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df["OC_Range"] = (df["Close"] - df["Open"]) / df["Open"]

    # RSI (manual implementation if ta not available)
    if HAS_TA:
        df["RSI_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        df["MACD"] = ta.trend.MACD(df["Close"]).macd()
        df["MACD_Signal"] = ta.trend.MACD(df["Close"]).macd_signal()
        df["MACD_Hist"] = ta.trend.MACD(df["Close"]).macd_diff()
        bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()
        df["BB_Mid"] = bb.bollinger_mavg()
        df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]
        df["ATR_14"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"], window=14
        ).average_true_range()
    else:
        # Manual RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss
        df["RSI_14"] = 100 - (100 / (1 + rs))

        # Simple MACD
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
        df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]

        # ATR approximation
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14).mean()

    # Lag features for Close
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df[f"Return_Lag_{lag}"] = df["Return"].shift(lag)

    # Target: Next day Close
    df["Target"] = df["Close"].shift(-1)
    df["Target_Return"] = df["Return"].shift(-1)

    # Direction (1 = up, 0 = down)
    df["Target_Direction"] = (df["Target"] > df["Close"]).astype(int)

    return df


def prepare_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """Select final feature set and clean."""
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Return", "Log_Return",
        "SMA_7", "SMA_21", "SMA_50", "SMA_200",
        "EMA_7", "EMA_21",
        "Volatility_14", "Volatility_30",
        "Volume_Change", "Volume_SMA_7",
        "HL_Range", "OC_Range",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_High", "BB_Low", "BB_Mid", "BB_Width", "ATR_14",
        "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5", "Close_Lag_7",
        "Return_Lag_1", "Return_Lag_2", "Return_Lag_3",
    ]

    # Keep only existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    keep_cols = ["Date"] + feature_cols + ["Target", "Target_Return", "Target_Direction"]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df_clean = df[keep_cols].copy()
    if drop_na:
        df_clean = df_clean.dropna().reset_index(drop=True)
    return df_clean, feature_cols


def time_series_split(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15):
    """Chronological train / val / test split (no shuffle)."""
    n = len(df)
    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - val_size))

    train = df.iloc[:val_start].copy()
    val = df.iloc[val_start:test_start].copy()
    test = df.iloc[test_start:].copy()
    return train, val, test


def scale_features(train, val, test, feature_cols, scaler_type="minmax"):
    """Fit scaler on train only and transform all sets."""
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train = scaler.fit_transform(train[feature_cols])
    X_val = scaler.transform(val[feature_cols])
    X_test = scaler.transform(test[feature_cols])

    y_train = train["Target"].values
    y_val = val["Target"].values
    y_test = test["Target"].values

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def create_sequences(X, y, lookback: int = 60):
    """Create 3D sequences for LSTM: (samples, timesteps, features)."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def full_pipeline(filepath=None, lookback=60, save_scaler=True):
    """Complete preprocessing pipeline used by training scripts."""
    df = load_data(filepath)
    df = add_technical_features(df)
    df_clean, feature_cols = prepare_features(df)
    train, val, test = time_series_split(df_clean)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = scale_features(
        train, val, test, feature_cols
    )

    # Also prepare sequences for LSTM
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, lookback)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback)

    if save_scaler:
        models_dir = Path(__file__).parent.parent / "models"
        models_dir.mkdir(exist_ok=True)
        joblib.dump(scaler, models_dir / "scaler.joblib")
        joblib.dump(feature_cols, models_dir / "feature_cols.joblib")

    return {
        "df": df_clean,
        "feature_cols": feature_cols,
        "train": train,
        "val": val,
        "test": test,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_seq": X_train_seq,
        "X_val_seq": X_val_seq,
        "X_test_seq": X_test_seq,
        "y_train_seq": y_train_seq,
        "y_val_seq": y_val_seq,
        "y_test_seq": y_test_seq,
        "scaler": scaler,
        "lookback": lookback,
    }


if __name__ == "__main__":
    data = full_pipeline()
    print("Preprocessing complete.")
    print(f"Train samples: {len(data['y_train'])}")
    print(f"Val samples:   {len(data['y_val'])}")
    print(f"Test samples:  {len(data['y_test'])}")
    print(f"Features:      {len(data['feature_cols'])}")
    print(f"LSTM sequences (train): {data['X_train_seq'].shape}")
