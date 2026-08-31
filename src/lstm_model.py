"""
LSTM Model for Bitcoin Price Prediction
"""

import json
import numpy as np
from pathlib import Path
import joblib

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not installed. LSTM training will be skipped.")

from preprocess import full_pipeline
from train import evaluate


def build_lstm(input_shape, units=64, dropout=0.2):
    """Build a 2-layer LSTM model."""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def train_lstm(data=None, epochs=50, batch_size=32, lookback=60):
    """Train LSTM and save the best model."""
    if not HAS_TF:
        print("TensorFlow missing – skipping LSTM.")
        return None, {}

    if data is None:
        data = full_pipeline(lookback=lookback)

    X_train = data["X_train_seq"]
    y_train = data["y_train_seq"]
    X_val = data["X_val_seq"]
    y_val = data["y_val_seq"]
    X_test = data["X_test_seq"]
    y_test = data["y_test_seq"]

    print(f"LSTM input shape: {X_train.shape}")

    model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    ckpt_path = models_dir / "lstm_best.h5"

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_pred_val = model.predict(X_val, verbose=0).flatten()
    val_metrics = evaluate(y_val, y_pred_val, name="LSTM (Validation)")

    y_pred_test = model.predict(X_test, verbose=0).flatten()
    test_metrics = evaluate(y_test, y_pred_test, name="LSTM (Test)")

    # Save final model
    model.save(models_dir / "lstm_model.h5")
    metrics = {
        "LSTM_Validation": val_metrics,
        "LSTM_Test": test_metrics,
        "history": {
            "loss": [float(x) for x in history.history["loss"]],
            "val_loss": [float(x) for x in history.history["val_loss"]],
        }
    }
    with open(models_dir / "lstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ LSTM model saved to models/lstm_model.h5")
    return model, metrics


def predict_next(model, last_sequence, scaler, feature_cols, current_close):
    """Helper to predict next day close from the last lookback window."""
    pred_scaled = model.predict(last_sequence.reshape(1, *last_sequence.shape), verbose=0)
    # Note: because we scaled all features together, inverse transform needs care.
    # For simplicity we return the scaled prediction and let the caller handle inverse.
    return float(pred_scaled[0][0])


if __name__ == "__main__":
    train_lstm(epochs=40)
