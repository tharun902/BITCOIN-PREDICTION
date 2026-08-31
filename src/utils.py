"""
Utility helpers for Bitcoin Price Prediction project
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def load_metrics():
    path = get_project_root() / "models" / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_best_model():
    path = get_project_root() / "models" / "best_model.joblib"
    if path.exists():
        return joblib.load(path)
    return None


def load_scaler_and_features():
    root = get_project_root() / "models"
    scaler = joblib.load(root / "scaler.joblib") if (root / "scaler.joblib").exists() else None
    cols = joblib.load(root / "feature_cols.joblib") if (root / "feature_cols.joblib").exists() else None
    return scaler, cols


def directional_accuracy(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(y_pred[1:] - y_true[:-1])
    return float(np.mean(true_dir == pred_dir) * 100)
