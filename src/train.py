"""
Train classical ML models for Bitcoin Price Prediction
"""

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from preprocess import full_pipeline


def evaluate(y_true, y_pred, name="Model"):
    """Compute regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy
    direction_true = np.sign(np.diff(y_true, prepend=y_true[0]))
    direction_pred = np.sign(y_pred - np.roll(y_true, 1))
    direction_pred[0] = 0
    dir_acc = np.mean(direction_true[1:] == direction_pred[1:]) * 100

    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
        "Directional_Accuracy": float(dir_acc),
    }
    print(f"\n{name}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}")
    print(f"  Dir.Acc: {dir_acc:.1f}%")
    return metrics


def train_all_models(data=None, tune=True):
    """Train multiple models and select the best one."""
    if data is None:
        data = full_pipeline()

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=5000),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )

    results = {}
    best_name = None
    best_score = float("inf")
    best_model = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)
        metrics = evaluate(y_val, y_pred_val, name=f"{name} (Validation)")
        results[name] = metrics

        if metrics["RMSE"] < best_score:
            best_score = metrics["RMSE"]
            best_name = name
            best_model = model

    # Optional light hyperparameter tuning for the best classical model
    if tune and best_name in ["RandomForest", "XGBoost", "GradientBoosting"]:
        print(f"\n>>> Tuning best model so far: {best_name}")
        if best_name == "RandomForest":
            param_dist = {
                "n_estimators": [150, 200, 300],
                "max_depth": [8, 10, 12, 15],
                "min_samples_leaf": [3, 5, 8],
            }
            base = RandomForestRegressor(random_state=42, n_jobs=-1)
        elif best_name == "XGBoost" and HAS_XGB:
            param_dist = {
                "n_estimators": [200, 300, 400],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.08],
                "subsample": [0.7, 0.8, 0.9],
            }
            base = XGBRegressor(random_state=42, n_jobs=-1)
        else:
            param_dist = {
                "n_estimators": [100, 150, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.05, 0.1],
            }
            base = GradientBoostingRegressor(random_state=42)

        search = RandomizedSearchCV(
            base, param_distributions=param_dist, n_iter=12,
            scoring="neg_root_mean_squared_error", cv=3, random_state=42, n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
        y_pred_val = best_model.predict(X_val)
        metrics = evaluate(y_val, y_pred_val, name=f"{best_name}_Tuned (Validation)")
        results[f"{best_name}_Tuned"] = metrics
        best_name = f"{best_name}_Tuned"

    # Final evaluation on test set
    y_pred_test = best_model.predict(X_test)
    test_metrics = evaluate(y_test, y_pred_test, name=f"BEST MODEL ({best_name}) - Test")
    results["BEST_TEST"] = test_metrics
    results["best_model_name"] = best_name

    # Save artifacts
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, models_dir / "best_model.joblib")
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Best model ({best_name}) saved to models/best_model.joblib")
    return best_model, results


if __name__ == "__main__":
    train_all_models(tune=True)
