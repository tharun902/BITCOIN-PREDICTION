"""
One-click full pipeline:
1. Preprocess data
2. Train classical ML models
3. Train LSTM (if TensorFlow available)
4. Save all artifacts for Streamlit app
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

print("=" * 60)
print("₿  Bitcoin Price Prediction – Full Pipeline")
print("=" * 60)

print("\n[1/3] Running preprocessing & feature engineering...")
from preprocess import full_pipeline
data = full_pipeline()
print(f"   Features: {len(data['feature_cols'])}")
print(f"   Train / Val / Test: {len(data['y_train'])} / {len(data['y_val'])} / {len(data['y_test'])}")

print("\n[2/3] Training classical ML models...")
from train import train_all_models
best_model, metrics = train_all_models(data=data, tune=True)

print("\n[3/3] Training LSTM (optional)...")
try:
    from lstm_model import train_lstm
    train_lstm(data=data, epochs=30)
except Exception as e:
    print(f"   LSTM skipped or failed: {e}")

print("\n" + "=" * 60)
print("✅ Pipeline finished successfully!")
print("   Artifacts saved in models/")
print("   Launch app with:  streamlit run app/streamlit_app.py")
print("=" * 60)
