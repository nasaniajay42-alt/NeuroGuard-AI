
import os
import joblib
import pandas as pd

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    ROOT, "backend", "datasets",
    "cleaned_wednesday_dataset.csv"
)

MODEL_DIR = os.path.join(
    ROOT, "backend", "saved_models"
)

print("=" * 60)
print("WEDNESDAY FEATURE COMPATIBILITY CHECK")
print("=" * 60)

df = pd.read_csv(DATA_PATH, nrows=5)
wednesday_features = [
    col for col in df.columns if col != "Label"
]

models = {
    "XGBoost": "xgboost_features.pkl",
    "Random Forest": "adaptive_features.pkl",
    "Autoencoder": "robust_features.pkl"
}

for model_name, filename in models.items():

    model_path = os.path.join(MODEL_DIR, filename)

    model_features = joblib.load(model_path)

    print(f"\n{model_name}")
    print("-" * 40)

    print("Wednesday features:", len(wednesday_features))
    print("Model features:", len(model_features))

    same_names = set(wednesday_features) == set(model_features)
    same_order = wednesday_features == list(model_features)

    print("Same feature names:", same_names)
    print("Same feature order:", same_order)

    missing = [
        f for f in model_features
        if f not in wednesday_features
    ]

    extra = [
        f for f in wednesday_features
        if f not in model_features
    ]

    print("Missing features:", missing)
    print("Extra features:", extra)

    if same_order:
        print("STATUS: COMPATIBLE")
    else:
        print("STATUS: NOT COMPATIBLE")

print("\n" + "=" * 60)
print("COMPATIBILITY CHECK COMPLETED")
print("=" * 60)
