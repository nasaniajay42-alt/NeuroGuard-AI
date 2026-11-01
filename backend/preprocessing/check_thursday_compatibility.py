
import os
import joblib
import pandas as pd

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(ROOT, "backend", "datasets")
MODEL_DIR = os.path.join(ROOT, "backend", "saved_models")

datasets = {
    "Web Attacks": "cleaned_thursday_webattacks.csv",
    "Infiltration": "cleaned_thursday_infiltration.csv"
}

models = {
    "XGBoost": "xgboost_features.pkl",
    "Random Forest": "adaptive_features.pkl",
    "Autoencoder": "robust_features.pkl"
}

print("=" * 65)
print("THURSDAY FEATURE COMPATIBILITY")
print("=" * 65)

for dataset_name, filename in datasets.items():

    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path, nrows=5)

    features = [
        col for col in df.columns if col != "Label"
    ]

    print("\nDATASET:", dataset_name)
    print("-" * 55)
    print("Dataset features:", len(features))

    for model_name, model_file in models.items():

        model_features = joblib.load(
            os.path.join(MODEL_DIR, model_file)
        )

        same_names = set(features) == set(model_features)
        same_order = features == list(model_features)

        print("\nModel:", model_name)
        print("Model features:", len(model_features))
        print("Same names:", same_names)
        print("Same order:", same_order)

        if same_order:
            print("STATUS: COMPATIBLE")
        else:
            print("STATUS: NOT COMPATIBLE")

print("\n" + "=" * 65)
print("COMPATIBILITY CHECK COMPLETED")
print("=" * 65)
