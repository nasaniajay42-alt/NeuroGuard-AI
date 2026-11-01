import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
DATASET_PATH = os.path.join(
    BASE_DIR, "datasets", "cleaned_ddos_dataset.csv"
)

# Load trained XGBoost model and feature names
model = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_classifier.pkl")
)

features = joblib.load(
    os.path.join(MODEL_DIR, "xgboost_features.pkl")
)

# Load a sample of the dataset
df = pd.read_csv(DATASET_PATH)

X = df[features].sample(
    n=min(1000, len(df)),
    random_state=42
)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Create output directory
OUTPUT_DIR = os.path.join(BASE_DIR, "research", "experiments")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global feature importance plot
shap.summary_plot(
    shap_values,
    X,
    show=False
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "shap_summary.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("SHAP analysis completed successfully.")
print("Feature importance chart saved to:")
print(os.path.join(OUTPUT_DIR, "shap_summary.png"))