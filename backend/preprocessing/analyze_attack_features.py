import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

path = os.path.join(
    BASE_DIR,
    "datasets",
    "cleaned_tuesday_dataset.csv"
)

df = pd.read_csv(path)

features = [
    "Fwd IAT Min",
    "Bwd IAT Min",
    "Bwd IAT Mean",
    "Fwd IAT Mean",
    "Flow IAT Mean",
    "Flow Packets/s",
    "FIN Flag Count"
]

labels = ["BENIGN", "FTP-Patator", "SSH-Patator"]

for label in labels:

    subset = df[df["Label"].str.strip() == label]

    print(f"\n========== {label} ==========")
    print("Records:", len(subset))

    for feature in features:

        values = pd.to_numeric(
            subset[feature], errors="coerce"
        )

        print(
            f"{feature}: "
            f"Median={values.median():.4f}, "
            f"P95={values.quantile(0.95):.4f}, "
            f"P99={values.quantile(0.99):.4f}, "
            f"Negative={(values < 0).sum()}"
        )