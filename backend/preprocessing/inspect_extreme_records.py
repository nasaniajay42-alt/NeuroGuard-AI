import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

path = os.path.join(
    BASE_DIR,
    "datasets",
    "cleaned_tuesday_dataset.csv"
)

df = pd.read_csv(path)

features = [
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Flow IAT Mean",
    "Fwd IAT Min",
    "Bwd IAT Min",
    "Flow Packets/s"
]

benign = df[df["Label"].str.strip() == "BENIGN"].copy()

# Find records with very large inter-arrival times
extreme = benign[
    (benign["Fwd IAT Mean"] > 10000000) |
    (benign["Bwd IAT Mean"] > 10000000) |
    (benign["Flow IAT Mean"] > 10000000)
]

print("Extreme BENIGN records:", len(extreme))

print("\nFirst 10 extreme records:")
print(extreme[features].head(10).to_string(index=False))

print("\nExtreme records with multiple large features:")

large_counts = (
    extreme[features[:3]] > 10000000
).sum(axis=1)

print(large_counts.value_counts().sort_index())

print("\nInspection completed.")