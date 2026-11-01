
import os
import pandas as pd

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

EVAL_DIR = os.path.join(ROOT, "research", "evaluation")
EXPERIMENT_DIR = os.path.join(ROOT, "research", "experiments")

os.makedirs(EVAL_DIR, exist_ok=True)

print("=" * 65)
print("NEUROGUARD AI - EXPERIMENT SUMMARY")
print("=" * 65)

# Load existing evaluation reports
files = {
    "Wednesday": "aura_wednesday_model_comparison.csv",
    "Thursday": "aura_thursday_model_comparison.csv"
}

all_results = []

for day, filename in files.items():

    path = os.path.join(EVAL_DIR, filename)

    if os.path.exists(path):

        df = pd.read_csv(path)
        df["Evaluation_Day"] = day

        all_results.append(df)

if all_results:

    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined.to_csv(
        os.path.join(
            EVAL_DIR,
            "neuroguard_combined_evaluation.csv"
        ),
        index=False
    )

    print("\nMODEL PERFORMANCE")
    print(
        combined[
            [
                "Evaluation_Day",
                "Dataset",
                "Model",
                "Precision",
                "Recall",
                "F1",
                "FPR",
                "ROC_AUC",
                "PR_AUC"
            ]
        ].round(4).to_string(index=False)
    )

# Load attack family reports
attack_files = [
    "aura_wednesday_attack_detection.csv",
    "aura_thursday_attack_detection.csv"
]

attack_results = []

for filename in attack_files:

    path = os.path.join(EVAL_DIR, filename)

    if os.path.exists(path):

        df = pd.read_csv(path)
        attack_results.append(df)

if attack_results:

    attacks = pd.concat(
        attack_results,
        ignore_index=True
    )

    attacks.to_csv(
        os.path.join(
            EVAL_DIR,
            "neuroguard_attack_family_summary.csv"
        ),
        index=False
    )

    print("\nATTACK FAMILY PERFORMANCE")

    print(
        attacks.to_string(index=False)
    )

print("\n" + "=" * 65)
print("RESEARCH SUMMARY GENERATED")
print("=" * 65)

print("Output:")
print(
    os.path.join(
        EVAL_DIR,
        "neuroguard_combined_evaluation.csv"
    )
)

print(
    "\nNote: Wednesday was used for threshold calibration. "
    "Thursday was evaluated using frozen thresholds. "
    "These results do not establish universal zero-day detection."
)
