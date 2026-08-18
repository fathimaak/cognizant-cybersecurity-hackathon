"""
src/evaluate_models.py
Patch 6 rigor: confusion matrix + full per-class report, saved as artifacts
for the dashboard and for docs/pair_b_report.md / pair_c_report.md.
"""

import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    with open(PROCESSED_DIR / "data_contract.json") as f:
        contract = json.load(f)
    feature_cols = contract["feature_columns"]

    with open(MODELS_DIR / "model_metadata.json") as f:
        model_meta = json.load(f)
    best_name = model_meta["best_model"]
    clf = joblib.load(MODELS_DIR / f"{best_name.lower()}_model.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")

    X_test = test_df[feature_cols]
    y_test = le.transform(test_df["Label"])
    y_pred = clf.predict(X_test)
    classes = le.classes_

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'confusion_matrix.png'}")

    report = classification_report(
        le.inverse_transform(y_test), le.inverse_transform(y_pred),
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(RESULTS_DIR / "per_class_recall.csv")
    print(f"Saved {RESULTS_DIR / 'per_class_recall.csv'}")
    print(report_df)


if __name__ == "__main__":
    main()