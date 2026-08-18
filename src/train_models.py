"""
src/train_models.py
Pair B pipeline (locked v3 + Patch 1, 2, 5, 7):
Logistic Regression (scaled) + Random Forest + XGBoost, class weights as default,
dev-subsample validation before full training, model selection by
macro F1 + rare-class recall, frozen predict() output format.
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
LABEL_COL = "Label"

ULTRA_RARE = {"Heartbleed", "Infiltration", "Web Attack - SQL Injection"}


def load_data():
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    with open(PROCESSED_DIR / "data_contract.json") as f:
        contract = json.load(f)
    feature_cols = contract["feature_columns"]
    return train_df, test_df, feature_cols


def build_models(class_weight_dict):
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight=class_weight_dict,
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight=class_weight_dict,
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=RANDOM_SEED, n_jobs=-1, tree_method="hist",
    )
    return {"LogisticRegression": lr, "RandomForest": rf, "XGBoost": xgb}


def evaluate(model_name, model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    macro_f1 = f1_score(y_test_labels, y_pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test_labels, y_pred_labels, average="weighted", zero_division=0)

    report = classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=0)
    rare_recalls = {c: report[c]["recall"] for c in ULTRA_RARE if c in report}
    rare_class_recall = recall_score(
        y_test_labels, y_pred_labels,
        labels=[c for c in label_encoder.classes_ if c not in ("BENIGN",) and c not in ULTRA_RARE],
        average="macro", zero_division=0,
    )

    print(f"\n=== {model_name} ===")
    print(f"Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f} | "
          f"Rare-attack-class macro recall: {rare_class_recall:.4f}")
    print(f"Ultra-rare classes (statistical caveat - too few samples for a stable estimate): {rare_recalls}")

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "rare_class_recall": rare_class_recall,
        "ultra_rare_recalls": rare_recalls,
        "full_report": report,
    }


def run_stage(train_df, test_df, feature_cols, stage_name, save_models=False):
    X_train = train_df[feature_cols]
    y_train_raw = train_df[LABEL_COL]
    X_test = test_df[feature_cols]
    y_test_raw = test_df[LABEL_COL]

    le = LabelEncoder()
    le.fit(pd.concat([y_train_raw, y_test_raw]))
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    class_weight_dict = "balanced"

    models = build_models(class_weight_dict)
    results = {}
    trained = {}

    for name, model in models.items():
        t0 = time.time()
        if name == "XGBoost":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        elapsed = time.time() - t0

        metrics = evaluate(name, model, X_test, y_test, le)
        metrics["train_seconds"] = elapsed
        results[name] = metrics
        trained[name] = model
        print(f"Trained in {elapsed:.1f}s")

    print(f"\n--- {stage_name} summary (sorted by macro F1 + rare-class recall) ---")
    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1]["macro_f1"] + kv[1]["rare_class_recall"]),
        reverse=True,
    )
    for name, m in ranked:
        print(f"{name}: macro_f1={m['macro_f1']:.4f}  rare_recall={m['rare_class_recall']:.4f}  "
              f"weighted_f1={m['weighted_f1']:.4f}  time={m['train_seconds']:.1f}s")

    if save_models:
        best_name = ranked[0][0]
        print(f"\nBest model by selection criterion: {best_name}")
        for name, model in trained.items():
            joblib.dump(model, MODELS_DIR / f"{name.lower()}_model.joblib")
        joblib.dump(le, MODELS_DIR / "label_encoder.joblib")

        metadata = {
            "feature_columns": feature_cols,
            "classes": le.classes_.tolist(),
            "best_model": best_name,
            "model_selection_criterion": "macro_f1 + rare_class_recall (primary), weighted_f1 (secondary), train_seconds (tiebreaker)",
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "full_report"} for k, v in results.items()},
        }
        with open(MODELS_DIR / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        with open(RESULTS_DIR / "classification_reports.json", "w") as f:
            json.dump({k: v["full_report"] for k, v in results.items()}, f, indent=2)

    return results, trained, le


def predict(feature_row: dict, model, label_encoder, feature_cols, model_name: str) -> dict:
    X = pd.DataFrame([feature_row])[feature_cols]
    proba = model.predict_proba(X)[0]
    classes = label_encoder.classes_
    prob_dict = {cls: float(p) for cls, p in zip(classes, proba)}

    pred_idx = int(np.argmax(proba))
    attack_category = classes[pred_idx]
    confidence = float(proba[pred_idx])
    p_benign = prob_dict.get("BENIGN", 0.0)
    attack_confidence = 1.0 - p_benign

    return {
        "attack_category": attack_category,
        "confidence": confidence,
        "attack_confidence": attack_confidence,
        "probabilities": prob_dict,
        "model_name": model_name,
    }


def main():
    train_df, test_df, feature_cols = load_data()

    print("=" * 60)
    print("STAGE 1: Dev-subsample validation (150K rows, stratified)")
    print("=" * 60)
    from sklearn.model_selection import train_test_split as _tts
    dev_frac = min(1.0, 150_000 / len(train_df))
    if dev_frac < 1.0:
        dev_sample, _ = _tts(
            train_df, train_size=dev_frac,
            stratify=train_df[LABEL_COL], random_state=RANDOM_SEED,
        )
    else:
        dev_sample = train_df.copy()
    dev_test_sample = test_df.sample(min(len(test_df), 40_000), random_state=RANDOM_SEED)
    run_stage(dev_sample, dev_test_sample, feature_cols, "Dev-subsample", save_models=False)

    proceed = input("\nDev-subsample stage looks correct? Type 'yes' to proceed to full training: ")
    if proceed.strip().lower() != "yes":
        print("Stopped before full training - fix issues above and re-run.")
        return

    print("\n" + "=" * 60)
    print("STAGE 2: Full training (all rows)")
    print("=" * 60)
    results, trained, le = run_stage(train_df, test_df, feature_cols, "Full dataset", save_models=True)

    print("\nSaved to models/: logisticregression_model.joblib, randomforest_model.joblib, "
          "xgboost_model.joblib, label_encoder.joblib, model_metadata.json")
    print("Saved to results/: classification_reports.json")


if __name__ == "__main__":
    main()