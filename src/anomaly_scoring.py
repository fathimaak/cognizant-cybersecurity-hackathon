"""
src/anomaly_scoring.py
Pair C pipeline: Isolation Forest (BENIGN-only) + locked threat-scoring formula.

Design notes (write these into docs/pair_c_report.md as-is):
- Isolation Forest is trained ONLY on BENIGN rows from train.parquet - it learns what
  "normal" looks like, so it can flag deviations even for attack types not in training.
- Normalization method is locked here and never changes: raw decision_function scores
  (higher = more normal, sklearn's convention) are inverted and scaled using the 1st/99th
  percentile of BENIGN training scores as anchors, then clipped to [0, 1]. Percentiles
  instead of true min/max avoid a single extreme outlier distorting the whole scale.
- Isolation Forest has no "accuracy" - it's unsupervised. We report anomaly rate on
  known-BENIGN rows and detection rate on known-attack rows instead.
- Threat score formula (locked, do not change during the hackathon):
      threat_score = 100 * (0.50 * attack_confidence + 0.30 * anomaly_score + 0.20 * category_impact)
      attack_confidence = 1 - P(BENIGN), computed once in Niya's predict()
- category_impact is this team's own designed severity-by-attack-type weighting,
  not an industry standard - documented explicitly so nobody states it as one in Q&A.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
LABEL_COL = "Label"
BENIGN_LABEL = "BENIGN"

# This team's own designed impact weighting per attack category (0 = none, 1 = severe).
# Document in pair_c_report.md: NOT an industry standard, this team's judgment call.
CATEGORY_IMPACT = {
    "BENIGN": 0.0,
    "DDoS": 1.0,
    "DoS Hulk": 0.85,
    "DoS GoldenEye": 0.85,
    "DoS Slowloris": 0.7,
    "DoS Slowhttptest": 0.7,
    "PortScan": 0.4,
    "Bot": 0.8,
    "Infiltration": 1.0,
    "Heartbleed": 1.0,
    "FTP-Patator": 0.6,
    "SSH-Patator": 0.6,
    "Web Attack - Brute Force": 0.7,
    "Web Attack - XSS": 0.75,
    "Web Attack - SQL Injection": 0.9,
}

# Severity bands - this team's own thresholds on the 0-100 threat score.
SEVERITY_BANDS = [
    (0, 25, "Low"),
    (25, 50, "Medium"),
    (50, 75, "High"),
    (75, 101, "Critical"),
]


def severity_from_score(score: float) -> str:
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= score < hi:
            return label
    return "Critical"


def train_isolation_forest(train_df: pd.DataFrame, feature_cols):
    benign_train = train_df[train_df[LABEL_COL] == BENIGN_LABEL][feature_cols]
    print(f"Training Isolation Forest on {len(benign_train):,} BENIGN-only rows")

    iso = IsolationForest(
        n_estimators=200, contamination="auto",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    iso.fit(benign_train)

    # lock normalization anchors from BENIGN training scores - never recompute later
    raw_scores = iso.decision_function(benign_train)
    p1, p99 = np.percentile(raw_scores, [1, 99])
    print(f"Normalization anchors locked: p1={p1:.4f}, p99={p99:.4f}")

    return iso, {"p1": float(p1), "p99": float(p99)}


def anomaly_score(iso_model, X: pd.DataFrame, anchors: dict) -> np.ndarray:
    """Higher = more anomalous. Locked method - do not change during the hackathon."""
    raw = iso_model.decision_function(X)  # higher raw = more normal
    p1, p99 = anchors["p1"], anchors["p99"]
    scaled = (p99 - raw) / (p99 - p1)  # invert + scale
    return np.clip(scaled, 0.0, 1.0)


def evaluate_separation(iso_model, test_df, feature_cols, anchors):
    """No 'accuracy' for Isolation Forest - report separation instead (Patch 6)."""
    benign_test = test_df[test_df[LABEL_COL] == BENIGN_LABEL][feature_cols]
    attack_test = test_df[test_df[LABEL_COL] != BENIGN_LABEL][feature_cols]

    benign_scores = anomaly_score(iso_model, benign_test, anchors)
    attack_scores = anomaly_score(iso_model, attack_test, anchors)

    # Binary evaluation: BENIGN = 0, attack = 1
    y_true = np.concatenate([
        np.zeros(len(benign_scores)),
        np.ones(len(attack_scores))
    ])

    y_score = np.concatenate([
        benign_scores,
        attack_scores
    ])

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    print(f"\nSeparation check (NOT 'accuracy' - Isolation Forest is unsupervised):")
    print(f"  Known-BENIGN anomaly score:  mean={benign_scores.mean():.4f}  median={np.median(benign_scores):.4f}")
    print(f"  Known-attack anomaly score:  mean={attack_scores.mean():.4f}  median={np.median(attack_scores):.4f}")
    print(f"  Detection rate (attack rows scoring > 0.5): "
          f"{(attack_scores > 0.5).mean() * 100:.1f}%")
    print(f"  False-alarm rate (BENIGN rows scoring > 0.5): "
          f"{(benign_scores > 0.5).mean() * 100:.1f}%")
    print(f"  ROC-AUC (BENIGN vs attack): {roc_auc:.4f}")
    print(f"  PR-AUC (BENIGN vs attack):  {pr_auc:.4f}")


def score_event(classifier_predict_output: dict, iso_anomaly_score: float) -> dict:
    """
    score_event() - Pair C's frozen interface.
    classifier_predict_output: the dict returned by Niya's predict() (Patch 7 format)
    iso_anomaly_score: a single float from anomaly_score(), already 0-1
    """
    attack_confidence = classifier_predict_output["attack_confidence"]
    predicted_category = classifier_predict_output["attack_category"]
    category_impact = CATEGORY_IMPACT.get(predicted_category, 0.5)  # 0.5 default for unmapped categories

    threat_score = 100 * (
        0.50 * attack_confidence +
        0.30 * iso_anomaly_score +
        0.20 * category_impact
    )
    threat_score = float(np.clip(threat_score, 0, 100))
    severity = severity_from_score(threat_score)

    return {
        "anomaly_score": float(iso_anomaly_score),
        "threat_score": threat_score,
        "severity": severity,
    }


def sanity_check(train_df, test_df, feature_cols, iso_model, anchors):
    """Load Niya's best model and run a few known examples through the full chain."""
    with open(MODELS_DIR / "model_metadata.json") as f:
        model_meta = json.load(f)
    best_name = model_meta["best_model"]
    clf = joblib.load(MODELS_DIR / f"{best_name.lower()}_model.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")

    print(f"\n--- Full chain sanity check using {best_name} ---")
    for label in ["BENIGN", "DDoS", "PortScan", "Web Attack - SQL Injection"]:
        rows = test_df[test_df[LABEL_COL] == label]
        if len(rows) == 0:
            continue
        X = rows.iloc[[0]][feature_cols].astype(float)

        proba = clf.predict_proba(X)[0]
        classes = le.classes_
        prob_dict = {c: float(p) for c, p in zip(classes, proba)}
        pred_idx = int(np.argmax(proba))
        clf_output = {
            "attack_category": classes[pred_idx],
            "confidence": float(proba[pred_idx]),
            "attack_confidence": 1.0 - prob_dict.get("BENIGN", 0.0),
            "probabilities": prob_dict,
            "model_name": best_name,
        }

        a_score = anomaly_score(iso_model, X, anchors)[0]
        result = score_event(clf_output, a_score)

        print(f"True label: {label:30s} | predicted: {clf_output['attack_category']:25s} | "
              f"anomaly={a_score:.3f} | threat_score={result['threat_score']:.1f} | severity={result['severity']}")


def main():
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    with open(PROCESSED_DIR / "data_contract.json") as f:
        contract = json.load(f)
    feature_cols = contract["feature_columns"]

    iso_model, anchors = train_isolation_forest(train_df, feature_cols)
    evaluate_separation(iso_model, test_df, feature_cols, anchors)

    joblib.dump(iso_model, MODELS_DIR / "isolation_forest_model.joblib")
    with open(MODELS_DIR / "anomaly_normalization.json", "w") as f:
        json.dump(anchors, f, indent=2)
    with open(MODELS_DIR / "category_impact.json", "w") as f:
        json.dump(CATEGORY_IMPACT, f, indent=2)

    sanity_check(train_df, test_df, feature_cols, iso_model, anchors)

    print("\nSaved: models/isolation_forest_model.joblib, "
          "models/anomaly_normalization.json, models/category_impact.json")


if __name__ == "__main__":
    main()