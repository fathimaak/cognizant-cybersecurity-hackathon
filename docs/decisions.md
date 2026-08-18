# Design Decisions

## 1. Supervised + Unsupervised Detection

The system combines supervised classification with unsupervised anomaly detection.

The supervised classifier learns known attack categories from labelled training data. Isolation Forest is trained only on BENIGN traffic so that it learns the characteristics of normal traffic and can provide an additional anomaly signal.

This combination is intentional: the classifier is stronger for known attack categories, while anomaly detection provides an additional signal for unusual traffic patterns.

## 2. Model Selection

Three supervised models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

XGBoost was selected as the final classifier based on the model-selection criteria defined in the project:

1. Macro F1 and rare-class recall as primary criteria
2. Weighted F1 as a secondary criterion
3. Training time as a tiebreaker

The final dashboard therefore uses the selected XGBoost model.

## 3. Isolation Forest

Isolation Forest was selected for anomaly detection because it:

- Does not require labelled attack examples
- Can be trained efficiently on large datasets
- Does not require feature scaling
- Provides a relatively simple and explainable anomaly-detection mechanism

The model is trained using BENIGN-only training data.

## 4. Anomaly Score Normalization

Isolation Forest's `decision_function()` follows the convention that higher values indicate more normal observations.

The system therefore inverts and normalizes the raw scores to produce an anomaly score where:

- `0` represents relatively normal traffic
- `1` represents highly anomalous traffic

The 1st and 99th percentiles of BENIGN training scores are used as normalization anchors. The resulting score is clipped to the `[0, 1]` range.

The normalization anchors are saved in `models/anomaly_normalization.json` so that the same normalization is used consistently during later predictions.

## 5. Threat Score

The final threat score combines three signals:

```text
threat_score =
100 × (
    0.50 × attack_confidence
    + 0.30 × anomaly_score
    + 0.20 × category_impact
)
```

Where:

- `attack_confidence = 1 - P(BENIGN)`
- `anomaly_score` comes from Isolation Forest
- `category_impact` represents the relative impact assigned to the predicted attack category

These weights are a project-specific design decision and are not presented as an industry-standard formula.

The supervised classifier receives the largest weight because it is trained using labelled ground-truth data. The anomaly score provides an additional safety signal, while category impact reflects the fact that different attack categories can have different operational importance.

## 6. Severity Bands

The threat score is mapped to four severity levels:

| Threat Score | Severity |
|---:|---|
| 0–25 | Low |
| 25–50 | Medium |
| 50–75 | High |
| 75–100 | Critical |

## 7. Category Impact

Category-impact values are project-specific judgements and are stored separately in:

```text
models/category_impact.json
```

Higher-impact categories receive larger values because their potential operational consequences are considered more significant.

PortScan receives a lower impact value because it primarily represents reconnaissance rather than an active compromise.

## 8. MITRE ATT&CK Triage

The system includes a lightweight MITRE ATT&CK triage layer.

The predicted attack category is mapped to relevant ATT&CK techniques and supporting investigation notes.

This layer is intended to assist analyst interpretation and prioritization. It is not a replacement for full incident-response investigation.

## 9. Dashboard Simulation

The Streamlit dashboard uses replayed test-set traffic to simulate near-real-time detection.

It is deliberately described as a **simulation/replay system**, not as a live packet-capture system.

This allows the complete ML detection and scoring pipeline to be demonstrated without requiring live network traffic during the hackathon presentation.

## 10. Data Leakage Prevention

Potentially identifying or leakage-prone fields such as Flow ID, source/destination IP addresses and timestamps are removed during preprocessing.

Imputation statistics are fitted on the training data and then applied to the test data. This prevents test-set statistics from influencing the training process.

## 11. Dataset Handling

The original CIC-IDS2017 data is not committed to GitHub because of its size.

Raw data and generated training data remain local, while the repository contains the code, configuration and documentation required to reproduce the pipeline when the dataset is available.