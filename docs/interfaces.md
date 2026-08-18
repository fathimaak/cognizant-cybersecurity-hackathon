# Interfaces

This document defines the interfaces between the main components of the detection pipeline.

## 1. Classifier Prediction

The supervised classifier receives a one-row DataFrame containing the required feature columns.

```python
X = rows.iloc[[0]][feature_cols].astype(float)
```

### Important implementation rule

When extracting a single row for prediction, keep it as a DataFrame.

Use:

```python
.iloc[[0]]
```

Do **not** use:

```python
.iloc[0]
```

followed by a transpose.

Using `.iloc[0]` converts the row into a pandas Series and can cause mixed numeric columns to become `object` dtype. This can cause compatibility problems with models such as XGBoost.

The prediction input must therefore remain a numeric DataFrame.

## 2. Classifier Output

The classifier prediction is converted into a dictionary containing:

```python
{
    "attack_category": predicted_class,
    "confidence": predicted_probability,
    "attack_confidence": 1.0 - benign_probability,
    "probabilities": probability_by_class,
    "model_name": model_name
}
```

Where:

- `attack_category` is the predicted attack category.
- `confidence` is the probability of the predicted category.
- `attack_confidence` is `1 - P(BENIGN)`.
- `probabilities` contains the probability for every known class.
- `model_name` identifies the classifier used.

## 3. Anomaly Score

The anomaly detector receives the same feature DataFrame used by the classifier.

```python
anomaly_score(iso_model, X, anchors)
```

The returned anomaly score is normalized to the range:

```text
0 ≤ anomaly_score ≤ 1
```

Higher values indicate greater deviation from the learned BENIGN traffic distribution.

## 4. Threat Scoring

The classifier output and anomaly score are passed to:

```python
score_event(clf_output, anomaly_score)
```

The function returns a threat assessment containing the calculated threat score and severity.

The current threat-score formula is:

```text
100 × (
    0.50 × attack_confidence
    + 0.30 × anomaly_score
    + 0.20 × category_impact
)
```

## 5. Severity

The threat score is converted into one of four severity levels:

```text
0–25     → Low
25–50    → Medium
50–75    → High
75–100   → Critical
```

## 6. MITRE ATT&CK Triage

The predicted attack category is passed to the triage layer.

The triage layer provides:

- Relevant MITRE ATT&CK technique information
- Investigation guidance
- Analyst-oriented notes
- Severity/context information

The triage layer is an interpretation and prioritization component; it does not perform the underlying attack classification.

## 7. Dashboard Data Flow

The dashboard follows this general flow:

```text
Test-set row
    ↓
Feature extraction
    ↓
Supervised classifier
    ↓
Classifier output
    ↓
Isolation Forest
    ↓
Anomaly score
    ↓
Threat scoring
    ↓
Severity
    ↓
MITRE ATT&CK triage
    ↓
Streamlit dashboard
```

## 8. Prediction Input Contract

Any component sending a single event to the prediction pipeline must provide:

1. All required model feature columns
2. Numeric values for those features
3. The same feature ordering used during training
4. A pandas DataFrame rather than a pandas Series

This contract is important for both the dashboard simulator and future integrations.

## 9. Reproducibility

The trained models and normalization configuration are loaded from the `models/` directory.

The following files are used by the inference pipeline:

```text
models/
├── isolation_forest_model.joblib
├── logisticregression_model.joblib
├── randomforest_model.joblib
├── xgboost_model.joblib
├── label_encoder.joblib
├── model_metadata.json
├── anomaly_normalization.json
└── category_impact.json
```

The large model files are intentionally excluded from Git tracking where required by `.gitignore`.