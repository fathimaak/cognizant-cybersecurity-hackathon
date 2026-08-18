# AI-Based Cybersecurity Threat Detection — CIC-IDS2017

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

Run the following in order:

```bash
python src\data_pipeline.py
python src\train_models.py
python src\anomaly_scoring.py
python src\evaluate_models.py
streamlit run app.py
```

The data pipeline requires the CIC-IDS2017 CSV files in `data/raw/`.

## Dataset

CIC-IDS2017 is not committed to GitHub because of its size (~2.5 GB). Download the 8 official CSVs and place them in `data/raw/`.

See `docs/dataset.md` for the full class distribution and preprocessing decisions.

## Architecture

Data pipeline → Supervised classifiers (Logistic Regression / Random Forest / XGBoost) → Isolation Forest (anomaly detection) → Threat scoring → MITRE ATT&CK triage layer → Streamlit dashboard.

The dashboard performs a near-real-time replay simulation; it is **not live packet capture**.

## Key Results

- Best model: **XGBoost** — macro F1 0.914, weighted F1 0.998, rare-class recall 0.921
- Isolation Forest: ROC-AUC 0.799, PR-AUC 0.625
- Weakest class: **Web Attack - XSS** (F1 0.41)
- Flow-level features do not contain payload content, which limits discrimination between some web attack classes.

## Documentation

- `docs/decisions.md` — key design decisions and why
- `docs/interfaces.md` — frozen function contracts
- `docs/dataset.md` — class distribution and preprocessing steps