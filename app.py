"""
app.py
Pair D - Dashboard, Near-Real-Time Simulation (combined for now, split into
src/simulator.py + app.py later per the plan).

Loads: Niya's best classifier + label encoder, Fathima's Isolation Forest +
normalization anchors + category impact + threat scoring, Annet's data contract + test set.
Simulates: replays rows from test.parquet with a short delay - NOT live packet capture.
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

st.set_page_config(page_title="AI Cybersecurity Threat Detection - SOC Dashboard", layout="wide")

SEVERITY_BANDS = [(0, 25, "Low"), (25, 50, "Medium"), (50, 75, "High"), (75, 101, "Critical")]


def severity_from_score(score):
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= score < hi:
            return label
    return "Critical"


@st.cache_resource
def load_artifacts():
    with open(PROCESSED_DIR / "data_contract.json") as f:
        contract = json.load(f)
    feature_cols = contract["feature_columns"]

    with open(MODELS_DIR / "model_metadata.json") as f:
        model_meta = json.load(f)
    best_name = model_meta["best_model"]
    clf = joblib.load(MODELS_DIR / f"{best_name.lower()}_model.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")

    iso_model = joblib.load(MODELS_DIR / "isolation_forest_model.joblib")
    with open(MODELS_DIR / "anomaly_normalization.json") as f:
        anchors = json.load(f)
    with open(MODELS_DIR / "category_impact.json") as f:
        category_impact = json.load(f)

    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    return {
        "feature_cols": feature_cols, "clf": clf, "clf_name": best_name, "le": le,
        "iso_model": iso_model, "anchors": anchors, "category_impact": category_impact,
        "test_df": test_df, "model_meta": model_meta,
    }


def validate_input_row(row_df, feature_cols):
    """Lightweight input check - catches a malformed event before it reaches the models."""
    missing = [c for c in feature_cols if c not in row_df.columns]
    if missing:
        raise ValueError(f"Missing expected features: {missing}")
    if row_df[feature_cols].isna().any().any():
        raise ValueError("Row contains NaN in feature columns.")
    return True


def anomaly_score_single(iso_model, X, anchors):
    raw = iso_model.decision_function(X)
    p1, p99 = anchors["p1"], anchors["p99"]
    scaled = (p99 - raw) / (p99 - p1)
    return float(np.clip(scaled, 0.0, 1.0)[0])


def predict_row(clf, le, X, model_name):
    proba = clf.predict_proba(X)[0]
    classes = le.classes_
    prob_dict = {c: float(p) for c, p in zip(classes, proba)}
    pred_idx = int(np.argmax(proba))
    return {
        "attack_category": classes[pred_idx],
        "confidence": float(proba[pred_idx]),
        "attack_confidence": 1.0 - prob_dict.get("BENIGN", 0.0),
        "probabilities": prob_dict,
        "model_name": model_name,
    }


def score_event(clf_output, a_score, category_impact):
    attack_confidence = clf_output["attack_confidence"]
    impact = category_impact.get(clf_output["attack_category"], 0.5)
    threat_score = float(np.clip(
        100 * (0.50 * attack_confidence + 0.30 * a_score + 0.20 * impact), 0, 100
    ))
    return {"anomaly_score": a_score, "threat_score": threat_score,
            "severity": severity_from_score(threat_score)}


def process_event(row, feature_cols, artifacts):
    X = row[feature_cols].to_frame().T.astype(float)
    validate_input_row(X, feature_cols)
    a_score = anomaly_score_single(artifacts["iso_model"], X, artifacts["anchors"])
    clf_output = predict_row(artifacts["clf"], artifacts["le"], X, artifacts["clf_name"])
    scoring = score_event(clf_output, a_score, artifacts["category_impact"])
    return {"true_label": row.get("Label", "unknown"), **clf_output, **scoring}


artifacts = load_artifacts()
feature_cols = artifacts["feature_cols"]

st.title("🛡️ AI-Based Cybersecurity Threat Detection — SOC Dashboard")
st.caption("Use Case 16 · CIC-IDS2017 · Near-real-time replay simulation (not live packet capture)")

with st.sidebar:
    st.header("Simulation Controls")
    n_events = st.slider("Number of events to replay", 10, 200, 50, step=10)
    delay = st.slider("Delay between events (seconds)", 0.05, 2.0, 0.3, step=0.05)
    seed = st.number_input("Sample seed", value=42, step=1)
    start = st.button("▶ Start Simulation", type="primary", use_container_width=True)
    reset = st.button("Reset", use_container_width=True)
    st.divider()
    st.subheader("Model in use")
    st.write(f"**Classifier:** {artifacts['clf_name']}")
    st.write(f"**Selection criterion:** {artifacts['model_meta']['model_selection_criterion']}")

if "events" not in st.session_state or reset:
    st.session_state.events = []
    st.session_state.total = 0
    st.session_state.threats = 0
    st.session_state.critical = 0

counters_ph = st.empty()
feed_header_ph = st.empty()
feed_ph = st.empty()
charts_ph = st.empty()


def render(events, total, threats, critical):
    with counters_ph.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Events", total)
        c2.metric("Threats Detected", threats)
        c3.metric("Critical Alerts", critical)
        c4.metric("Detection Rate", f"{(threats / total * 100 if total else 0):.1f}%")

    feed_header_ph.subheader("Live Threat Feed")
    if events:
        feed_df = pd.DataFrame(events[-20:][::-1])
        feed_ph.dataframe(
            feed_df[["true_label", "attack_category", "confidence", "anomaly_score", "threat_score", "severity"]],
            use_container_width=True, height=400,
        )
    else:
        feed_ph.info("No events yet - click Start Simulation.")

    with charts_ph.container():
        col1, col2 = st.columns(2)
        if events:
            ev_df = pd.DataFrame(events)
            with col1:
                st.subheader("Incident Timeline")
                timeline = ev_df.reset_index().rename(columns={"index": "event_order"})
                fig1 = px.scatter(timeline, x="event_order", y="threat_score", color="severity",
                                   color_discrete_map={"Low": "green", "Medium": "gold",
                                                        "High": "orange", "Critical": "red"})
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                st.subheader("Attack Category Breakdown")
                cat_counts = ev_df["attack_category"].value_counts().reset_index()
                cat_counts.columns = ["category", "count"]
                fig2 = px.bar(cat_counts, x="category", y="count")
                st.plotly_chart(fig2, use_container_width=True)


render(st.session_state.events, st.session_state.total, st.session_state.threats, st.session_state.critical)

st.divider()
st.subheader("Model Comparison")
results = artifacts["model_meta"]["results"]
comp_rows = [{
    "Model": name, "Macro F1": round(m["macro_f1"], 4), "Weighted F1": round(m["weighted_f1"], 4),
    "Rare-class Recall": round(m["rare_class_recall"], 4), "Train Time (s)": round(m["train_seconds"], 1),
} for name, m in results.items()]
st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

if start:
    sample = artifacts["test_df"].sample(n=n_events, random_state=int(seed)).reset_index(drop=True)
    for i in range(len(sample)):
        row = sample.iloc[i]
        try:
            result = process_event(row, feature_cols, artifacts)
        except Exception as e:
            st.error(f"Event {i} failed validation: {e}")
            continue

        st.session_state.events.append(result)
        st.session_state.total += 1
        if result["severity"] in ("High", "Critical"):
            st.session_state.threats += 1
        if result["severity"] == "Critical":
            st.session_state.critical += 1

        render(st.session_state.events, st.session_state.total, st.session_state.threats, st.session_state.critical)
        time.sleep(delay)