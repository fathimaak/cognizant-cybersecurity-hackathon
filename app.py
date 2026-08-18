"""
app.py — SOC Dashboard, redesigned with tabs and MITRE/triage integration.
"""

import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path
from src.triage import get_global_top_features, build_triage_note, MITRE_MAPPING

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

st.set_page_config(page_title="AI Cybersecurity Threat Detection - SOC Dashboard",
                    layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    .stMetric { background-color: #1a1f2e; border: 1px solid #2d3548; border-radius: 8px; padding: 12px; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

SEVERITY_BANDS = [(0, 25, "Low"), (25, 50, "Medium"), (50, 75, "High"), (75, 101, "Critical")]
SEVERITY_COLOR = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"}


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
    top_features = get_global_top_features(clf, feature_cols, top_n=8)

    return {
        "feature_cols": feature_cols, "clf": clf, "clf_name": best_name, "le": le,
        "iso_model": iso_model, "anchors": anchors, "category_impact": category_impact,
        "test_df": test_df, "model_meta": model_meta, "top_features": top_features,
    }


def anomaly_score_single(iso_model, X, anchors):
    raw = iso_model.decision_function(X)
    p1, p99 = anchors["p1"], anchors["p99"]
    return float(np.clip((p99 - raw) / (p99 - p1), 0.0, 1.0)[0])


def predict_row(clf, le, X, model_name):
    proba = clf.predict_proba(X)[0]
    classes = le.classes_
    prob_dict = {c: float(p) for c, p in zip(classes, proba)}
    pred_idx = int(np.argmax(proba))
    return {
        "attack_category": classes[pred_idx], "confidence": float(proba[pred_idx]),
        "attack_confidence": 1.0 - prob_dict.get("BENIGN", 0.0),
        "probabilities": prob_dict, "model_name": model_name,
    }


def score_event(clf_output, a_score, category_impact):
    impact = category_impact.get(clf_output["attack_category"], 0.5)
    threat_score = float(np.clip(
        100 * (0.50 * clf_output["attack_confidence"] + 0.30 * a_score + 0.20 * impact), 0, 100))
    return {"anomaly_score": a_score, "threat_score": threat_score, "severity": severity_from_score(threat_score)}


def process_event(row, feature_cols, artifacts):
    X = row[feature_cols].to_frame().T.astype(float)
    a_score = anomaly_score_single(artifacts["iso_model"], X, artifacts["anchors"])
    clf_output = predict_row(artifacts["clf"], artifacts["le"], X, artifacts["clf_name"])
    scoring = score_event(clf_output, a_score, artifacts["category_impact"])
    triage_note = build_triage_note(clf_output["attack_category"], scoring["severity"],
                                     clf_output["confidence"], artifacts["top_features"])
    return {"true_label": row.get("Label", "unknown"), **clf_output, **scoring, "triage_note": triage_note}


artifacts = load_artifacts()
feature_cols = artifacts["feature_cols"]

st.title("🛡️ AI-Based Cybersecurity Threat Detection")
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

if "events" not in st.session_state or reset:
    st.session_state.events = []
    st.session_state.total = 0
    st.session_state.threats = 0
    st.session_state.critical = 0

tab1, tab2, tab3, tab4 = st.tabs(["🔴 Live Feed", "📊 Analytics", "🧭 Threat Intel", "📈 Model Evaluation"])

with tab1:
    counters_ph = st.empty()
    feed_ph = st.empty()
    timeline_ph = st.empty()

with tab2:
    st.subheader("Attack Category Breakdown")
    category_ph = st.empty()
    st.subheader("Severity Distribution")
    severity_ph = st.empty()

with tab3:
    st.subheader("MITRE ATT&CK Mapping")
    mitre_rows = [{"Category": k, "Technique ID": v[0], "Technique": v[1]} for k, v in MITRE_MAPPING.items()]
    st.dataframe(pd.DataFrame(mitre_rows), use_container_width=True, hide_index=True)
    st.caption("This team's own mapping for demo purposes — not an official/certified MITRE ATT&CK submission.")
    st.subheader("Live Triage Notes")
    triage_ph = st.empty()

with tab4:
    st.subheader("Model Comparison")
    results = artifacts["model_meta"]["results"]
    comp_rows = [{"Model": name, "Macro F1": round(m["macro_f1"], 4), "Weighted F1": round(m["weighted_f1"], 4),
                  "Rare-class Recall": round(m["rare_class_recall"], 4), "Train Time (s)": round(m["train_seconds"], 1)}
                 for name, m in results.items()]
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.subheader("Confusion Matrix")
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)
    else:
        st.info("Run `python src/evaluate_models.py` to generate this.")

    st.subheader("Per-Class Metrics")
    pc_path = RESULTS_DIR / "per_class_recall.csv"
    if pc_path.exists():
        st.dataframe(pd.read_csv(pc_path, index_col=0), use_container_width=True)
    else:
        st.info("Run `python src/evaluate_models.py` to generate this.")


def render(events, total, threats, critical):
    with counters_ph.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Events", total)
        c2.metric("Threats Detected", threats)
        c3.metric("Critical Alerts", critical)
        c4.metric("Detection Rate", f"{(threats / total * 100 if total else 0):.1f}%")

    if events:
        feed_df = pd.DataFrame(events[-20:][::-1])
        feed_ph.dataframe(
            feed_df[["true_label", "attack_category", "confidence", "anomaly_score", "threat_score", "severity"]],
            use_container_width=True, height=380)

        ev_df = pd.DataFrame(events)
        timeline = ev_df.reset_index().rename(columns={"index": "event_order"})
        fig1 = px.scatter(timeline, x="event_order", y="threat_score", color="severity",
                           color_discrete_map=SEVERITY_COLOR)
        timeline_ph.plotly_chart(fig1, use_container_width=True)

        cat_counts = ev_df["attack_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        category_ph.plotly_chart(px.bar(cat_counts, x="category", y="count"), use_container_width=True)

        sev_counts = ev_df["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        severity_ph.plotly_chart(
            px.pie(sev_counts, names="severity", values="count", color="severity", color_discrete_map=SEVERITY_COLOR),
            use_container_width=True)

        recent_critical = [e for e in events[-15:] if e["severity"] in ("High", "Critical")][::-1]
        if recent_critical:
            triage_ph.markdown("\n\n".join(
                f"**{e['attack_category']}** — {e['triage_note']}" for e in recent_critical[:8]))
        else:
            triage_ph.info("No High/Critical events yet in this run.")
    else:
        feed_ph.info("No events yet - click Start Simulation.")


render(st.session_state.events, st.session_state.total, st.session_state.threats, st.session_state.critical)

if start:
    sample = artifacts["test_df"].sample(n=n_events, random_state=int(seed)).reset_index(drop=True)
    for i in range(len(sample)):
        result = process_event(sample.iloc[i], feature_cols, artifacts)
        st.session_state.events.append(result)
        st.session_state.total += 1
        if result["severity"] in ("High", "Critical"):
            st.session_state.threats += 1
        if result["severity"] == "Critical":
            st.session_state.critical += 1
        render(st.session_state.events, st.session_state.total, st.session_state.threats, st.session_state.critical)
        time.sleep(delay)