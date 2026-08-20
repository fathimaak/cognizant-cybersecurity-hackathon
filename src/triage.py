"""
src/triage.py
Explainability + MITRE ATT&CK triage layer.
Template-based over real feature-importance values and category metadata -
no free-text generation, just structured formatting. This is this team's own
mapping, not an official/verified MITRE certification.
"""

import numpy as np
import shap

MITRE_MAPPING = {
    "DDoS": ("T1498", "Network Denial of Service"),
    "DoS Hulk": ("T1499", "Endpoint Denial of Service"),
    "DoS GoldenEye": ("T1499", "Endpoint Denial of Service"),
    "DoS Slowloris": ("T1499", "Endpoint Denial of Service"),
    "DoS Slowhttptest": ("T1499", "Endpoint Denial of Service"),
    "PortScan": ("T1595", "Active Scanning"),
    "Bot": ("T1071", "Application Layer Protocol (C2)"),
    "Infiltration": ("T1190", "Exploit Public-Facing Application"),
    "Heartbleed": ("T1190", "Exploit Public-Facing Application (CVE-2014-0160)"),
    "FTP-Patator": ("T1110", "Brute Force"),
    "SSH-Patator": ("T1110", "Brute Force"),
    "Web Attack - Brute Force": ("T1110", "Brute Force"),
    "Web Attack - XSS": ("T1190", "Exploit Public-Facing Application"),
    "Web Attack - SQL Injection": ("T1190", "Exploit Public-Facing Application"),
}

RECOMMENDED_ACTIONS = {
    "DDoS": "Rate-limit or blackhole the source, engage upstream DDoS mitigation.",
    "DoS Hulk": "Rate-limit source, inspect for HTTP flood patterns.",
    "DoS GoldenEye": "Rate-limit source, inspect for HTTP flood patterns.",
    "DoS Slowloris": "Enable connection timeout hardening, rate-limit source.",
    "DoS Slowhttptest": "Enable connection timeout hardening, rate-limit source.",
    "PortScan": "Monitor source IP, verify firewall rules block unused ports.",
    "Bot": "Isolate host, check for command-and-control (C2) traffic.",
    "Infiltration": "Isolate host immediately, begin forensic review.",
    "Heartbleed": "Patch OpenSSL immediately, rotate all affected certificates/keys.",
    "FTP-Patator": "Lock account after threshold, enforce stronger auth policy.",
    "SSH-Patator": "Lock account after threshold, enforce key-based auth.",
    "Web Attack - Brute Force": "Lock account after threshold, review login rate-limiting.",
    "Web Attack - XSS": "Review WAF rules, sanitize/escape affected input fields.",
    "Web Attack - SQL Injection": "Review WAF rules, patch parameterized queries.",
}


def get_global_top_features(model, feature_cols, top_n=5):
    """Global feature importance from the trained model - not per-instance SHAP (too heavy for a live demo)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("clf", None), "coef_"):
        importances = np.abs(model.named_steps["clf"].coef_).mean(axis=0)
    else:
        importances = np.zeros(len(feature_cols))
    idx = np.argsort(importances)[::-1][:top_n]
    return [(feature_cols[i], float(importances[i])) for i in idx]

_explainer_cache = {}

def get_shap_top_features(model, model_name, X_row, feature_cols, top_n=3):
    """
    Per-instance explainability using SHAP - explains THIS specific prediction,
    not just global model behavior. Falls back to empty list if the model
    type isn't tree-based (SHAP TreeExplainer only supports RF/XGBoost cleanly).
    """
    if model_name not in ("XGBoost", "RandomForest"):
        return []
    try:
        if model_name not in _explainer_cache:
            _explainer_cache[model_name] = shap.TreeExplainer(model)
        explainer = _explainer_cache[model_name]
        shap_values = explainer.shap_values(X_row)

        if isinstance(shap_values, list):
            pred_class_idx = int(np.argmax(model.predict_proba(X_row)[0]))
            values = shap_values[pred_class_idx][0]
        else:
            pred_class_idx = int(np.argmax(model.predict_proba(X_row)[0]))
            values = shap_values[0, :, pred_class_idx]

        idx = np.argsort(np.abs(values))[::-1][:top_n]
        return [(feature_cols[i], float(values[i])) for i in idx]
    except Exception:
        return []  # never let explainability break the live dashboard


def build_triage_note(category: str, severity: str, confidence: float, top_features: list, is_shap: bool = False) -> str:
    if category == "BENIGN":
        return "No action needed — traffic matches normal baseline patterns."

    technique = MITRE_MAPPING.get(category)
    tech_str = f"{technique[0]} ({technique[1]})" if technique else "no direct MITRE mapping available"
    if top_features:
        if is_shap:
            feat_str = ", ".join(f"{f} ({'↑' if v > 0 else '↓'} contribution)" for f, v in top_features[:3])
        else:
            feat_str = ", ".join(f for f, _ in top_features[:3])
    else:
        feat_str = "overall flow statistics"
    action = RECOMMENDED_ACTIONS.get(category, "Investigate source traffic and correlate with other alerts.")

    return (f"Flagged as {category} ({confidence*100:.0f}% classifier confidence, severity {severity}). "
            f"Maps to MITRE ATT&CK {tech_str}. Driven primarily by: {feat_str}. "
            f"Recommended action: {action}")