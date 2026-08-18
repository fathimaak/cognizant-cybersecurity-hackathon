"""
src/triage.py
Explainability + MITRE ATT&CK triage layer.
Template-based over real feature-importance values and category metadata -
no free-text generation, just structured formatting. This is this team's own
mapping, not an official/verified MITRE certification.
"""

import numpy as np

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


def build_triage_note(category: str, severity: str, confidence: float, top_features: list) -> str:
    if category == "BENIGN":
        return "No action needed — traffic matches normal baseline patterns."

    technique = MITRE_MAPPING.get(category)
    tech_str = f"{technique[0]} ({technique[1]})" if technique else "no direct MITRE mapping available"
    feat_str = ", ".join(f for f, _ in top_features[:3]) if top_features else "overall flow statistics"
    action = RECOMMENDED_ACTIONS.get(category, "Investigate source traffic and correlate with other alerts.")

    return (f"Flagged as {category} ({confidence*100:.0f}% classifier confidence, severity {severity}). "
            f"Maps to MITRE ATT&CK {tech_str}. Driven primarily by model signal from: {feat_str}. "
            f"Recommended action: {action}")