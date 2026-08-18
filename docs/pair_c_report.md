# Pair C — Anomaly Detection & Threat Scoring

## Isolation Forest

Isolation Forest was trained on 1,022,107 BENIGN-only rows from `train.parquet`.

It learns the characteristics of normal traffic and detects deviations from
that normal behaviour. This complements the supervised classifier because
anomaly detection does not require labelled examples of every possible attack.

Isolation Forest is unsupervised, so accuracy is not used as its primary metric.

Evaluation results:

- Known-BENIGN mean anomaly score: 0.161
- Known-BENIGN median anomaly score: 0.076
- Known-attack mean anomaly score: 0.535
- Known-attack median anomaly score: 0.594
- ROC-AUC (BENIGN vs attack): 0.799
- PR-AUC (BENIGN vs attack): 0.625
- Detection rate at 0.5 threshold: 58.9%
- False-alarm rate at 0.5 threshold: 8.6%

The results show meaningful separation between BENIGN and attack traffic.
The fixed 0.5 threshold does not detect every attack, which is why anomaly
detection is combined with the supervised classifier rather than being used
alone.

## Anomaly Score Normalization

The Isolation Forest `decision_function()` produces raw scores where higher
values indicate more normal traffic.

The raw score is therefore inverted and normalized using the 1st and 99th
percentiles of BENIGN training scores:

    normalized = (p99 - raw_score) / (p99 - p1)

The result is clipped to the range [0, 1].

The normalization anchors are calculated from BENIGN training data and then
locked so that the scale remains consistent across later predictions.

Percentile-based anchors are used instead of minimum and maximum values so
that extreme outliers do not distort the normalization.

## Threat Scoring

The final threat score combines three signals:

    threat_score = 100 × (
        0.50 × attack_confidence
        + 0.30 × anomaly_score
        + 0.20 × category_impact
    )

Where:

- `attack_confidence = 1 − P(BENIGN)`
- `anomaly_score` comes from Isolation Forest
- `category_impact` represents the designed severity weight for the predicted
  attack category

The threat-scoring formula and weights are this team's own design. They are
not an industry-standard formula.

The supervised classifier receives the largest weight because it is trained
using labelled attack data. The anomaly detector provides an additional
signal for unusual traffic, while category impact accounts for differences
in the potential severity of attack categories.

## Category Impact

The category-impact values are team-designed weights:

- DDoS: 1.0
- Infiltration: 1.0
- Heartbleed: 1.0
- Web Attack - SQL Injection: 0.9
- DoS Hulk: 0.85
- DoS GoldenEye: 0.85
- Bot: 0.8
- Web Attack - XSS: 0.75
- DoS Slowloris: 0.7
- DoS Slowhttptest: 0.7
- Web Attack - Brute Force: 0.7
- FTP-Patator: 0.6
- SSH-Patator: 0.6
- PortScan: 0.4
- BENIGN: 0.0

These values are design choices made by the team and are not claimed to be
industry standards.

## Severity Bands

| Threat Score | Severity |
|---|---|
| 0–25 | Low |
| 25–50 | Medium |
| 50–75 | High |
| 75–100 | Critical |

## Full-Chain Sanity Check

| True Label | Predicted | Anomaly Score | Threat Score | Severity |
|---|---|---:|---:|---|
| BENIGN | BENIGN | 0.150 | 4.5 | Low |
| DDoS | DDoS | 0.607 | 88.2 | Critical |
| PortScan | PortScan | 0.063 | 59.9 | High |
| Web Attack - SQL Injection | Web Attack - SQL Injection | 0.245 | 75.3 | Critical |

## Why Combine Classification and Anomaly Detection?

The two approaches provide complementary signals.

For example, PortScan produced a relatively low anomaly score of 0.063,
meaning it looked relatively close to normal traffic to the Isolation Forest.
However, the supervised classifier identified it as PortScan with high
confidence, resulting in a High overall threat score.

This demonstrates why the final system does not rely on anomaly detection
alone.

## Q&A Preparation

### Why train Isolation Forest only on BENIGN traffic?

The objective is to learn the characteristics of normal traffic. Deviations
from that learned normal behaviour can then be detected without requiring
labelled examples of every possible attack.

### Why use Isolation Forest?

Isolation Forest is suitable for this project because it is an unsupervised
method, does not require attack labels for training, is computationally
practical for a large dataset, and provides a relatively explainable anomaly
signal.

### Why not use anomaly detection alone?

Anomaly detection can identify unusual traffic but may miss attacks whose
flow-level characteristics resemble normal traffic. The supervised classifier
provides a complementary labelled classification signal.

### How were the threat-score weights selected?

The weights are a team-designed decision. The classifier receives 50% because
it is trained on labelled ground truth, anomaly detection receives 30% as an
additional signal for unusual behaviour, and category impact receives 20% to
reflect differences in attack severity.

The formula is explicitly a project design choice, not an industry standard.