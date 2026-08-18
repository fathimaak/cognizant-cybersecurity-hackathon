# Dataset — CIC-IDS2017

Source: 8 CSVs, read with `cp1252` encoding (source files use this encoding).

## Cleaning pipeline

1. Standardize column names per file before concatenating (strip whitespace)
2. Remove leakage columns (Flow ID, Source/Destination IP, Timestamp)
3. Deduplicate: 2,830,743 → 2,522,362 rows (308,381 duplicates removed)
4. Normalize labels: fixed a UTF-8 replacement-character artifact in the original Web Attack labels (e.g. "Web Attack ï¿½ Brute Force" → "Web Attack - Brute Force")
5. Class-aware sampling: kept all 425,878 attack rows, downsampled BENIGN to a 3:1 ratio (1,277,634 of 2,096,484 BENIGN rows) — before the split, per locked design
6. Stratified 80/20 split: 1,362,809 train / 340,703 test
7. Inf/NaN imputation: fit on train only, applied to test with the same medians (avoids leaking test statistics into training)

## Final class distribution (train + test combined)

| Class | Count |
|---|---:|
| BENIGN | 1,277,634 |
| DoS Hulk | 172,849 |
| DDoS | 128,016 |
| PortScan | 90,819 |
| DoS GoldenEye | 10,286 |
| FTP-Patator | 5,933 |
| DoS Slowloris | 5,385 |
| DoS Slowhttptest | 5,228 |
| SSH-Patator | 3,219 |
| Bot | 1,953 |
| Web Attack - Brute Force | 1,470 |
| Web Attack - XSS | 652 |
| Infiltration | 36 |
| Web Attack - SQL Injection | 21 |
| Heartbleed | 11 |

**Caveat:** Heartbleed (11), SQL Injection (21), and Infiltration (36) have very few samples — any recall/precision numbers on these classes should be treated as indicative, not statistically robust.