"""
src/data_pipeline.py
Pair A pipeline (locked v3 + Patch 1, 2, 4):
clean -> remove leakage columns -> dedupe -> normalize labels ->
class-aware sampling -> stratified split -> fix Inf/NaN (train-fit, test-transform) -> save

Design notes:
- Class-aware sampling happens BEFORE the split - this is the locked design from the
  master plan, not an oversight. It means the test set reflects the same reduced
  BENIGN ratio as train, not "natural" traffic proportions - documented here on purpose.
- Inf/NaN imputation happens AFTER the split, fit on train only, then applied to test
  with the SAME median values. Computing medians on the combined train+test data would
  leak test-set statistics into how missing training values get filled - this avoids that.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

LEAKAGE_COLUMNS = [
    "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
    "Timestamp", "SimillarHTTP", "Unnamed: 0",
]

# Canonical label set for CIC-IDS2017. Keys are lowercased+stripped.
CANONICAL_LABELS = {
    "benign": "BENIGN",
    "ddos": "DDoS",
    "portscan": "PortScan",
    "bot": "Bot",
    "infiltration": "Infiltration",
    "heartbleed": "Heartbleed",
    "dos hulk": "DoS Hulk",
    "dos goldeneye": "DoS GoldenEye",
    "dos slowloris": "DoS Slowloris",
    "dos slowhttptest": "DoS Slowhttptest",
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
    "web attack - brute force": "Web Attack - Brute Force",
    "web attack - xss": "Web Attack - XSS",
    "web attack - sql injection": "Web Attack - SQL Injection",
}


def load_and_standardize(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp1252", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_all_csvs() -> pd.DataFrame:
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {RAW_DIR}. Did you extract the dataset there?")

    frames = []
    reference_columns = None
    for f in csv_files:
        df = load_and_standardize(f)
        if reference_columns is None:
            reference_columns = set(df.columns)
        else:
            mismatch = set(df.columns) ^ reference_columns
            if mismatch:
                raise ValueError(f"Column mismatch in {f.name}: {mismatch}")
        print(f"Loaded {f.name}: {df.shape[0]:,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal combined: {combined.shape[0]:,} rows, {combined.shape[1]} columns")
    return combined


def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if cols_to_drop:
        print(f"Dropping leakage columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df):,} duplicate rows ({before:,} -> {len(df):,})")
    return df


def normalize_labels(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    if label_col not in df.columns:
        raise KeyError(f"Expected label column '{label_col}' not found. Columns: {list(df.columns)}")

    df[label_col] = df[label_col].astype(str).str.strip()
    # verified against the real raw data: the source CSVs already contain a
    # UTF-8-replacement-character sequence in place of the original en-dash
    df[label_col] = df[label_col].str.replace("ï¿½", "-", regex=False)
    df[label_col] = df[label_col].str.replace("\u2013", "-", regex=False)  # real en dash, just in case
    df[label_col] = df[label_col].str.replace("\u2014", "-", regex=False)  # em dash, just in case
    df[label_col] = df[label_col].str.replace(r"\s+", " ", regex=True).str.strip()

    lowered = df[label_col].str.lower()
    mapped = lowered.map(CANONICAL_LABELS)
    unmapped = df[label_col][mapped.isna()].unique().tolist()
    if unmapped:
        print(f"NOTE: labels not found in canonical map, kept as-is: {unmapped}")
    df[label_col] = mapped.fillna(df[label_col])

    print("\nClass distribution after normalization:")
    print(df[label_col].value_counts())
    return df


def class_aware_sample(df: pd.DataFrame, label_col: str = "Label",
                        benign_label: str = "BENIGN", benign_ratio: float = 3.0) -> pd.DataFrame:
    attack_df = df[df[label_col] != benign_label]
    benign_df = df[df[label_col] == benign_label]

    n_attacks = len(attack_df)
    n_benign_target = min(len(benign_df), int(n_attacks * benign_ratio))

    benign_sampled = benign_df.sample(n=n_benign_target, random_state=RANDOM_SEED)
    result = pd.concat([attack_df, benign_sampled], ignore_index=True)
    result = result.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\nClass-aware sampling: kept {n_attacks:,} attack rows, "
          f"sampled {n_benign_target:,} of {len(benign_df):,} BENIGN rows "
          f"(ratio {benign_ratio}:1)")
    print(result[label_col].value_counts())
    return result


def stratified_split(df: pd.DataFrame, label_col: str = "Label", test_size: float = 0.2):
    counts = df[label_col].value_counts()
    rare_classes = counts[counts < 2].index.tolist()

    if rare_classes:
        print(f"WARNING: classes with <2 rows go entirely to train (can't stratify): {rare_classes}")
        rare_df = df[df[label_col].isin(rare_classes)]
        stratifiable_df = df[~df[label_col].isin(rare_classes)]
    else:
        rare_df = df.iloc[0:0]
        stratifiable_df = df

    train_df, test_df = train_test_split(
        stratifiable_df, test_size=test_size, random_state=RANDOM_SEED,
        stratify=stratifiable_df[label_col]
    )
    train_df = pd.concat([train_df, rare_df], ignore_index=True)

    print(f"\nTrain: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    return train_df, test_df


def fix_inf_nan_fit(df: pd.DataFrame, numeric_cols):
    """Fit imputation medians on TRAIN ONLY, return the filled df + medians to reuse on test."""
    df = df.copy()
    n_inf = np.isinf(df[numeric_cols]).sum().sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    medians = df[numeric_cols].median()
    n_nan = df[numeric_cols].isna().sum().sum()
    df[numeric_cols] = df[numeric_cols].fillna(medians)
    print(f"[train] Fixed {int(n_inf):,} Inf values, imputed {int(n_nan):,} NaN values (train-only median)")
    return df, medians


def fix_inf_nan_transform(df: pd.DataFrame, numeric_cols, medians):
    """Apply TRAIN's medians to test - never compute test's own medians (avoids leakage)."""
    df = df.copy()
    n_inf = np.isinf(df[numeric_cols]).sum().sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    n_nan = df[numeric_cols].isna().sum().sum()
    df[numeric_cols] = df[numeric_cols].fillna(medians)
    print(f"[test]  Fixed {int(n_inf):,} Inf values, imputed {int(n_nan):,} NaN values (using train's medians)")
    return df


def main():
    df = load_all_csvs()
    df = remove_leakage_columns(df)
    df = dedupe(df)
    df = normalize_labels(df)
    df = class_aware_sample(df)

    train_df, test_df = stratified_split(df)

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    train_df, medians = fix_inf_nan_fit(train_df, numeric_cols)
    test_df = fix_inf_nan_transform(test_df, numeric_cols, medians)

    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    metadata = {
        "random_seed": RANDOM_SEED,
        "feature_columns": [c for c in train_df.columns if c != "Label"],
        "label_column": "Label",
        "classes": sorted(train_df["Label"].unique().tolist()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_class_counts": train_df["Label"].value_counts().to_dict(),
        "test_class_counts": test_df["Label"].value_counts().to_dict(),
        "imputation_medians": medians.to_dict(),
        "design_notes": {
            "class_aware_sampling_before_split": "Intentional, per locked master plan - "
                "test set reflects the same reduced BENIGN ratio as train, not natural traffic proportions.",
            "imputation_fit_on_train_only": "Medians computed from train only, applied identically to "
                "test - avoids leaking test-set statistics into training feature values. Also reused "
                "downstream (simulator/integration layer) to preprocess new incoming events consistently.",
        },
    }
    with open(PROCESSED_DIR / "data_contract.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved:")
    print(f"  {PROCESSED_DIR / 'train.parquet'}")
    print(f"  {PROCESSED_DIR / 'test.parquet'}")
    print(f"  {PROCESSED_DIR / 'data_contract.json'}")


if __name__ == "__main__":
    main()