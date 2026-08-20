"""
Class-Conditional Synthetic Network Flow Generator

Generates synthetic CIC-IDS2017-style network-flow records without
shipping the original dataset to deployment.

The generator:
1. Learns representative rows for each traffic class.
2. Learns per-feature variation within each class.
3. Creates synthetic flows by perturbing representative rows.
4. Preserves binary/network-flag features.
5. Clips generated values to observed class ranges.
6. Applies lightweight network-flow consistency constraints.
"""

import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path


PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

LABEL_COL = "Label"
RANDOM_SEED = 42

# Number of representative rows retained for each class.
TEMPLATES_PER_CLASS = 100


# ---------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------

# Network ports are discrete integer values.
INTEGER_FEATURES = {
    "Destination Port",

    "Total Fwd Packets",
    "Total Backward Packets",

    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Header Length.1",

    "Subflow Fwd Packets",
    "Subflow Bwd Packets",

    "act_data_pkt_fwd",

    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",

    "min_seg_size_forward",
}


# Packet-count / bulk-count features should be integer-valued.
COUNT_FEATURES = {
    "Total Fwd Packets",
    "Total Backward Packets",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "act_data_pkt_fwd",

    "Fwd Avg Packets/Bulk",
    "Bwd Avg Packets/Bulk",
}


# Binary network flags.
FLAG_FEATURES = {
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",

    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
}


# Packet-length relationships where:
#
# min <= mean <= max
#
# These are repaired after generation.
PACKET_LENGTH_GROUPS = [
    (
        "Fwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Fwd Packet Length Max",
    ),
    (
        "Bwd Packet Length Min",
        "Bwd Packet Length Mean",
        "Bwd Packet Length Max",
    ),
    (
        "Min Packet Length",
        "Packet Length Mean",
        "Max Packet Length",
    ),
]


# Standard deviation / variance cannot be negative.
NON_NEGATIVE_STAT_FEATURES = {
    "Fwd Packet Length Std",
    "Bwd Packet Length Std",
    "Packet Length Std",
    "Packet Length Variance",

    "Flow IAT Std",

    "Fwd IAT Std",
    "Bwd IAT Std",

    "Active Std",
    "Idle Std",
}


class SyntheticFlowGenerator:

    def __init__(self, model_data):

        self.feature_cols = model_data["feature_cols"]
        self.classes = model_data["classes"]
        self.templates = model_data["templates"]
        self.stats = model_data["stats"]
        self.binary_features = model_data["binary_features"]

        self.rng = np.random.default_rng(RANDOM_SEED)

    # -----------------------------------------------------------------
    # Generate one synthetic flow
    # -----------------------------------------------------------------

    def generate_one(self, attack_type=None):

        """
        Generate one synthetic network-flow record.

        Returns a pandas Series containing the 78 model features
        plus the intended synthetic label.
        """

        if attack_type is None:
            attack_type = self.rng.choice(self.classes)

        if attack_type not in self.templates:
            raise ValueError(
                f"Unknown class '{attack_type}'. "
                f"Available classes: {self.classes}"
            )

        templates = self.templates[attack_type]

        # Select representative real flow.
        template_idx = self.rng.integers(
            0,
            len(templates)
        )

        base = np.asarray(
            templates[template_idx],
            dtype=np.float64
        )

        feature_stats = self.stats[attack_type]

        generated = base.copy()

        # -------------------------------------------------------------
        # Statistical perturbation
        # -------------------------------------------------------------

        for i, feature in enumerate(self.feature_cols):

            # Binary network flags remain unchanged.
            if feature in self.binary_features:
                continue

            mean = feature_stats[feature]["mean"]
            std = feature_stats[feature]["std"]
            minimum = feature_stats[feature]["min"]
            maximum = feature_stats[feature]["max"]

            if not np.isfinite(std) or std <= 0:
                continue

            # Conservative perturbation.
            noise_scale = 0.05

            noise = self.rng.normal(
                0,
                std * noise_scale
            )

            value = base[i] + noise

            # Network-flow quantities should not become negative.
            value = max(value, 0.0)

            # Keep generated values within the observed class range.
            value = np.clip(
                value,
                minimum,
                maximum
            )

            generated[i] = value

        result = pd.Series(
            generated,
            index=self.feature_cols,
            dtype=float
        )

        # -------------------------------------------------------------
        # Basic numerical safety
        # -------------------------------------------------------------

        result = result.replace(
            [np.inf, -np.inf],
            np.nan
        )

        for feature in self.feature_cols:

            if pd.isna(result[feature]):

                feature_stats = self.stats[attack_type][feature]

                result[feature] = feature_stats["mean"]

        # -------------------------------------------------------------
        # Binary features
        # -------------------------------------------------------------

        for feature in self.binary_features:

            if feature in result.index:

                result[feature] = round(
                    result[feature]
                )

                result[feature] = np.clip(
                    result[feature],
                    0,
                    1
                )

        # -------------------------------------------------------------
        # Integer / count features
        # -------------------------------------------------------------

        for feature in INTEGER_FEATURES:

            if feature in result.index:

                result[feature] = max(
                    0,
                    round(result[feature])
                )

        # -------------------------------------------------------------
        # Destination port
        # -------------------------------------------------------------

        if "Destination Port" in result.index:

            result["Destination Port"] = int(
                np.clip(
                    round(result["Destination Port"]),
                    0,
                    65535
                )
            )

        # -------------------------------------------------------------
        # Non-negative statistical features
        # -------------------------------------------------------------

        for feature in NON_NEGATIVE_STAT_FEATURES:

            if feature in result.index:

                result[feature] = max(
                    0.0,
                    result[feature]
                )

        # -------------------------------------------------------------
        # Packet-length consistency
        # -------------------------------------------------------------

        self._repair_packet_lengths(result)

        # -------------------------------------------------------------
        # Header/count sanity
        # -------------------------------------------------------------

        self._repair_packet_counts(result)

        # -------------------------------------------------------------
        # Final non-negative protection
        # -------------------------------------------------------------

        for feature in self.feature_cols:

            if feature in self.binary_features:
                continue

            if feature in INTEGER_FEATURES:
                continue

            if np.isfinite(result[feature]):

                # Most CIC-IDS2017 numerical flow quantities
                # should be non-negative.
                result[feature] = max(
                    0.0,
                    result[feature]
                )

        result[LABEL_COL] = attack_type

        return result

    # -----------------------------------------------------------------
    # Packet length consistency
    # -----------------------------------------------------------------

    def _repair_packet_lengths(self, result):

        """
        Enforce:

            min <= mean <= max

        for packet-length feature groups.

        Only repairs impossible ordering; it does not otherwise
        reconstruct the packet statistics.
        """

        for min_feature, mean_feature, max_feature in PACKET_LENGTH_GROUPS:

            if not all(
                feature in result.index
                for feature in (
                    min_feature,
                    mean_feature,
                    max_feature,
                )
            ):
                continue

            minimum = float(result[min_feature])
            mean = float(result[mean_feature])
            maximum = float(result[max_feature])

            # Sort the three values.

            ordered = sorted(
                [
                    minimum,
                    mean,
                    maximum,
                ]
            )

            result[min_feature] = ordered[0]
            result[mean_feature] = ordered[1]
            result[max_feature] = ordered[2]

    # -----------------------------------------------------------------
    # Packet count consistency
    # -----------------------------------------------------------------

    def _repair_packet_counts(self, result):

        """
        Apply lightweight consistency rules to packet counts.

        We avoid aggressively reconstructing derived features because
        they are part of the feature distribution learned by XGBoost.
        """

        packet_pairs = [
            (
                "Total Fwd Packets",
                "Total Backward Packets",
            ),
            (
                "Subflow Fwd Packets",
                "Subflow Bwd Packets",
            ),
        ]

        for fwd_feature, bwd_feature in packet_pairs:

            if fwd_feature in result.index:
                result[fwd_feature] = max(
                    0,
                    round(result[fwd_feature])
                )

            if bwd_feature in result.index:
                result[bwd_feature] = max(
                    0,
                    round(result[bwd_feature])
                )

        # act_data_pkt_fwd cannot exceed total forward packets.
        if (
            "act_data_pkt_fwd" in result.index
            and "Total Fwd Packets" in result.index
        ):

            result["act_data_pkt_fwd"] = min(
                result["act_data_pkt_fwd"],
                result["Total Fwd Packets"]
            )

    # -----------------------------------------------------------------
    # Generate n samples
    # -----------------------------------------------------------------

    def generate(self, attack_type=None, n=1):

        """
        Generate n synthetic flows.
        """

        rows = [
            self.generate_one(attack_type)
            for _ in range(n)
        ]

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Mixed traffic
    # -----------------------------------------------------------------

    def generate_mixed(self, n=50):

        """
        Generate a mixed traffic stream.

        Attack categories are selected uniformly so that the demo
        can visibly exercise multiple detector classes.
        """

        rows = []

        for _ in range(n):

            attack_type = self.rng.choice(
                self.classes
            )

            rows.append(
                self.generate_one(attack_type)
            )

        return pd.DataFrame(rows)


# =====================================================================
# TRAIN GENERATOR
# =====================================================================

def learn_generator():

    print("=" * 60)
    print("TRAINING SYNTHETIC NETWORK FLOW GENERATOR")
    print("=" * 60)

    train_path = PROCESSED_DIR / "train.parquet"

    print(f"Loading: {train_path}")

    df = pd.read_parquet(train_path)

    with open(
        PROCESSED_DIR / "data_contract.json"
    ) as f:

        contract = json.load(f)

    feature_cols = contract["feature_columns"]

    print(
        f"Training rows: {len(df):,}"
    )

    print(
        f"Features: {len(feature_cols)}"
    )

    classes = sorted(
        df[LABEL_COL].unique()
    )

    print(
        f"Classes: {len(classes)}"
    )

    templates = {}
    stats = {}

    # -------------------------------------------------------------
    # Detect binary features
    # -------------------------------------------------------------

    binary_features = []

    for feature in feature_cols:

        values = (
            df[feature]
            .dropna()
            .unique()
        )

        if (
            len(values) <= 2
            and set(values).issubset({0, 1})
        ):

            binary_features.append(
                feature
            )

    print(
        f"Binary features detected: "
        f"{len(binary_features)}"
    )

    # -------------------------------------------------------------
    # Learn each class
    # -------------------------------------------------------------

    for class_name in classes:

        class_df = df[
            df[LABEL_COL] == class_name
        ]

        print(
            f"\n{class_name}: "
            f"{len(class_df):,} training rows"
        )

        # Keep representative subset.
        sample_size = min(
            TEMPLATES_PER_CLASS,
            len(class_df)
        )

        sampled = class_df.sample(
            n=sample_size,
            random_state=RANDOM_SEED
        )

        templates[class_name] = (
            sampled[feature_cols]
            .astype(float)
            .values
            .tolist()
        )

        class_stats = {}

        # ---------------------------------------------------------
        # Feature statistics
        # ---------------------------------------------------------

        for feature in feature_cols:

            values = pd.to_numeric(
                class_df[feature],
                errors="coerce"
            )

            values = values.replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            if len(values) == 0:

                class_stats[feature] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0
                }

                continue

            class_stats[feature] = {
                "mean": float(
                    values.mean()
                ),
                "std": float(
                    values.std()
                ),
                "min": float(
                    values.min()
                ),
                "max": float(
                    values.max()
                )
            }

        stats[class_name] = class_stats

    # -------------------------------------------------------------
    # Save generator
    # -------------------------------------------------------------

    model_data = {
        "feature_cols": feature_cols,
        "classes": classes,
        "templates": templates,
        "stats": stats,
        "binary_features": binary_features
    }

    output_path = (
        MODELS_DIR /
        "synthetic_generator.joblib"
    )

    joblib.dump(
        model_data,
        output_path,
        compress=3
    )

    print("\n" + "=" * 60)
    print("GENERATOR TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Saved: {output_path}"
    )

    size_mb = (
        output_path.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"Generator size: "
        f"{size_mb:.2f} MB"
    )

    return model_data


# =====================================================================
# SANITY TEST
# =====================================================================

def test_generator():

    generator_path = (
        MODELS_DIR /
        "synthetic_generator.joblib"
    )

    generator = SyntheticFlowGenerator(
        joblib.load(generator_path)
    )

    print("\n" + "=" * 60)
    print("GENERATOR SANITY TEST")
    print("=" * 60)

    for class_name in generator.classes:

        sample = generator.generate_one(
            class_name
        )

        print(
            f"{class_name:30s} "
            f"features="
            f"{len(sample[generator.feature_cols])}"
        )

    print(
        "\nGenerating mixed traffic..."
    )

    mixed = generator.generate_mixed(20)

    print(
        mixed[LABEL_COL]
        .value_counts()
    )

    print(
        "\nFirst generated event:"
    )

    print(
        mixed.iloc[0][
            generator.feature_cols
        ].head(10)
    )


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    learn_generator()

    test_generator()