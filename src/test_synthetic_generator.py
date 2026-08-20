import json
import joblib
import pandas as pd

from pathlib import Path

from synthetic_generator import SyntheticFlowGenerator


MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")


def main():

    generator_data = joblib.load(
        MODELS_DIR / "synthetic_generator.joblib"
    )

    generator = SyntheticFlowGenerator(
        generator_data
    )

    clf = joblib.load(
        MODELS_DIR / "xgboost_model.joblib"
    )

    le = joblib.load(
        MODELS_DIR / "label_encoder.joblib"
    )

    feature_cols = generator.feature_cols

    results = []

    print("=" * 70)
    print("SYNTHETIC GENERATOR → XGBOOST VALIDATION")
    print("=" * 70)

    for attack_type in generator.classes:

        generated = generator.generate(
            attack_type=attack_type,
            n=100
        )

        X = generated[feature_cols]

        predictions = clf.predict(X)

        predicted_labels = le.inverse_transform(
            predictions.astype(int)
        )

        correct = (
            predicted_labels == attack_type
        ).sum()

        accuracy = correct / len(predicted_labels)

        print(
            f"{attack_type:30s} "
            f"{correct:3d}/100 "
            f"({accuracy * 100:6.2f}%)"
        )

        results.append({
            "intended_class": attack_type,
            "correct": int(correct),
            "total": 100,
            "generator_detection_rate": accuracy
        })

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(
        f"Average class recognition: "
        f"{result_df['generator_detection_rate'].mean() * 100:.2f}%"
    )

    print("\nResults:")
    print(result_df.to_string(index=False))

    result_df.to_csv(
        "results/synthetic_generator_validation.csv",
        index=False
    )

    print(
        "\nSaved: "
        "results/synthetic_generator_validation.csv"
    )


if __name__ == "__main__":
    main()