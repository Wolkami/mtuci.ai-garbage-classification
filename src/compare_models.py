import json
from pathlib import Path

import pandas as pd


MODELS = [
    "resnet18",
    "densenet121",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "vit_b_16",
]


def main():
    rows = []

    for model_name in MODELS:
        metrics_path = Path("runs") / model_name / "metrics.json"

        if not metrics_path.exists():
            print(f"Skip missing file: {metrics_path}")
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        rows.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision_weighted"],
            "recall": metrics["recall_weighted"],
            "f1": metrics["f1_weighted"],
            "fps": metrics["fps"],
            "time_per_image_sec": metrics["time_per_image_sec"],
            "device": metrics["device"],
            "test_images": metrics["test_images"],
        })

    df = pd.DataFrame(rows)

    output_path = Path("runs") / "models_comparison.csv"
    df.to_csv(output_path, index=False)

    print(df.sort_values(by="f1", ascending=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()