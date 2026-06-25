import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


CLASS_NAMES = ["cardboard", "glass", "metal", "other", "paper", "plastic"]
NUM_CLASSES = len(CLASS_NAMES)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model


def save_confusion_matrix(cm, class_names, output_path: Path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = range(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=224)

    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    test_dir = Path(args.data_dir) / "test"
    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=get_transform(args.input_size),
    )

    print("Classes:", test_dataset.classes)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = create_model(args.model, NUM_CLASSES)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    end_time = time.perf_counter()

    total_time = end_time - start_time
    images_count = len(test_dataset)
    time_per_image = total_time / images_count
    fps = images_count / total_time

    accuracy = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        all_labels,
        all_preds,
        target_names=test_dataset.classes,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds)

    runs_dir = Path("runs") / args.model
    runs_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": args.model,
        "weights": args.weights,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "total_inference_time_sec": total_time,
        "time_per_image_sec": time_per_image,
        "fps": fps,
        "device": str(device),
        "test_images": images_count,
    }

    with open(runs_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    with open(runs_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    save_confusion_matrix(
        cm,
        test_dataset.classes,
        runs_dir / "confusion_matrix.png",
    )

    print("\nEvaluation finished.")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"FPS:       {fps:.2f}")
    print(f"Time/img:  {time_per_image:.6f} sec")
    print(f"Saved to:  {runs_dir}")


if __name__ == "__main__":
    main()