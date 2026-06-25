import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = ["cardboard", "glass", "metal", "other", "paper", "plastic"]

RECOMMENDATIONS = {
    "cardboard": "Сортировать как картон. Желательно удалить остатки пищи и сложить упаковку.",
    "glass": "Сортировать как стекло. По возможности промыть тару и снять крышку.",
    "metal": "Сортировать как металл. Банки желательно очистить от остатков содержимого.",
    "other": "Отнести к смешанным отходам. Если материал неоднозначен, лучше проверить правила местной сортировки.",
    "paper": "Сортировать как бумагу. Бумага должна быть сухой и без сильных загрязнений.",
    "plastic": "Сортировать как пластик. Желательно очистить упаковку от остатков пищи.",
}


def get_device():
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


def create_model(num_classes: int):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str, default="models/efficientnet_b0_best.pth")
    parser.add_argument("--input-size", type=int, default=224)

    args = parser.parse_args()

    device = get_device()

    model = create_model(len(CLASS_NAMES))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()

    image_path = Path(args.image)
    image = Image.open(image_path).convert("RGB")

    transform = get_transform(args.input_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    probability = confidence.item() * 100

    print(f"Файл: {image_path}")
    print(f"Класс: {predicted_class}")
    print(f"Вероятность: {probability:.2f}%")
    print(f"Рекомендация: {RECOMMENDATIONS[predicted_class]}")


if __name__ == "__main__":
    main()