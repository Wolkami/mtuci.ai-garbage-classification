from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "other",
    "paper",
    "plastic",
]

RECOMMENDATIONS = {
    "cardboard": "Сортировать как картон. Желательно удалить остатки пищи и сложить упаковку.",
    "glass": "Сортировать как стекло. По возможности промыть тару и снять крышку.",
    "metal": "Сортировать как металл. Банки желательно очистить от остатков содержимого.",
    "other": "Отнести к смешанным отходам. При необходимости уточнить правила сортировки.",
    "paper": "Сортировать как бумагу. Бумага должна быть сухой и чистой.",
    "plastic": "Сортировать как пластик. Желательно очистить упаковку от остатков пищи.",
}


@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASS_NAMES)
    )

    weights_path = Path("models/efficientnet_b0_best.pth")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.load_state_dict(
        torch.load(weights_path, map_location=device)
    )

    model.to(device)
    model.eval()

    return model, device


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
)

st.title("♻️ Классификация отходов")

st.write(
    "Загрузите изображение отхода для определения класса и получения рекомендации по сортировке."
)

uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Загруженное изображение",
        use_container_width=True,
    )

    model, device = load_model()

    transform = get_transform()

    input_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=0
        )

    predicted_class = CLASS_NAMES[
        predicted_idx.item()
    ]

    probability = confidence.item() * 100

    st.subheader("Результат")

    st.success(
        f"Класс: {predicted_class}"
    )

    st.metric(
        "Вероятность",
        f"{probability:.2f}%"
    )

    st.info(
        RECOMMENDATIONS[predicted_class]
    )

    st.subheader("Вероятности по классам")

    chart_data = {
        cls: float(probabilities[i].cpu())
        for i, cls in enumerate(CLASS_NAMES)
    }

    st.bar_chart(chart_data)