# ИИ: Классификация мусора для сортировки отходов

Разработано в рамках производственной практики на 4 курсе МТУСИ.

## Первый запуск

```bash
# ResNet18
python src/train.py --model resnet18 --epochs 10 --batch-size 32
# DenseNet121
python src/train.py --model densenet121 --epochs 10 --batch-size 32
# MobileNetV3-Large
python src/train.py --model mobilenet_v3_large --epochs 10 --batch-size 32
# EfficientNet-B0
python src/train.py --model efficientnet_b0 --epochs 10 --batch-size 32
# ViT-B-16
python src/train.py --model vit_b_16 --epochs 10 --batch-size 16
```