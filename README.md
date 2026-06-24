# ИИ: Классификация мусора для сортировки отходов

Разработано в рамках производственной практики на 4 курсе МТУСИ.

## Обучение моделей

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

## Оценка моделей

```bash
# ResNet18
python src/evaluate.py --model resnet18 --weights models/resnet18_best.pth
# DenseNet121
python src/evaluate.py --model densenet121 --weights models/densenet121_best.pth
# MobileNetV3-Large
python src/evaluate.py --model mobilenet_v3_large --weights models/mobilenet_v3_large_best.pth
# EfficientNet-B0
python src/evaluate.py --model efficientnet_b0 --weights models/efficientnet_b0_best.pth
# ViT-B-16
python src/evaluate.py --model vit_b_16 --weights models/vit_b_16_best.pth
```

## Сравнение моделей

```bash
python src/compare_models.py
```

## Проверка работы прототипа

Для проверки работы прототипа необходимо разместить изображения в папку `demo/samples` и выполнить консольную команду из корня проекта:

```bash
python src/inference.py --image demo/samples/<image>
```