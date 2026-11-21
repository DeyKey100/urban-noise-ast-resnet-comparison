# Сравнение AST и ResNet34 для классификации городского шума

Практическая часть статьи о классификации звуковых событий UrbanSound8K
моделями **Audio Spectrogram Transformer (AST)** и **ResNet34**.

Проект содержит единый скрипт сравнения, файлы конфигурации двух чекпоинтов,
результаты оценки и исходный текст статьи в LaTeX.

## Структура

- `evaluate_models.py` — загрузка моделей, предсказание и расчёт метрик.
- `ModelA/` — конфигурация и метаданные чекпоинта AST.
- `ModelB/` — конфигурация, метки и код инференса ResNet34.
- `evaluation_results_all/` — результаты оценки на всех десяти fold.
- `evaluation_results_fold10/` — результаты контрольной оценки на fold 10.
- `final.tex` и `sn-jnl.cls` — исходный текст и шаблон статьи.
- `MODEL_COMPARISON_REPORT.md` — краткий анализ полученных результатов.

## Основной протокол

Сравнение выполняется на наборе UrbanSound8K. Все оценки используют одинаковые
аудиофайлы и порядок меток.

```bash
pip install torch torchvision transformers librosa matplotlib pillow safetensors numpy scipy soundfile
python evaluate_models.py --fold all --device cuda --batch-size 16 --output evaluation_results_all
```

Для быстрой проверки на fold 10:

```bash
python evaluate_models.py --fold 10 --device cpu --limit 20 --output evaluation_results_fold10
```

Если CUDA недоступна, используйте `--device cpu`.

## Данные и модели

Корень проекта ожидает файл `UrbanSound8K.csv` и папку `Sounds/fold1`–
`Sounds/fold10`. Эти данные не включены в репозиторий из-за размера и условий
распространения набора UrbanSound8K.

Веса моделей (`*.safetensors`) также исключены из репозитория. Для запуска
нужно разместить их в папках `ModelA/` и `ModelB/` или настроить загрузку
соответствующих чекпоинтов в скрипте.
