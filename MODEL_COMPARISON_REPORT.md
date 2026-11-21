# Сравнение ModelA и ModelB



## Краткий вывод

Для общего использования на UrbanSound8K лучше ModelA (AST). На полном
наборе из 8 732 размеченных записей она существенно превосходит ModelB:

|Модель|Accuracy|Macro F1|
|-|-:|-:|
|ModelA (AST)|**0.9699**|**0.9706**|
|ModelB (ResNet34)|0.7854|0.7932|

Разница Accuracy составляет **+18.45 процентных пункта** в пользу ModelA.
Bootstrap 95% доверительный интервал разницы:
**\[+17.53 п.п.; +19.37 п.п.]**. Интервал не пересекает ноль, поэтому
преимущество не объясняется случайным разбросом конкретных записей.

Полные исходные результаты находятся в
[`evaluation\_results\_all/summary.json`](./evaluation_results_all/summary.json),
а подробные предсказания — в
[`evaluation\_results\_all/predictions.csv`](./evaluation_results_all/predictions.csv).

## Почему fold 10 дал другой результат

На отдельном fold 10 результаты были следующими:

|Модель|Accuracy|Macro F1|
|-|-:|-:|
|ModelA|0.8889|0.9009|
|ModelB|**0.9654**|**0.9658**|

Поэтому оценка только fold 10 действительно была неоднозначной и давала преимущество ModelB.

### ModelA

На полном наборе наиболее частые ошибки:

* `air\_conditioner -> street\_music`: 32;
* `drilling -> jackhammer`: 32;
* `gun\_shot -> drilling`: 25;
* `jackhammer -> drilling`: 20;
* `air\_conditioner -> engine\_idling`: 19.

ModelA сохраняет высокий recall для всех классов: от 0.9278 для `gun\_shot`
до 0.9920 для `street\_music`.



### ModelB

Наиболее серьёзная проблема ModelB — различение строительных шумов:

* `jackhammer -> drilling`: **460**;
* `air\_conditioner -> drilling`: 119;
* `engine\_idling -> drilling`: 102;
* `car\_horn -> street\_music`: 86;
* `dog\_bark -> children\_playing`: 82.

Хуже всего ModelB распознаёт `jackhammer` (recall 0.4770, F1 0.6252),
`drilling` (F1 0.6930) и `car\_horn` (F1 0.7440). Лучшие результаты у
`children\_playing` (recall 0.9130), `gun\_shot` (0.9198) и `drilling`
(0.9570), но высокий recall `drilling` сопровождается большим числом
ложных срабатываний.

## Практический выбор

* **Для максимальной точности:** ModelA.
* **Для равномерного качества между классами:** ModelA, поскольку её
Macro F1 выше на 17.74 процентных пункта.
* **Для Edge/IoT:** ModelB всё ещё может быть интересна из-за CNN-архитектуры,
но текущий checkpoint требует дообучения, особенно на паре
`drilling`/`jackhammer`. Меньшая архитектурная сложность сама по себе не
компенсирует потерю качества.

