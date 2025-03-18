# Diogen

Система сопоставления типа транспортного средства и его государственного номера.

# Обучение

Для работы основной системы нужны обученные модели. Эти модели должны находиться в директории `models`. Можно либо воспользоваться готовыми моделями, либо обучить ее самому. Для вторых целей есть программа `src/train.py`:

```
usage: train.py [-h] -s {detect,type,number} [-r] [-l LEARNING_RATE] [-b BATCH_SIZE] [-e EPOCHS] [-p PART]

Диоген учащийся.

options:
  -h, --help            show this help message and exit
  -s {detect,type,number}, --step {detect,type,number}
                        Модель, которую необходимо обучить/дообучить. (default: None)
  -r, --retrain
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  -e EPOCHS, --epochs EPOCHS
  -p PART, --part PART  Доля данных, которые станут тренировочными. Значение должно быть в пределах (0; 1]. (default: 0.8)
```

Обученные модели будут сохранены в директорию `models`.

Для обучения необходимы обучающие данные, которые должны храниться в директории `data`.

# Использование

Основная система. Для работы необходимы обученные модели, хранящиеся в директории `models`.

```
usage: diogen.py [-h] -n NUMBER -i PATH_TO_IMG

Диоген. Программа позволяет найти на изображении грузовик, определить его тип и прочитать автомобильный номер.

options:
  -h, --help            show this help message and exit
  -n NUMBER, --number NUMBER
  -i PATH_TO_IMG, --image PATH_TO_IMG
```