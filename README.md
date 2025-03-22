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
  -r, --retrain         Необходимо ли переообучить модель вместо дообучения. (default: False)
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        Темп обучения. (default: 0.001)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Размер батча. (default: 64)
  -e EPOCHS, --epochs EPOCHS
                        Количество эпох обучения. (default: 20)
  -p PART, --part PART  Доля данных, которые станут тренировочными. Значение должно быть в пределах (0; 1]. (default: 0.8)
```

Обученные модели будут сохранены в директорию `models`.

Для обучения необходимы обучающие данные, которые должны храниться в директории `data`.

# Использование

Основная система. Для работы необходимы обученные модели, хранящиеся в директории `models`.

Является сервисом. Документацию к нему можно получить по адресу `127.0.0.1:[port]/docs`.

```
usage: diogen.py [-h] [-H HOST] -p PORT [-w WORKERS] [-a]

Диоген. Программа позволяет найти на изображении грузовик, определить его тип и прочитать автомобильный номер

options:
  -h, --help            show this help message and exit
  -H HOST, --host HOST  Адрес сервиса. (default: 127.0.0.1)
  -p PORT, --port PORT  Порт сервиса. (default: None)
  -w WORKERS, --workers WORKERS
                        Количество потоков, обрабатывающих подключения. (default: 1)
  -a, --use-accelerator
                        Использовать ли ускоритель, если он доступен. (default: False)
```