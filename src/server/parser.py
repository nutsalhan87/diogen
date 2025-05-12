import argparse

parser = argparse.ArgumentParser(
    prog="diogen.py",
    description="Диоген. Программа позволяет найти на изображении грузовик, определить его тип и прочитать автомобильный номер",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-H", "--host", type=str, default="127.0.0.1", help="Адрес сервиса."
)
parser.add_argument("-p", "--port", type=int, required=True, help="Порт сервиса.")
parser.add_argument(
    "-w",
    "--workers",
    type=int,
    default=1,
    help="Количество потоков, обрабатывающих подключения.",
)
parser.add_argument(
    "-a",
    "--use-accelerator",
    action="store_true",
    help="Использовать ли ускоритель, если он доступен.",
)
