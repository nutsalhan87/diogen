import argparse

parser = argparse.ArgumentParser(
    description="Программа позволяет найти на изображении грузовики и прочитать их автомобильный номер",
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
    help="Количество обработчиков запросов.",
)
parser.add_argument(
    "-a",
    "--use-accelerator",
    action="store_true",
    help="Использовать ли ускоритель, если он доступен.",
)
parser.add_argument(
    "-r",
    "--registry-server",
    type=str, default="127.0.0.1:9000", help="Адрес сервера реестра с типами ТС."
)
