from abc import abstractmethod
import argparse
import torch

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else  torch.device("cpu")

class Diogen:
    @abstractmethod
    def load(self):
        ...


class CNN(Diogen):
    pass


class Transformer(Diogen):
    pass


parser = argparse.ArgumentParser(
    prog="diogen.py",
    description="Диоген. Программа позволяет найти на изображении грузовик, определить его тип и прочитать автомобильный номер",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("-n", "--number", required=True)
parser.add_argument("-i", "--image", type=argparse.FileType("r"), required=True, metavar='PATH_TO_IMG')
parser.add_argument(
    "-a",
    "--algorithm",
    choices=["cnn", "tf"],
    default="cnn"
)

if __name__ == "__main__":
    args = parser.parse_args()
