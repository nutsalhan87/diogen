import argparse
from enum import Enum
from typing import Annotated
from fastapi import FastAPI, File, status
from pydantic import BaseModel
import torch
import torchvision
import uvicorn

from step.detect import DetectNN
from step.number import NumberNN
from step.type import TypeNN

device = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)


class AutoType(str, Enum):
    FREIGHTLINER = "Freightliner"
    OTHER = "Other"


class DiogenAnswer(BaseModel):
    plate: str
    auto_type: AutoType


class Diogen:
    def __init__(self) -> None:
        self.detect = DetectNN()
        self.detect.load_state_dict(torch.load("models/detect.pth", weights_only=True))
        self.detect.eval()

        self.type = TypeNN()
        self.type.load_state_dict(torch.load("models/type.pth", weights_only=True))
        self.type.eval()

        self.number = NumberNN()
        self.number.load_state_dict(torch.load("models/number.pth", weights_only=True))
        self.number.eval()

    def to(self, device):
        self.detect.to(device)
        self.type.to(device)
        self.number.to(device)

    def predict(self, img_stream: bytes) -> DiogenAnswer:
        img = torchvision.io.decode_image(
            torch.frombuffer(img_stream, dtype=torch.uint8)
        )
        # TODO

        return DiogenAnswer(plate="0AAA00", auto_type=AutoType.FREIGHTLINER)


diogen = Diogen()


app = FastAPI()


@app.post(
    path="/tell",
    summary="Определение типа грузовика и его автомобильного номера",
    status_code=status.HTTP_200_OK,
    description="Грузовик найден, его тип и автомобильный номер определены.",
    response_model=DiogenAnswer,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Грузовик не найден."}},
)
async def tell(plate: str, image: Annotated[bytes, File()]):
    return diogen.predict(image)


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
if __name__ == "__main__":
    args = parser.parse_args()
    if args.use_accelerator and torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        diogen.to(device)

    uvicorn.run("diogen:app", host=args.host, port=args.port, workers=args.workers)
