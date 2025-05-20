import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
from diogen.model.plate import IMG_SIZE

YOLO_PRETRAINED_MODEL = "yolo11s.pt"
YOLO_MODEL_DESTINATION = "models/yolo11s-plates.pt"
YOLO_DATASET = "data/carplates_yolo/dataset.yaml"


def train_yolo(epochs=100, batch=16, device="cpu", lr0=0.01):
    print(
        f"Начинается обучением с параметрами epochs={epochs}, batch={batch}, device={device}, lr0={lr0}"
    )

    model = YOLO(YOLO_PRETRAINED_MODEL)
    model.train(
        data=str(Path(YOLO_DATASET).resolve()),
        imgsz=IMG_SIZE,
        single_cls=True,
        save_period=5,
        deterministic=False,
        batch=batch,
        epochs=epochs,
        device=device,
        lr0=lr0,
    )


def validate_yolo(device="cpu", batch=16):
    print(f"Начинается валидация")

    model = YOLO(YOLO_MODEL_DESTINATION)
    model.val(
        data=str(Path(YOLO_DATASET).resolve()),
        imgsz=IMG_SIZE,
        verbose=True,
        single_cls=True,
        batch=batch,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--val", action="store_true", help="Запустить валидацию")

    parser.add_argument("--epochs", type=int, default=50, help="Количество эпох")
    parser.add_argument("--batch", type=float, default=16, help="Размер батча")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Устройство",
    )
    parser.add_argument("--lr0", type=float, default=0.001, help="Шаг обучения")

    args = parser.parse_args()

    if args.batch.as_integer_ratio()[1] == 1:
        batch = int(args.batch)
    else:
        batch = args.batch

    if args.val:
        validate_yolo(device=args.device, batch=batch)
    else:
        train_yolo(epochs=args.epochs, batch=batch, device=args.device, lr0=args.lr0)


if __name__ == "__main__":
    main()
