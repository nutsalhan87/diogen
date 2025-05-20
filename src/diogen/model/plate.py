import re
from typing import List
import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms.v2 import functional as F

from diogen.common.types import Plate, PlateReadSuccess, PlateReadFailed
from diogen.common.letterbox import LetterboxTransform

IMG_SIZE = 480
MIN_CONF = 0.75
DETECT_MODEL_PATH = "models/yolo11s-plates.pt"
READER_MODEL_PATH = "models/crnn-plates.pt"
PLATE_INPUT_SIZE = [32, 150]  # [height, width]

ALPHABET = "0123456789АВЕКМНОРСТУХ"
BLANK_IDX = len(ALPHABET)
PLATE_REGEX = re.compile(r"^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$")


def is_plausible_plate(text: str) -> bool:
    return bool(PLATE_REGEX.fullmatch(text))


def resize_plate(tensor_img: torch.Tensor) -> torch.Tensor:
    # tensor_img: 3xHxW, RGB
    img = F.resize(tensor_img, PLATE_INPUT_SIZE, antialias=True)
    img = img.mean(dim=0, keepdim=True)  # Convert to 1xHxW (monochrome)
    return img


class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # CNN-бэкбон: resnet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=(7, 7),
            stride=(1, 1),
            padding=(3, 3),
            bias=False,
            dilation=2,
        )
        resnet.maxpool = nn.Identity()  # type: ignore
        self.cnn = nn.Sequential(
            *list(resnet.children())[:-2], nn.AdaptiveMaxPool2d((1, None))
        )

        # BiLSTM + Dropout
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=256, num_layers=2, bidirectional=True
        )
        self.dropout = nn.Dropout(p=0.3)

        # Классификатор
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 32, W)
        x = self.cnn(x)  # -> (B, C, 1, T)
        x = x.squeeze(2)  # -> (B, C, T)
        x = x.permute(2, 0, 1)  # -> (T, B, C)
        x, _ = self.lstm(x)  # -> (T, B, 512)
        x = self.dropout(x)
        x = self.classifier(x)  # -> (T, B, C)
        return x

    def decode_with_confidence(self, logits: torch.Tensor) -> List[tuple[str, float]]:
        # logits: (T, B, C)
        _T, B, _C = logits.shape
        probs = nn.functional.softmax(logits, dim=2)  # (T, B, C)
        pred = torch.argmax(probs, dim=2)  # (T, B)
        max_probs = probs.gather(2, pred.unsqueeze(-1)).squeeze(2)  # (T, B)

        results: List[tuple[str, float]] = []

        for b in range(B):
            pred_b = pred[:, b].tolist()
            probs_b = max_probs[:, b]

            chars = []
            prev = -1
            for p in pred_b:
                if p != prev and p != BLANK_IDX:
                    chars.append(ALPHABET[p])
                prev = p

            confidence = float(probs_b.mean())  # Средняя уверенность по временным шагам
            results.append(("".join(chars), confidence))

        return results


class PlateReader:
    def __init__(self):
        self.detect_model = YOLO(DETECT_MODEL_PATH)
        self.detect_model.eval()
        self.crnn = CRNN(num_classes=len(ALPHABET) + 1)
        state_dict = torch.load(READER_MODEL_PATH, map_location="cpu")
        self.crnn.load_state_dict(state_dict)
        self.crnn.eval()

    def to(self, device):
        self.detect_model.to(device)
        self.crnn.to(device)

    @torch.inference_mode()
    def predict(self, imgs: torch.Tensor) -> List[List[Plate]]:
        # imgs: (B, 3, H, W)
        letterbox = LetterboxTransform(imgs, IMG_SIZE)
        results = self.detect_model.predict(
            letterbox.resized,
            imgsz=IMG_SIZE,
            conf=MIN_CONF,
            verbose=False,
        )

        imgs_plates = []
        for batch_idx, result in enumerate(results):
            if result.boxes is None:
                imgs_plates.append([])
                continue
            boxes = result.boxes.cpu()

            original_boxes = letterbox.reverse_boxes(batch_idx, boxes.xyxy)  # type: ignore

            plates: List[Plate] = []
            for xyxy in original_boxes:
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    plates.append(
                        Plate(
                            xyxy=((x1, y1), (x2, y2)),
                            read_attempt=PlateReadFailed(read_status="failed"),
                        )
                    )
                    continue

                plate_crop = imgs[:, :, y1:y2, x1:x2]
                plate_tensor = resize_plate(plate_crop.squeeze(0)).unsqueeze(0)
                logits = self.crnn(plate_tensor)
                number, confidence = self.crnn.decode_with_confidence(logits)[0]

                if (
                    confidence < MIN_CONF
                    or number == ""
                    or not is_plausible_plate(number)
                ):
                    plates.append(
                        Plate(
                            xyxy=((x1, y1), (x2, y2)),
                            read_attempt=PlateReadFailed(read_status="failed"),
                        )
                    )
                else:
                    plates.append(
                        Plate(
                            xyxy=((x1, y1), (x2, y2)),
                            read_attempt=PlateReadSuccess(
                                read_status="success",
                                number=number,
                                confidence=confidence,
                            ),
                        )
                    )
            imgs_plates.append(plates)

        return imgs_plates
