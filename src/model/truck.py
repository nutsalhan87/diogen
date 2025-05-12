from typing import List
import torch
from ultralytics import YOLO

from ..common.types import BoxCoordinates
from ..common.letterbox import LetterboxTransform

IMG_SIZE = 640
MIN_CONF = 0.75
MODEL_PATH = "models/yolo11l.pt"


class TruckDetector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.model.eval()
        for cls, label in self.model.names.items():
            if label == "truck":
                self._truck_class = cls
                break
        if self._truck_class is None:
            raise Exception("Truck detector model does not have 'truck' label")

    def to(self, device):
        self.model.to(device)

    @torch.inference_mode()
    def predict(self, imgs: torch.Tensor) -> List[List[BoxCoordinates]]:
        # imgs: (B, 3, H, W)
        letterbox = LetterboxTransform(imgs, IMG_SIZE)
        results = self.model.predict(
            letterbox.resized, imgsz=IMG_SIZE, conf=MIN_CONF, classes=[self._truck_class]
        )

        each_result_boxes = []
        for batch_idx, result in enumerate(results):
            if result.boxes is None:
                each_result_boxes.append([])
                continue
            boxes = result.boxes.cpu()

            original_boxes = letterbox.reverse_boxes(batch_idx, boxes.xyxy) # type: ignore

            result_boxes: List[BoxCoordinates] = []
            for xyxy in original_boxes:
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                result_boxes.append(((x1, y1), (x2, y2)))
            each_result_boxes.append(result_boxes)

        return each_result_boxes
