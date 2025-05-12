import torch
import torchvision

from ..common.types import Truck, DiogenAnswer
from ..common.draw_boxes import draw_boxes_on_image
from ..model.truck import TruckDetector
from ..model.plate import PlateReader

class Diogen:
    def __init__(self) -> None:
        self.truck_detector = TruckDetector()
        self.plate_reader = PlateReader()

    def to(self, device):
        self.truck_detector.to(device)
        self.plate_reader.to(device)

    def predict(self, img_stream: bytes) -> DiogenAnswer:
        img = torchvision.io.decode_image(
            torch.frombuffer(img_stream, dtype=torch.uint8),
            torchvision.io.ImageReadMode.RGB,
        ).unsqueeze(0) / 255.

        truck_boxes = self.truck_detector.predict(img)[0]

        trucks = []
        for truck_box in truck_boxes:
            (x1_truck, y1_truck), (x2_truck, y2_truck) = truck_box
            truck_img = img[:, :, y1_truck:y2_truck, x1_truck:x2_truck]
            plates = self.plate_reader.predict(truck_img)[0]

            for plate in plates:
                ((x1_plate, y1_plate), (x2_plate, y2_plate)) = plate.xyxy
                new_xyxy = (
                    (x1_truck + x1_plate, y1_truck + y1_plate),
                    (x1_truck + x2_plate, y1_truck + y2_plate)
                )
                plate.xyxy = new_xyxy
            
            trucks.append(Truck(xyxy=truck_box, plates=plates))

        draw_boxes_on_image(img, trucks)
        return trucks


diogen = Diogen()
