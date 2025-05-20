import torch

from diogen.common.types import Truck, PipelineAnswer
from diogen.model.truck import TruckDetector
from diogen.model.plate import PlateReader

class Pipeline:
    def __init__(self) -> None:
        self.truck_detector = TruckDetector()
        self.plate_reader = PlateReader()
        self._device = "cpu"

    def to(self, device):
        self._device = device
        self.truck_detector.to(device)
        self.plate_reader.to(device)

    def predict(self, img: torch.Tensor) -> PipelineAnswer:
        # imgs: (3, H, W), dtype=float
        img = img.unsqueeze(0).to(self._device)

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

        return trucks

