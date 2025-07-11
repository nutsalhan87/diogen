from typing import List, Literal, Tuple

from pydantic import BaseModel, Field, NonNegativeInt


Coordinate = Tuple[NonNegativeInt, NonNegativeInt]
BoxCoordinates = Tuple[Coordinate, Coordinate]


class PlateReadFailed(BaseModel):
    read_status: Literal["failed"]


class PlateReadSuccess(BaseModel):
    read_status: Literal["success"]
    number: str
    confidence: float


class Plate(BaseModel):
    read_attempt: PlateReadFailed | PlateReadSuccess = Field(
        discriminator="read_status"
    )
    xyxy: BoxCoordinates


class Truck(BaseModel):
    xyxy: BoxCoordinates
    plates: List[Plate]


PipelineAnswer = List[Truck]
