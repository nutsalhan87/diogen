from enum import Enum
import os
import httpx
import torch
import torchvision
from typing import Annotated
from fastapi import Depends, FastAPI, File, HTTPException, status
from pydantic import BaseModel

from diogen.common.types import PipelineAnswer
from diogen.common.draw_boxes import draw_boxes_on_image
from diogen.model.pipeline import Pipeline


pipeline = Pipeline()
pipeline.to(os.environ["DEVICE"])
PipelineDep = Annotated[Pipeline, Depends(lambda: pipeline)]

REGISTRY_URL = os.environ.get("REGISTRY_URL", "http://127.0.0.1:9000/get_info")


app = FastAPI()


def analyze_image_bytes(image_stream: bytes, pipeline: Pipeline) -> PipelineAnswer:
    img = (
        torchvision.io.decode_image(
            torch.frombuffer(image_stream, dtype=torch.uint8),
            torchvision.io.ImageReadMode.RGB,
        )
        / 255.0
    )
    ans = pipeline.predict(img)
    draw_boxes_on_image(img, ans)
    return ans


@app.post(
    path="/analyze",
    summary="Определение грузовиков и их автомобильных номеров",
    status_code=status.HTTP_200_OK,
    response_model=PipelineAnswer,
)
async def analyze(image_stream: Annotated[bytes, File()], pipeline: PipelineDep):
    return analyze_image_bytes(image_stream, pipeline)


class VehicleType(Enum):
    truck = "грузовой"
    not_truck = "не грузовой"
    unknown = "неизвестно"


class MatchResult(BaseModel):
    number: str
    matched_type: VehicleType


@app.post(
    path="/match",
    summary="Сопоставление номеров с и грузового ТС",
    status_code=status.HTTP_200_OK,
    response_model=list[MatchResult],
)
async def match(image_stream: Annotated[bytes, File()], pipeline: PipelineDep):
    results = []
    pipeline_answer = analyze_image_bytes(image_stream, pipeline)

    async with httpx.AsyncClient() as client:
        for truck in pipeline_answer:
            for plate in truck.plates:
                attempt = plate.read_attempt
                if attempt.read_status != "success":
                    continue
                number = attempt.number.upper()

                try:
                    resp = await client.get(REGISTRY_URL, params={"plate": number})
                    resp.raise_for_status()
                    is_truck = resp.json()["is_truck"]
                    matched_type = (
                        VehicleType.truck if is_truck else VehicleType.not_truck
                    )
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 404:
                        matched_type = VehicleType.unknown
                    else:
                        raise HTTPException(
                            status_code=502, detail="Ошибка запроса к реестру"
                        )
                except httpx.RequestError:
                    raise HTTPException(status_code=502, detail="Реестр недоступен")

                results.append(
                    MatchResult(
                        number=number,
                        matched_type=matched_type,
                    )
                )

    return results
