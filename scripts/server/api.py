import os
from typing import Annotated
from fastapi import Depends, FastAPI, File, status
import torch
import torchvision

from diogen.common.types import PipelineAnswer
from diogen.common.draw_boxes import draw_boxes_on_image
from diogen.model.pipeline import Pipeline


pipeline = Pipeline()
pipeline.to(os.environ["DEVICE"])
PipelineDep = Annotated[Pipeline, Depends(lambda: pipeline)]

app = FastAPI()


@app.post(
    path="/analyze",
    summary="Определение грузовиков и их автомобильных номеров",
    status_code=status.HTTP_200_OK,
    response_model=PipelineAnswer,
)
async def analyze(image_stream: Annotated[bytes, File()], pipeline: PipelineDep):
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
