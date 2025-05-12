from typing import Annotated
from fastapi import FastAPI, File, status

from ..common.types import DiogenAnswer
from .diogen import diogen


app = FastAPI()


@app.post(
    path="/tell",
    summary="Определение грузовиков и их автомобильных номеров",
    status_code=status.HTTP_200_OK,
    description="Грузовик или грузовики найдены, их автомобильные номера определены.",
    response_model=DiogenAnswer,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Грузовик не найден."}},
)
async def tell(image: Annotated[bytes, File()]):
    return diogen.predict(image)
