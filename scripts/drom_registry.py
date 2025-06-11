import asyncio
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import aiohttp

app = FastAPI(title="Drom VIN Gateway")

MAX_ATTEMPTS = 15
SLEEP_SECONDS = 1

class IsTruckResponse(BaseModel):
    number: str
    is_truck: bool


async def fetch_car_data(carplate: str) -> dict:
    buy_token_url = "https://vin.drom.ru/report/get_buy_token/"
    car_data_url = "https://vin.drom.ru/report/get_car_data/"

    async with aiohttp.ClientSession() as session:
        form_data1 = aiohttp.FormData()
        form_data1.add_field("token", "")
        form_data1.add_field("carplate", carplate)

        async with session.post(
            buy_token_url,
            data=form_data1,
            ssl=False
        ) as resp1:
            if resp1.status != 200:
                raise HTTPException(status_code=502, detail=f"Ошибка при получении токена: {resp1.status}")
            data1 = await resp1.json()

        if not data1.get("status") or "token" not in data1:
            raise HTTPException(status_code=502, detail=f"Не удалось получить токен: {data1}")

        token = data1["token"]

        for _ in range(MAX_ATTEMPTS):
            form_data2 = aiohttp.FormData()
            form_data2.add_field("token", token)

            async with session.post(
                car_data_url,
                data=form_data2,
                ssl=False
            ) as resp2:
                if resp2.status != 200:
                    raise HTTPException(status_code=502, detail=f"Ошибка при получении данных: {resp2.status}")
                data2 = await resp2.json()

                if data2.get("state") != "pending":
                    return data2

            await asyncio.sleep(SLEEP_SECONDS)

        raise HTTPException(status_code=504, detail="Истек лимит ожидания ответа от сервиса")


@app.get(
    "/get_info",
    summary="Получить информацию о том, принадлежит ли номер грузовому ТС",
    response_model=IsTruckResponse,
)
async def get_info(plate: str = Query(..., description="Номер автомобиля")):
    normalized_plate = plate.upper()
    data = await fetch_car_data(normalized_plate)

    car_data = data.get("carData")
    if not car_data or "type" not in car_data:
        raise HTTPException(status_code=404, detail="Информация о транспортном средстве не найдена")

    car_type = car_data["type"].lower()
    is_truck = any(keyword in car_type for keyword in ["груз", "тягач"])

    return IsTruckResponse(
        number=normalized_plate,
        is_truck=is_truck
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("drom_registry:app", host="127.0.0.1", port=9000, reload=True)
