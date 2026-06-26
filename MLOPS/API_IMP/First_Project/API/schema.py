from pydantic import BaseModel, Field


class DeliveryInput(BaseModel):
    Distance_km: float = Field(..., examples=[8.5])
    Preparation_Time_min: float = Field(..., examples=[20])
    Weather: str = Field(..., examples=["Sunny"])
    Traffic_Level: str = Field(..., examples=["High"])