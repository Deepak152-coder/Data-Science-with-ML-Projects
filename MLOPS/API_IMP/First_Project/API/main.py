from fastapi import FastAPI
from API.schema import DeliveryInput
from API.model import model, features
import pandas as pd

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Food Delivery Time Prediction API"}


@app.post("/predict")
def predict(data: DeliveryInput):

    df = pd.DataFrame([data.model_dump()])

    df = pd.get_dummies(df)

    df = df.reindex(columns=features, fill_value=0)

    prediction = model.predict(df)

    return {
        "Predicted_Delivery_Time": float(prediction[0])
    }