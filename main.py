
import threading

from fastapi import FastAPI, HTTPException
from housing.api_models import PredictionRequest, PredictionResponse, TrainRequest
from housing.services import predict_housing_price, train_model_asynchronous


app = FastAPI()


@app.get("/")
def root():
    return "Housing Price Recommendation!"


@app.post("/predict", response_model=PredictionResponse)
def predict(input: PredictionRequest):
    price = predict_housing_price(state=input.state, data=input.to_dict())
    response = PredictionResponse(price=price)
    return response


@app.post("/train")
def train(input: TrainRequest):
    state = input.state
    try:
        train_model_asynchronous(state)
    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail=f"Model training for '{state}' failed! \n {err}"
        )
    return f"Model for '{state}' is scheduled for training!"
