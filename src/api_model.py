from fastapi import FastAPI
import requests
from pydantic import BaseModel
from typing import List


app = FastAPI()
MLFLOW_MODEL_URI = "http://mlflow:5000/invocations"
print(MLFLOW_MODEL_URI)

class DataRequest(BaseModel):
    data: List[List[float]]  

@app.post("/predict/")
def predict(request: DataRequest):
    print(request)
    data = request.data  
    payload = {"instances": data}
    response = requests.post(MLFLOW_MODEL_URI, json=payload)
    return response.json()

