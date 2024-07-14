from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
import pickle
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    Manufacturer: str
    Category: int
    Screen: str
    GPU: int
    OS: int
    CPU_core: int
    Screen_Size_cm: float
    CPU_frequency: float
    RAM_GB: int
    Storage_GB_SSD: int
    Weight_kg: float

pipeline = pickle.load(open('model.pkl', 'rb'))

@app.get("/")
async def read_index():
    return FileResponse(os.path.join("templates", "index.html"))


@app.post("/predict")
def predict(data: InputData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.model_dump()])
    
    # Generate predictions
    try:
        prediction = pipeline.predict(input_data)
        return {"prediction": np.exp(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)