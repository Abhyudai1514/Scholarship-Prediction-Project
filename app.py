from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize app
app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")

# Input schema (keep simple for Swagger)
class Prediction(BaseModel):
    feature1: float   # Countries (encoded)
    feature2: float   # Departments (encoded)
    feature3: float   # Exam score

# Prediction endpoint
@app.post("/predict")
def predict(request: Prediction):
    try:
        # Convert input to DataFrame with EXACT training column names
        data = pd.DataFrame(
            [[request.feature1, request.feature2, request.feature3]],
            columns=["Countries", "Departments", "Exam score"]
        )

        # Make prediction
        prediction = model.predict(data)

        # Return result (convert numpy → int)
        return {"prediction": int(prediction[0])}

    except Exception as e:
        # Print error in terminal for debugging
        print("ERROR:", e)
        return {"error": str(e)}}
