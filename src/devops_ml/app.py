import os
import numpy as np
import onnxruntime as rt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Initialize the FastAPI Application
app = FastAPI(
    title="Housing Price Prediction Service",
    description="DevOps MLOps Production API serving an ONNX model",
    version="1.0.0"
)

# 2. Define the expected input data structure using Pydantic
class HouseFeatures(BaseModel):
    num_rooms: float
    square_feet: float


# 3. Load the ONNX model into memory when the server starts
MODEL_PATH = os.path.join("models", "housing_model.onnx")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Did you run 'dvc repro'?")

# Create an ONNX Inference Session
session = rt.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

@app.get("/")
def home():
    """Health check endpoint to verify the API is alive."""
    return {"status": "healthy", "model": "ONNX Housing Model"}

@app.post("/predict")
def predict(features: HouseFeatures):
    """Prediction endpoint that takes house features and returns an estimated price."""
    try:
        # Convert incoming JSON data into a 2D NumPy array for the model
        input_data = np.array([[
            features.num_rooms,
            features.square_feet
        ]], dtype=np.float32)
        
        # Run inference using ONNX Runtime
        onnx_outputs = session.run(None, {input_name: input_data})
        prediction = float(onnx_outputs[0][0])
        
        return {
            "estimated_price": round(prediction, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")