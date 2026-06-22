# FILE: src/devops_ml/train.py
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import skl2onnx

# Import ONNX conversion utilities
from skl2onnx import __max_supported_opset__
from skl2onnx import to_onnx

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the housing dataset from a CSV file."""
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame):
    """Cleans data and splits it into features (X) and target (y)."""
    print("Preprocessing data...")
    df['num_rooms'] = df['num_rooms'].fillna(df['num_rooms'].mean())
    df['square_feet'] = df['square_feet'].fillna(df['square_feet'].mean())
    
    X = df[['num_rooms', 'square_feet']]
    y = df['price']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train) -> LinearRegression:
    """Trains a basic Linear Regression model."""
    print("Training the linear regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def save_model_artifacts(model, X_train, model_dir: str, base_filename: str):
    """Saves the trained model as BOTH a traditional joblib pickle and a production ONNX file."""
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Save traditional Pickle format
    pkl_filepath = os.path.join(model_dir, f"{base_filename}.pkl")
    print(f"Saving joblib pickle to {pkl_filepath}...")
    joblib.dump(model, pkl_filepath)
    
    # 2. Convert and Save ONNX format
    onnx_filepath = os.path.join(model_dir, f"{base_filename}.onnx")
    print(f"Converting and saving ONNX model to {onnx_filepath}...")
    
    # ONNX needs to know the exact data type and shape entering the model.
    # We pass an example input conversion using your X_train data type (Float32).
    initial_input = X_train.astype(np.float32).values[:1]
    onnx_model = to_onnx(model, initial_input, target_opset=__max_supported_opset__)
    
    with open(onnx_filepath, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print("All model artifacts saved successfully!")

if __name__ == "__main__":
    DATA_PATH = "data/raw/housing_data.csv"
    MODEL_DIR = "models"
    BASE_NAME = "housing_model" # Will create housing_model.pkl and housing_model.onnx
    
    try:
        data = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        save_model_artifacts(model, X_train, MODEL_DIR, BASE_NAME)
    except FileNotFoundError:
        print(f"\n[ERROR]: Could not run training. Please make sure '{DATA_PATH}' exists first!")