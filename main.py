from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load the saved KMeans model and scaler
kmeans_model = joblib.load('K_means_model.joblib')
Kmeans_scaler = joblib.load('scaler_means.joblib')
dbscan_model = joblib.load('DBSCAN_model.joblib')
DBSCAN_scaler = joblib.load('scaler_means.joblib')


# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    current_value: float
    goals: int
    age: int
    award: int

# Preprocessing functionuvicorn app:app --reload
def preprocess_kmeans(input_features: InputFeatures):
    data = [[input_features.current_value, input_features.goals, input_features.age, input_features.award]]
    scaled_data = Kmeans_scaler.transform(data)
    return scaled_data

# Preprocessing function for DBSCAN
def preprocess_dbscan(input_features: InputFeatures):
    data = [[input_features.current_value, input_features.goals, input_features.age, input_features.award]]
    scaled_data = DBSCAN_scaler.transform(data)
    return scaled_data

@app.post("/predict_kmeans")
async def predict_kmeans(input_features: InputFeatures):
    data = preprocess_kmeans(input_features)
    y_pred = kmeans_model.predict(data)
    return {"kmeans_pred": int(y_pred[0])}

@app.post("/predict_dbscan")
async def predict_dbscan(input_features: InputFeatures):
    data = preprocess_dbscan(input_features)
    y_pred = dbscan_model.fit_predict(data)
    return {"dbscan_pred": int(y_pred[0])}
