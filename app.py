import os
import logging
import pickle
import json
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from fastapi.concurrency import run_in_threadpool
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Pydantic Model for Input Data ---
class CarFeatures(BaseModel):
    car_ID: int
    symboling: int
    CarName: str
    fueltype: str
    aspiration: str
    doornumber: str
    carbody: str
    drivewheel: str
    enginelocation: str
    wheelbase: float
    carlength: float
    carwidth: float
    carheight: float
    curbweight: int
    enginetype: str
    cylindernumber: str
    enginesize: int
    fuelsystem: str
    boreratio: float
    stroke: float
    compressionratio: float
    horsepower: int
    peakrpm: int
    citympg: int
    highwaympg: int

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Car Price Predictor API",
    description="An API to predict car prices and demonstrate MLOps principles.",
    version="1.0.0"
)

# --- Prometheus Monitoring ---
Instrumentator().instrument(app).expose(app)
logging.info("Prometheus instrumentator has been set up.")

# Custom metrics for the regression model
predictions_total = Counter(
    "ml_predictions_total",
    "Total number of car price predictions served."
)
model_performance_gauge = Gauge(
    "ml_model_mae",  # Using Mean Absolute Error as the metric
    "Current Mean Absolute Error (MAE) of the loaded model."
)
prediction_value_histogram = Histogram(
    "ml_prediction_price_usd",
    "Distribution of predicted car prices in USD."
)

# --- MLflow Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") # Default to local
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Optional: Set credentials if using a remote artifact store like S3
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID", "")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY", "")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "")

MODEL_NAME = "RandomForestRegressor"
MODEL_STAGE = "prod"

# Global variables to hold the loaded model and its artifacts
model = None
scaler = None
model_features = None
model_mae = None

# --- Model Loading on Startup ---
@app.on_event("startup")
def load_model_and_artifacts():
    """
    Loads the model, scaler, and feature list from the MLflow Model Registry.
    This function is executed when the FastAPI application starts.
    """
    global model, scaler, model_features, model_mae
    model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"
    logging.info(f"Attempting to load model and artifacts from URI: {model_uri}")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logging.info(f"Model '{MODEL_NAME}@{MODEL_STAGE}' loaded successfully.")

        client = MlflowClient()
        model_version_details = client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE)
        run_id = model_version_details.run_id
        logging.info(f"Associated Run ID: {run_id}")

        # Download artifacts associated with the model's run
        scaler_path = client.download_artifacts(run_id, "scaler.pkl")
        features_path = client.download_artifacts(run_id, "model_features.json")

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(features_path, "r") as f:
            model_features = json.load(f)

        logging.info("Scaler and model features loaded successfully.")

        run_data = client.get_run(run_id).data
        model_mae = run_data.metrics.get("mae", 0.0)
        logging.info(f"Registered model MAE: {model_mae:.2f}")

        model_performance_gauge.set(model_mae)

    except MlflowException as e:
        logging.warning(f"Model or artifacts not found in MLflow. API will run without a model. Error: {e}")
        model, scaler, model_features, model_mae = None, None, None, 0.0
        model_performance_gauge.set(0.0)
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading the model. Error: {e}", exc_info=True)
        model, scaler, model_features, model_mae = None, None, None, 0.0
        model_performance_gauge.set(0.0)

# --- Data Preprocessing for Batch Input ---
def preprocess_batch_data(input_df: pd.DataFrame, scaler: StandardScaler, model_features: list) -> pd.DataFrame:
    """Preprocesses a DataFrame of raw car features for prediction."""
    
    # 1. Drop columns not used in training
    cols_to_drop = ['car_ID','symboling','carlength', 'carwidth', 'enginesize', 'curbweight', 'highwaympg']
    df = input_df.drop(columns=cols_to_drop, errors='ignore')

    # 2. Feature Engineering: Extract car brand
    if 'CarName' in df.columns:
        df['carbrand'] = df['CarName'].apply(lambda x: x.split(' ')[0])
        df = df.drop(columns=['CarName'])

    # 3. One-Hot Encode categorical features
    categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'carbrand']
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    # 4. Align columns with the training data (crucial step!)
    # This ensures all required columns are present and in the correct order, filling missing ones with 0.
    df_aligned = df_encoded.reindex(columns=model_features, fill_value=0)

    # 5. Scale numerical features using the loaded scaler
    numerical_cols = ['wheelbase', 'carheight', 'horsepower', 'peakrpm', 'citympg']
    cols_to_scale = [col for col in numerical_cols if col in df_aligned.columns]
    if cols_to_scale:
        df_aligned[cols_to_scale] = scaler.transform(df_aligned[cols_to_scale])

    return df_aligned

# --- Blocking Inference Function ---
def blocking_batch_inference(
    model_instance,
    scaler_instance: StandardScaler,
    features_list: list,
    input_dataframe: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Performs preprocessing and prediction for a batch of data."""
    processed_df = preprocess_batch_data(input_dataframe, scaler_instance, features_list)
    predictions = model_instance.predict(processed_df)
    results = [{"predicted_price": float(price)} for price in predictions]
    return results

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint providing API and model status."""
    model_status = "ready" if all([model, scaler, model_features]) else "not ready (model/artifacts not loaded)"
    mae_info = f"${model_mae:,.2f}" if isinstance(model_mae, float) and model_mae > 0 else "N/A"
    
    return {
        "api_status": "ok",
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_status": model_status,
        "model_performance_mae": mae_info
    }

@app.post("/predict", response_model=Dict[str, List[Dict[str, float]]])
async def predict(car_batch: List[CarFeatures]):
    """Endpoint to perform batch prediction asynchronously."""
    if not all([model, scaler, model_features]):
        raise HTTPException(status_code=503, detail="Model is not ready for predictions.")
    
    try:
        input_df = pd.DataFrame([item.model_dump() for item in car_batch])

        # Execute the blocking inference function in a separate thread
        results = await run_in_threadpool(
            blocking_batch_inference, model, scaler, model_features, input_df
        )

        # Update Prometheus metrics for each prediction
        for result in results:
            predictions_total.inc()
            prediction_value_histogram.observe(result["predicted_price"])
            
        return {"predictions": results}

    except Exception as e:
        logging.error(f"Error during async batch prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction.")

@app.post("/refresh-model")
def refresh_model():
    """Endpoint to manually trigger reloading the model from the registry."""
    logging.info("Received request to refresh the model.")
    load_model_and_artifacts()
    if model:
        return {"message": f"Model '{MODEL_NAME}@{MODEL_STAGE}' reloaded successfully."}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload the model.")