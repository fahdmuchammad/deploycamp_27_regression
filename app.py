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

MODEL_NAME = "xgboost_regressor"
MODEL_STAGE = "prod"

# Global variables to hold the loaded model and its artifacts
model = None
scaler = None
model_features = None
model_mae = None
target_encoder = None  
ordinal_encoder = None 

# --- Model Loading on Startup ---
@app.on_event("startup")
def load_model_and_artifacts():
    """
    Loads the model, scaler, and feature list from the MLflow Model Registry.
    This function is executed when the FastAPI application starts.
    """
    global model, scaler, model_features, model_mae, target_encoder, ordinal_encoder

    client = MlflowClient()

    model_version_details = client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE)
    model_source_uri = model_version_details.source.split('models:/')[1]
    model_path = f's3://mlflow/1/models/{model_source_uri}/artifacts'
    logging.info(f"Attempting to load model and artifacts from URI: {model_path}")

    try:
        model = mlflow.pyfunc.load_model(model_path)
        logging.info(f"Model '{MODEL_NAME}@{MODEL_STAGE}' loaded successfully.")

        model_version_details = client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE)
        run_id = model_version_details.run_id
        logging.info(f"Associated Run ID: {run_id}")

        # Download artifacts associated with the model's run
        scaler_path = client.download_artifacts(run_id, "scaler.sav")
        features_path = client.download_artifacts(run_id, "model_features.json")
        target_encoder_path = client.download_artifacts(run_id, "target_encoder.pkl")
        ordinal_encoder_path = client.download_artifacts(run_id, "ordinal_encoder.pkl")

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        with open(features_path, "r") as f:
            model_features = json.load(f)
        with open(target_encoder_path, "rb") as f:
            target_encoder = pickle.load(f) 
        with open(ordinal_encoder_path, "rb") as f:
            ordinal_encoder = pickle.load(f)

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
def preprocess_batch_data(input_df: pd.DataFrame, scaler: StandardScaler, model_features: list, target_encoder, ordinal_encoder) -> pd.DataFrame:
    """Preprocesses a DataFrame of raw car features for prediction."""
    
    # 1. Drop columns not used in training
    cols_to_drop = ['car_ID','symboling','carlength', 'carwidth', 'enginesize', 'curbweight', 'highwaympg']
    df = input_df.drop(columns=cols_to_drop, errors='ignore')

    # 2. Feature Engineering: Extract car brand
    if 'CarName' in df.columns:
        df['carbrand'] = df['CarName'].apply(lambda x: x.split(' ')[0])
        df['cartype'] = df['CarName'].apply(lambda x: ' '.join(x.split(' ')[1:]) if len(x.split(' ')) > 1 else 'unknown')
        df = df.drop(columns=['CarName'])

    # 3. One-Hot Encode categorical features
    target_encoded_features = ['cartype', 'carbrand']  
    ordinal_features = ['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'cylindernumber', 'fuelsystem', 'enginetype']
    numerical_features = ['wheelbase', 'carheight', 'horsepower', 'peakrpm', 'citympg']
    cols_to_encode = [col for col in categorical_cols if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

# intersect with present columns
    target_encoded_features = [c for c in target_encoded_features if c in df.columns]
    ordinal_features = [c for c in ordinal_features if c in df.columns]
    numerical_features = [c for c in numerical_features if c in df.columns]

    # 4. Handle missing values
    # For numerical: simple impute with median (safe default for batch predict)
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # For categorical: fillna with 'missing'
    categorical_cols = list(set(target_encoded_features + ordinal_features))
    for col in categorical_cols:
        df[col] = df[col].fillna('missing').astype(str).str.lower()

    # 5. Apply encoders if provided, else fall back to one-hot encoding
    encoded_df = df.copy()

    # If target/ordinal encoders are provided, transform them (they must be pre-fit)
    try:
        if target_encoder is not None and len(target_encoded_features) > 0:
            encoded_df[target_encoded_features] = target_encoder.transform(encoded_df[target_encoded_features])
    except Exception:
        # fallback: leave the original columns for one-hot below
        pass

    try:
        if ordinal_encoder is not None and len(ordinal_features) > 0:
            encoded_df[ordinal_features] = ordinal_encoder.transform(encoded_df[ordinal_features])
    except Exception:
        # fallback: leave the original columns for one-hot below
        pass

    # Decide which columns remain categorical (those not transformed numerically by provided encoders)
    # If encoders were used and produced numeric columns, they are already numeric in encoded_df.
    # We'll one-hot encode any remaining object/dtype 'category' columns among categorical_cols.
    cols_for_ohe = [c for c in categorical_cols if encoded_df[c].dtype == object or encoded_df[c].dtype.name == 'category']
    if len(cols_for_ohe) > 0:
        encoded_df = pd.get_dummies(encoded_df, columns=cols_for_ohe, drop_first=True)

    # 6. Scale numerical features using provided scaler (assumed fitted)
    # If scaler is None, we will not scale but warn by using identity.
    if scaler is not None:
        num_cols_present = [c for c in numerical_features if c in encoded_df.columns]
        if len(num_cols_present) > 0:
            # scaler expects 2D array with columns in a consistent order: use num_cols_present
            try:
                encoded_df[num_cols_present] = scaler.transform(encoded_df[num_cols_present])
            except Exception as e:
                raise ValueError(f"Scaler transform failed: {e}")
    else:
        # no scaler provided: leave numericals as-is
        pass

    # 7. Ensure final columns match model_features: add missing cols filled with 0, and drop extras
    final_df = encoded_df.copy()

    # Ensure all model_features present
    for col in model_features:
        if col not in final_df.columns:
            # create missing column with zeros
            final_df[col] = 0

    # Keep only model_features and preserve order
    final_df = final_df[model_features]

    # Return final DataFrame ready for model.predict
    return final_df

# --- Blocking Inference Function ---
def blocking_batch_inference(
    model_instance,
    scaler_instance: StandardScaler,
    features_list: list,
    target_encoder,      
    ordinal_encoder,         
    input_dataframe: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Performs preprocessing and prediction for a batch of data."""
    processed_df = preprocess_batch_data(input_dataframe, scaler_instance, features_list, target_encoder, ordinal_encoder)
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
            blocking_batch_inference, model, scaler, model_features, input_df, target_encoder, ordinal_encoder
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
