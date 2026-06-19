import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.pipeline.prediction_pipeline import PredictionPipeline

import logging

# Separate logger for prediction monitoring — distinct from
# training pipeline logs. This file can later feed a drift
# detection. (e.g. comparing prediction distributions
# week-over-week).
prediction_logger = logging.getLogger("prediction_monitor")
prediction_logger.setLevel(logging.INFO)
prediction_log_path = os.path.join("logs", "predictions.log")
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler(prediction_log_path)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(message)s")
)
prediction_logger.addHandler(file_handler)

load_dotenv()

app = FastAPI(
    title="Network Security ML Pipeline",
    description="Near-Real-Time Network Flow Threat Detection with SHAP Explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NetworkFlowInput(BaseModel):
    flow_duration: float
    total_fwd_packets: float
    total_backward_packets: float
    flow_bytes_s: float
    flow_packets_s: float
    flow_iat_mean: float
    fwd_psh_flags: float
    bwd_packet_length_max: float
    fwd_packet_length_mean: float
    packet_length_mean: float


# ── Load model ONCE at startup ──────
prediction_pipeline = None

import boto3

S3_BUCKET = os.getenv("S3_BUCKET_NAME", "network-security-ml-pipeline-fatema")
S3_MODEL_PREFIX = "models/"

def download_models_from_s3():
    """
    Download model.pkl, preprocessor.pkl, shap_explainer.pkl
    from S3 into saved_models/ before loading.

    Uses IAM role attached to EC2 — boto3 automatically discovers
    credentials, no access keys hardcoded anywhere.
    """
    os.makedirs("saved_models", exist_ok=True)
    s3_client = boto3.client("s3")

    files = ["model.pkl", "preprocessor.pkl", "shap_explainer.pkl"]
    for filename in files:
        local_path = os.path.join("saved_models", filename)
        s3_key = f"{S3_MODEL_PREFIX}{filename}"
        try:
            s3_client.download_file(S3_BUCKET, s3_key, local_path)
            logger.info(f"Downloaded {filename} from S3")
        except Exception as e:
            logger.warning(
                f"Could not download {filename} from S3: {e}. "
                f"Falling back to local saved_models/ if present."
            )


@app.on_event("startup")
async def load_model():
    global prediction_pipeline
    try:
        # downloading fresh artifacts from S3 first
        download_models_from_s3()
        prediction_pipeline = PredictionPipeline()
        logger.info("Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"Failed to load model at startup: {e}")
        raise NetworkSecurityException(e, sys)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the dashboard UI"""
    html_path = os.path.join(
        os.path.dirname(__file__), "templates", "index.html"
    )
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "model_loaded": prediction_pipeline is not None
    }


@app.post("/predict")
async def predict(input_data: NetworkFlowInput):
    """
    Predict whether a network flow is BENIGN or ATTACK.
    Returns confidence score and top 5 SHAP-driven features
    if the flow is flagged as malicious.
    """
    try:
        if prediction_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded yet. Try again shortly."
            )

        # Map pydantic fields to model's expected feature names
        feature_dict = {
            "Flow Duration"          : input_data.flow_duration,
            "Total Fwd Packets"      : input_data.total_fwd_packets,
            "Total Backward Packets" : input_data.total_backward_packets,
            "Flow Bytes/s"           : input_data.flow_bytes_s,
            "Flow Packets/s"         : input_data.flow_packets_s,
            "Flow IAT Mean"          : input_data.flow_iat_mean,
            "Fwd PSH Flags"          : input_data.fwd_psh_flags,
            "Bwd Packet Length Max" : input_data.bwd_packet_length_max,
            "Fwd Packet Length Mean": input_data.fwd_packet_length_mean,
            "Packet Length Mean"    : input_data.packet_length_mean,
        }

        result = prediction_pipeline.predict(feature_dict)

        # ── Log request + prediction for monitoring ────────────
        # Anonymized — no IP or user-identifying info logged,
        # only feature values and prediction outcome.
        # If predictions cluster differently over time,
        # it signals retraining may be needed.
        
        logger.info(
            f"Prediction made | "
            f"{result['prediction']} | "
            f"confidence: {result['confidence']}"
        )
        prediction_logger.info(
            f"prediction={result['prediction']} | "
            f"confidence={result['confidence']} | "
            f"input={feature_dict}"
        )

        return result

    except NetworkSecurityException as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)