import os

# ── Database constants ─────────────────────────────────────────────
DATABASE_NAME = "network_security"
COLLECTION_NAME = "network_data"

# ── Pipeline constants ─────────────────────────────────────────────
PIPELINE_NAME: str = "NetworkSecurityPipeline"
ARTIFACT_DIR: str = "artifacts"

# ── Data Ingestion constants ───────────────────────────────────────
DATA_INGESTION_COLLECTION_NAME: str = "network_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# ── Data Validation constants ──────────────────────────────────────
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"

# ── Data Transformation constants ─────────────────────────────────
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
DATA_TRANSFORMATION_TRAIN_FILE_NAME: str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_NAME: str = "test.npy"
DATA_TRANSFORMATION_PREPROCESSOR_FILE_NAME: str = "preprocessor.pkl"
DATA_TRANSFORMATION_SAMPLE_FRACTION: float = 1.0

# ── Model Trainer constants ────────────────────────────────────────
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.80
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join(
    "config", "model.yaml"
)

# ── Model Evaluation constants ─────────────────────────────────────
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_EVALUATION_REPORT_NAME: str = "evaluation_report.yaml"

# ── Model Pusher constants ─────────────────────────────────────────
MODEL_PUSHER_DIR_NAME: str = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR: str = os.path.join("saved_models")

# ── Saved model constants ──────────────────────────────────────────
SAVED_MODEL_DIR: str = os.path.join("saved_models")
MODEL_FILE_NAME: str = "model.pkl"
PREPROCESSOR_FILE_NAME: str = "preprocessor.pkl"
SHAP_EXPLAINER_FILE_NAME: str = "shap_explainer.pkl"

# ── Schema file path ───────────────────────────────────────────────
SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")

# ── MLflow / DagsHub constants ────────────────────────────────────
MLFLOW_TRACKING_URI_ENV_KEY: str = "MLFLOW_TRACKING_URI"
MLFLOW_TRACKING_USERNAME_ENV_KEY: str = "MLFLOW_TRACKING_USERNAME"
MLFLOW_TRACKING_PASSWORD_ENV_KEY: str = "MLFLOW_TRACKING_PASSWORD"


# ── Prediction threshold ───────────────────────────────────────────
# Default 0.5 — lower this if Recall needs improvement
# Cybersecurity context: False Negatives (missed attacks) are
# more costly than False Positives (false alarms)
PREDICTION_THRESHOLD: float = 0.5