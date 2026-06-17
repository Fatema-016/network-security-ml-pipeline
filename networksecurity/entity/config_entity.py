from dataclasses import dataclass
import os
from networksecurity.constants.training_pipeline import (
    ARTIFACT_DIR,
    DATA_INGESTION_COLLECTION_NAME,
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_FEATURE_STORE_DIR,
    DATA_INGESTION_INGESTED_DIR,
    DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    DATA_TRANSFORMATION_TRAIN_FILE_NAME,
    DATA_TRANSFORMATION_TEST_FILE_NAME,
    DATA_TRANSFORMATION_PREPROCESSOR_FILE_NAME,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_TRAINER_EXPECTED_SCORE,
    MODEL_TRAINER_MODEL_CONFIG_FILE_PATH,
    DATA_VALIDATION_DIR_NAME,
    DATA_VALIDATION_VALID_DIR,
    DATA_VALIDATION_INVALID_DIR,
    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
    MODEL_EVALUATION_DIR_NAME,
    MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE,
    MODEL_EVALUATION_REPORT_NAME,
    SAVED_MODEL_DIR,
    MODEL_FILE_NAME,
    MODEL_PUSHER_DIR_NAME,
    MODEL_PUSHER_SAVED_MODEL_DIR,
    SCHEMA_FILE_PATH,
)


@dataclass
class TrainingPipelineConfig:
    """
    All component configs derive their artifact paths from this.
    """
    pipeline_name: str = "NetworkSecurityPipeline"
    artifact_dir: str = os.path.join(ARTIFACT_DIR)
    timestamp: str = None

    def __post_init__(self):
        from datetime import datetime
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_dir = os.path.join(
            ARTIFACT_DIR, self.timestamp
        )


@dataclass
class DataIngestionConfig:
    """
    Config for Data Ingestion component.
    Defines where raw data is stored after pulling from MongoDB.
    """
    training_pipeline_config: TrainingPipelineConfig

    def __post_init__(self):
        self.data_ingestion_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_FEATURE_STORE_DIR,
            "network_data.csv"
        )
        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            "train.csv"
        )
        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            "test.csv"
        )
        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = DATA_INGESTION_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    """
    Config for Data Validation component.
    Defines where validated/invalid data and drift reports are stored.
    """
    training_pipeline_config: TrainingPipelineConfig

    def __post_init__(self):
        self.data_validation_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            DATA_VALIDATION_DIR_NAME
        )
        self.valid_train_file_path = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_VALID_DIR,
            "train.csv"
        )
        self.valid_test_file_path = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_VALID_DIR,
            "test.csv"
        )
        self.invalid_train_file_path = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_INVALID_DIR,
            "train.csv"
        )
        self.invalid_test_file_path = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_INVALID_DIR,
            "test.csv"
        )
        self.drift_report_file_path = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )
        self.schema_file_path: str = SCHEMA_FILE_PATH


@dataclass
class DataTransformationConfig:
    """
    Config for Data Transformation component.
    Defines where transformed numpy arrays and preprocessor are stored.
    """
    training_pipeline_config: TrainingPipelineConfig

    def __post_init__(self):
        self.data_transformation_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            DATA_TRANSFORMATION_TRAIN_FILE_NAME
        )
        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            DATA_TRANSFORMATION_TEST_FILE_NAME
        )
        self.transformed_object_file_path = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            DATA_TRANSFORMATION_PREPROCESSOR_FILE_NAME
        )


@dataclass
class ModelTrainerConfig:
    """
    Config for Model Trainer component.
    Defines where trained models are stored and performance thresholds.
    """
    training_pipeline_config: TrainingPipelineConfig

    def __post_init__(self):
        self.model_trainer_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_TRAINED_MODEL_DIR,
            "model.pkl"
        )
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
        self.model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH


@dataclass
class ModelEvaluationConfig:
    """
    Config for Model Evaluation component.
    Defines evaluation thresholds and where reports are stored.
    """
    training_pipeline_config: TrainingPipelineConfig

    def __post_init__(self):
        self.model_evaluation_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            MODEL_EVALUATION_DIR_NAME
        )
        self.report_file_path = os.path.join(
            self.model_evaluation_dir,
            MODEL_EVALUATION_REPORT_NAME
        )
        self.changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
        self.schema_file_path: str = SCHEMA_FILE_PATH
        self.production_model_path = os.path.join(
            SAVED_MODEL_DIR, MODEL_FILE_NAME
        )


@dataclass
class ModelPusherConfig:
    """
    Config for Model Pusher component.
    Defines where the final accepted model is saved locally and on S3.
    """
    training_pipeline_config: TrainingPipelineConfig

    def __post_init__(self):
        self.model_pusher_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            MODEL_PUSHER_DIR_NAME
        )
        self.saved_model_path = os.path.join(
            MODEL_PUSHER_SAVED_MODEL_DIR,
            "model.pkl"
        )