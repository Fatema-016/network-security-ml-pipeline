import sys
from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig
)
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation

load_dotenv()

if __name__ == "__main__":
    try:
        logger.info("Starting Training Pipeline")

        training_pipeline_config = TrainingPipelineConfig()

        # ── Data Ingestion ─────────────────────────────────────
        data_ingestion_config   = DataIngestionConfig(
            training_pipeline_config
        )
        data_ingestion          = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logger.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

        # ── Data Validation ────────────────────────────────────
        data_validation_config   = DataValidationConfig(
            training_pipeline_config
        )
        data_validation          = DataValidation(
            data_ingestion_artifact,
            data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logger.info(f"Data Validation Artifact: {data_validation_artifact}")

        # ── Data Transformation ────────────────────────────────
        data_transformation_config   = DataTransformationConfig(
            training_pipeline_config
        )
        data_transformation          = DataTransformation(
            data_validation_artifact,
            data_transformation_config
        )
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )

        logger.info(
            f"Data Transformation Artifact: {data_transformation_artifact}"
        )
        print(f"\n Data Transformation complete.")
        print(
            f"   Train array : "
            f"{data_transformation_artifact.transformed_train_file_path}"
        )
        print(
            f"   Test array  : "
            f"{data_transformation_artifact.transformed_test_file_path}"
        )
        print(
            f"   Preprocessor: "
            f"{data_transformation_artifact.transformed_object_file_path}"
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)