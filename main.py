import sys
from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig
)
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation

load_dotenv()

if __name__ == "__main__":
    try:
        logger.info("Starting Training Pipeline")

        training_pipeline_config = TrainingPipelineConfig()

        # Data Ingestion
        data_ingestion_config   = DataIngestionConfig(
            training_pipeline_config
        )
        data_ingestion          = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logger.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

        # Data Validation
        data_validation_config   = DataValidationConfig(
            training_pipeline_config
        )
        data_validation          = DataValidation(
            data_ingestion_artifact,
            data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()

        logger.info(
            f"Data Validation Artifact: {data_validation_artifact}"
        )
        print(f"\n Data Validation complete.")
        print(
            f"   Status : {data_validation_artifact.validation_status}"
        )
        print(
            f"   Drift report: "
            f"{data_validation_artifact.drift_report_file_path}"
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)