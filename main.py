import sys
from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig
)
from networksecurity.components.data_ingestion import DataIngestion

load_dotenv()

if __name__ == "__main__":
    try:
        logger.info("Starting Training Pipeline")

        # Initialize configs
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config    = DataIngestionConfig(
            training_pipeline_config
        )

        # Run Data Ingestion
        data_ingestion          = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        logger.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")
        print("\n Data Ingestion complete.")
        print(f"   Train: {data_ingestion_artifact.trained_file_path}")
        print(f"   Test : {data_ingestion_artifact.test_file_path}")

    except Exception as e:
        raise NetworkSecurityException(e, sys)