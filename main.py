import sys
from dotenv import load_dotenv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logger
from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.components.model_evaluation import ModelEvaluation

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

        # ── Model Trainer ──────────────────────────────────────
        model_trainer_config   = ModelTrainerConfig(
            training_pipeline_config
        )
        model_trainer          = ModelTrainer(
            model_trainer_config,
            data_transformation_artifact
        )
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        logger.info(
            f"Model Trainer Artifact: {model_trainer_artifact}"
        )
        print(f"\n Model Trainer complete.")
        print(
            f"   Best model : "
            f"{model_trainer_artifact.trained_model_file_path}"
        )
        print(
            f"   Test F1    : "
            f"{model_trainer_artifact.test_metric_artifact.f1_score:.4f}"
        )
        print(
            f"   Test Recall: "
            f"{model_trainer_artifact.test_metric_artifact.recall_score:.4f}"
        )

        # ── Model Evaluation ────────────────────────────────────
        model_evaluation_config   = ModelEvaluationConfig(
            training_pipeline_config
        )
        model_evaluation          = ModelEvaluation(
            model_evaluation_config,
            data_transformation_artifact,
            model_trainer_artifact
        )
        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

        logger.info(
            f"Model Evaluation Artifact: {model_evaluation_artifact}"
        )
        print(f"\n Model Evaluation complete.")
        print(
            f"   Accepted   : {model_evaluation_artifact.is_model_accepted}"
        )
        print(
            f"   Test F1    : "
            f"{model_evaluation_artifact.best_model_metric_artifact.f1_score:.4f}"
        )
        print(
            f"   SHAP plot saved in: "
            f"{model_evaluation_config.model_evaluation_dir}"
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)