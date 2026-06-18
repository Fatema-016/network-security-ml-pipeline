import os
import sys
import shutil

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import ModelPusherConfig
from networksecurity.entity.artifact_entity import (
    ModelEvaluationArtifact,
    DataTransformationArtifact,
    ModelPusherArtifact
)
from networksecurity.constants.training_pipeline import (
    SAVED_MODEL_DIR,
    MODEL_FILE_NAME,
    PREPROCESSOR_FILE_NAME,
    SHAP_EXPLAINER_FILE_NAME
)


class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_evaluation_artifact: ModelEvaluationArtifact,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_pusher_config          = model_pusher_config
            self.model_evaluation_artifact    = model_evaluation_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Main method:
        1. Check if model was accepted in evaluation step
        2. Copy model.pkl, preprocessor.pkl, shap_explainer.pkl
           into saved_models/ — a stable, non-timestamped location
        3. This is what FastAPI loads from at startup
        4. Return artifact
        """
        try:
            logger.info("Starting Model Pusher component")

            if not self.model_evaluation_artifact.is_model_accepted:
                logger.warning(
                    "Model was NOT accepted in evaluation step. "
                    "Skipping push to production."
                )
                return ModelPusherArtifact(
                    saved_model_path="",
                    model_pusher_dir_path=self.model_pusher_config.model_pusher_dir
                )

            os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
            os.makedirs(
                self.model_pusher_config.model_pusher_dir,
                exist_ok=True
            )

            # ── Source paths (from earlier components) ────────
            source_model_path = (
                self.model_evaluation_artifact.trained_model_path
            )
            source_preprocessor_path = (
                self.data_transformation_artifact
                .transformed_object_file_path
            )
            # SHAP explainer was saved in model_evaluation_dir
            source_shap_path = os.path.join(
                os.path.dirname(
                    self.model_evaluation_artifact.trained_model_path
                ).replace("model_trainer\\trained_model", "model_evaluation")
                if os.name == "nt"
                else os.path.dirname(
                    self.model_evaluation_artifact.trained_model_path
                ).replace("model_trainer/trained_model", "model_evaluation"),
                "shap_explainer.pkl"
            )

            # ── Destination paths (saved_models/ folder) ─
            dest_model_path = os.path.join(
                SAVED_MODEL_DIR, MODEL_FILE_NAME
            )
            dest_preprocessor_path = os.path.join(
                SAVED_MODEL_DIR, PREPROCESSOR_FILE_NAME
            )
            dest_shap_path = os.path.join(
                SAVED_MODEL_DIR, SHAP_EXPLAINER_FILE_NAME
            )

            # ── Copy model ──────────────────────────────────────
            shutil.copy(source_model_path, dest_model_path)
            logger.info(
                f"Model copied: {source_model_path} - {dest_model_path}"
            )

            # ── Copy preprocessor ───────────────────────────────
            shutil.copy(source_preprocessor_path, dest_preprocessor_path)
            logger.info(
                f"Preprocessor copied: "
                f"{source_preprocessor_path} - {dest_preprocessor_path}"
            )

            # ── Copy SHAP explainer ─────────────────────────────
            if os.path.exists(source_shap_path):
                shutil.copy(source_shap_path, dest_shap_path)
                logger.info(
                    f"SHAP explainer copied: "
                    f"{source_shap_path} - {dest_shap_path}"
                )
            else:
                logger.warning(
                    f"SHAP explainer not found at {source_shap_path}, "
                    f"skipping copy. FastAPI will need to recompute "
                    f"explainer at startup if this is missing."
                )

            logger.info(
                f"All production artifacts saved in: {SAVED_MODEL_DIR}/"
            )
            logger.info(
                f"  - {MODEL_FILE_NAME}"
            )
            logger.info(
                f"  - {PREPROCESSOR_FILE_NAME}"
            )
            logger.info(
                f"  - {SHAP_EXPLAINER_FILE_NAME}"
            )

            model_pusher_artifact = ModelPusherArtifact(
                saved_model_path=dest_model_path,
                model_pusher_dir_path=self.model_pusher_config.model_pusher_dir
            )

            logger.info(
                f"Model Pusher complete | "
                f"Saved to: {dest_model_path}"
            )
            return model_pusher_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)