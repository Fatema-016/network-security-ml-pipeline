import os
import sys
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for saving plots without display
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import ModelEvaluationConfig
from networksecurity.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact
)
from networksecurity.utils.main_utils import (
    load_numpy_array_data,
    load_object,
    save_object,
    write_yaml_file
)
from networksecurity.constants.training_pipeline import (
    MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE,
    SAVED_MODEL_DIR,
    MODEL_FILE_NAME
)


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_evaluation_config      = model_evaluation_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact       = model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def get_shap_explainer(model, X_background: np.ndarray):
        """
        Select the correct SHAP explainer based on model type.

        - LogisticRegression : LinearExplainer
          Uses shap.kmeans to create a representative summary of
          the background data (k=50 clusters) 

        - RandomForestClassifier / XGBClassifier : TreeExplainer
          Uses tree structure directly — no background data needed,
          fast and exact Shapley values for tree models.
        """
        try:
            if isinstance(model, LogisticRegression):
                logger.info(
                    "Using shap.LinearExplainer for LogisticRegression "
                    "with kmeans-summarized background (k=50)"
                )
                background_summary = shap.kmeans(X_background, 50)
                explainer = shap.LinearExplainer(
                    model, background_summary
                )
            elif isinstance(model, (RandomForestClassifier, XGBClassifier)):
                logger.info(
                    f"Using shap.TreeExplainer for "
                    f"{type(model).__name__} (no background needed)"
                )
                explainer = shap.TreeExplainer(model)
            else:
                logger.warning(
                    f"Unknown model type {type(model).__name__}, "
                    f"falling back to shap.Explainer (generic)"
                )
                explainer = shap.Explainer(model, X_background)

            return explainer

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def get_top_features_for_prediction(
        self,
        explainer,
        X_sample: np.ndarray,
        feature_names: list,
        top_n: int = 5
    ) -> list:
        """
        Compute SHAP values for a single prediction and return
        the top N contributing features with their SHAP values.

        This is what gets shown in the UI when a connection
        is flagged as malicious.
        requirement: "show which features drove the decision"
        """
        try:
            shap_values = explainer.shap_values(X_sample)

            # Handle different SHAP output formats
            # TreeExplainer for binary classification can return
            # a list [class_0_values, class_1_values] or a single array
            if isinstance(shap_values, list):
                # Take class 1 (ATTACK) SHAP values
                values = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                # Shape: (samples, features, classes)
                values = shap_values[0, :, 1]
            else:
                values = shap_values[0]

            # Pair feature names with their SHAP values
            feature_impact = list(zip(feature_names, values))

            # Sort by absolute SHAP value 
            feature_impact.sort(
                key=lambda x: abs(x[1]), reverse=True
            )

            top_features = [
                {
                    "feature"   : name,
                    "shap_value": float(round(value, 4))
                }
                for name, value in feature_impact[:top_n]
            ]

            return top_features

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
            """
            Main method:
            1. Load trained model and test data
            2. Build correct SHAP explainer based on model type
            3. Generate SHAP summary plot (feature_importance.png)
            4. Demonstrate SHAP explanation on sample predictions
            5. Save SHAP explainer for use in FastAPI
            6. Handle "no production model" case — auto-accept first model
            7. Return artifact
            """
            try:
                logger.info("Starting Model Evaluation component")

                # ── Load trained model ─────────────────────────────
                model = load_object(
                    self.model_trainer_artifact.trained_model_file_path
                )
                logger.info(f"Loaded model: {type(model).__name__}")

                # ── Load test data ─────────────────────────────────
                test_arr = load_numpy_array_data(
                    self.data_transformation_artifact
                    .transformed_test_file_path
                )
                X_test = test_arr[:, :-1]
                y_test = test_arr[:, -1]

                feature_names = [
                    "Flow Duration", "Total Fwd Packets",
                    "Total Backward Packets", "Flow Bytes/s",
                    "Flow Packets/s", "Flow IAT Mean",
                    "Fwd PSH Flags", "Bwd Packet Length Max",
                    "Fwd Packet Length Mean", "Packet Length Mean"
                ]

                os.makedirs(
                    self.model_evaluation_config.model_evaluation_dir,
                    exist_ok=True
                )

                # ── Build SHAP explainer ───────────────────────────
                background_sample = X_test[:500]
                explainer = self.get_shap_explainer(
                    model, background_sample
                )

                # ── SHAP Summary Plot — feature_importance.png ─────
                # Uses a sample of test data (500 rows) for speed —
                # full 87K rows would take too long for TreeExplainer
                # on every evaluation run
                logger.info("Generating SHAP summary plot...")
                shap_sample_size = min(500, X_test.shape[0])
                X_shap_sample = X_test[:shap_sample_size]

                shap_values_summary = explainer.shap_values(X_shap_sample)

                # Handle different SHAP output shapes across model types
                if isinstance(shap_values_summary, list):
                    # [class_0_values, class_1_values] — use ATTACK class
                    plot_values = shap_values_summary[1]
                elif len(np.array(shap_values_summary).shape) == 3:
                    # Shape: (samples, features, classes)
                    plot_values = shap_values_summary[:, :, 1]
                else:
                    plot_values = shap_values_summary

                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    plot_values,
                    X_shap_sample,
                    feature_names=feature_names,
                    show=False,
                    plot_size=(10, 6)
                )
                plt.title(
                    f"SHAP Feature Importance — {type(model).__name__}\n"
                    f"Network Security Threat Detection",
                    fontsize=12,
                    fontweight="bold"
                )
                plt.tight_layout()

                feature_importance_path = os.path.join(
                    self.model_evaluation_config.model_evaluation_dir,
                    "feature_importance.png"
                )
                plt.savefig(
                    feature_importance_path, dpi=150, bbox_inches="tight"
                )
                plt.close()

                logger.info(
                    f"SHAP summary plot saved: {feature_importance_path}"
                )

                # ── Demonstrate SHAP on a few sample predictions ───
                logger.info(
                    "Generating SHAP explanations for sample predictions"
                )

                y_pred = model.predict(X_test)
                attack_indices = np.where(y_pred == 1)[0][:3]

                sample_explanations = []
                for idx in attack_indices:
                    sample = X_test[idx:idx+1]
                    top_features = self.get_top_features_for_prediction(
                        explainer, sample, feature_names, top_n=5
                    )
                    sample_explanations.append({
                        "sample_index": int(idx),
                        "prediction"  : "ATTACK",
                        "top_features": top_features
                    })
                    logger.info(
                        f"Sample {idx} flagged as ATTACK | "
                        f"Top feature: {top_features[0]['feature']} "
                        f"(SHAP: {top_features[0]['shap_value']})"
                    )

                # ── Save SHAP explainer for FastAPI use ────────────
                shap_explainer_path = os.path.join(
                    self.model_evaluation_config.model_evaluation_dir,
                    "shap_explainer.pkl"
                )
                save_object(shap_explainer_path, obj=explainer)
                logger.info(f"SHAP explainer saved: {shap_explainer_path}")

                # ── Handle "No Production Model" case ──────────────
                # Check if a previous production model exists
                # (e.g. saved_models/model.pkl from a prior pipeline run)
                
                production_model_path = self.model_evaluation_config.production_model_path

                test_f1 = (
                    self.model_trainer_artifact
                    .test_metric_artifact.f1_score
                )

                if not os.path.exists(production_model_path):
                    # First model ever trained — nothing to compare against
                    logger.info(
                        "No production model found — this is the first "
                        "model trained. Auto-accepting as initial "
                        "production artifact."
                    )
                    is_model_accepted = True
                    improved_accuracy = test_f1  # no baseline, report raw F1

                else:
                    # Compare against existing production model
                    logger.info(
                        f"Production model found at "
                        f"{production_model_path} — comparing performance"
                    )
                    production_model = load_object(production_model_path)
                    y_prod_pred = production_model.predict(X_test)

                    from networksecurity.utils.ml_utils import (
                        get_classification_score
                    )
                    prod_metric = get_classification_score(
                        y_test, y_prod_pred
                    )

                    f1_improvement = test_f1 - prod_metric.f1_score
                    is_model_accepted = (
                        f1_improvement >=
                        self.model_evaluation_config.changed_threshold_score
                    )
                    improved_accuracy = f1_improvement

                    logger.info(
                        f"Production F1: {prod_metric.f1_score:.4f} | "
                        f"New model F1: {test_f1:.4f} | "
                        f"Improvement: {f1_improvement:.4f} | "
                        f"Accepted: {is_model_accepted}"
                    )

                # ── Save evaluation report ─────────────────────────
                evaluation_report = {
                    "model_type"          : type(model).__name__,
                    "is_model_accepted"   : bool(is_model_accepted),
                    "test_f1_score"       : float(round(test_f1, 4)),
                    "improved_accuracy"   : float(round(improved_accuracy, 4)),
                    "explainer_type"      : type(explainer).__name__,
                    "feature_importance_plot": feature_importance_path,
                    "sample_explanations" : sample_explanations
                }

                write_yaml_file(
                    file_path=self.model_evaluation_config.report_file_path,
                    content=evaluation_report,
                    replace=True
                )
                logger.info(
                    f"Evaluation report saved: "
                    f"{self.model_evaluation_config.report_file_path}"
                )

                # ── Return artifact ────────────────────────────────
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted,
                    improved_accuracy=improved_accuracy,
                    best_model_path=self.model_trainer_artifact.trained_model_file_path,
                    trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                    train_model_metric_artifact=(
                        self.model_trainer_artifact.train_metric_artifact
                    ),
                    best_model_metric_artifact=(
                        self.model_trainer_artifact.test_metric_artifact
                    )
                )

                logger.info(
                    f"Model Evaluation complete | "
                    f"Accepted: {is_model_accepted}"
                )
                return model_evaluation_artifact

            except Exception as e:
                raise NetworkSecurityException(e, sys)