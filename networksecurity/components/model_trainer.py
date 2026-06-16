import os
import sys
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import dagshub
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from networksecurity.utils.main_utils import (
    load_numpy_array_data,
    save_object,
    load_object
)
from networksecurity.utils.ml_utils import (
    get_classification_score,
    evaluate_models
)
from networksecurity.constants.training_pipeline import (
    MODEL_TRAINER_EXPECTED_SCORE,
    PREDICTION_THRESHOLD
)

load_dotenv()


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config          = model_trainer_config
            self.data_transformation_artifact  = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _setup_mlflow(self):
        """
        Initialize MLflow tracking via DagsHub.
        All experiment runs will be visible at:
        https://dagshub.com/fatemahab.786/network-security-ml-pipeline.mlflow
        """
        try:
            dagshub.init(
                repo_owner="fatemahab.786",
                repo_name="network-security-ml-pipeline",
                mlflow=True
            )
            mlflow.set_experiment("NetworkSecurity-ModelTrainer")
            logger.info("MLflow tracking initialized via DagsHub")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _get_models_and_params(self) -> dict:
        """
        Define 3 models with hyperparameter grids.

        Model selection for CICIDS 2017:
        - LogisticRegression : linear baseline, fast, interpretable
                               needs scaling (done via preprocessor)
        - RandomForest       : ensemble, handles non-linearity well
                               robust to outliers in network data
        - XGBClassifier      : boosting, best for tabular data
                               scale_pos_weight handles imbalance
                               (though SMOTE already balanced train)

        """
        models = {
            "LogisticRegression": LogisticRegression(
                random_state=42,
                n_jobs=-1
            ),
            "RandomForestClassifier": RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            "XGBClassifier": XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0
            )
        }

        params = {
            "LogisticRegression": {
                "C"        : [0.1, 1.0, 10.0],
                "solver"   : ["lbfgs"],
                "max_iter" : [1000]
            },
            "RandomForestClassifier": {
                "n_estimators"    : [100, 200],
                "max_depth"       : [10, 20],
                "min_samples_split": [2, 5]
            },
            "XGBClassifier": {
                "n_estimators"  : [100, 200],
                "learning_rate" : [0.05, 0.1],
                "max_depth"     : [4, 6]
            }
        }

        return models, params

    def _train_with_grid_search(
        self,
        model_name: str,
        model,
        params: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Run GridSearchCV for one model, log all runs to MLflow.
        Returns best model and its metrics.
        """
        try:
            logger.info(f"Starting GridSearchCV for {model_name}")

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=3,
                scoring="f1",
                n_jobs=-1,
                verbose=0,
                refit=True
            )
            grid_search.fit(X_train, y_train)

            best_model  = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_f1  = grid_search.best_score_

            logger.info(
                f"{model_name} best params: {best_params} | "
                f"CV F1: {best_cv_f1:.4f}"
            )

            # Evaluate on train and test
            y_train_pred = best_model.predict(X_train)
            y_test_pred  = best_model.predict(X_test)

            train_metric = get_classification_score(
                y_train, y_train_pred
            )
            test_metric  = get_classification_score(
                y_test, y_test_pred
            )

            # Log to MLflow
            with mlflow.start_run(run_name=model_name):
                # Log hyperparameters
                mlflow.log_params(best_params)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param(
                    "prediction_threshold", PREDICTION_THRESHOLD
                )

                # Log metrics — F1, Precision, Recall, ROC-AUC
                mlflow.log_metrics({
                    "train_f1"       : train_metric.f1_score,
                    "train_precision": train_metric.precision_score,
                    "train_recall"   : train_metric.recall_score,
                    "train_roc_auc"  : train_metric.roc_auc_score,
                    "test_f1"        : test_metric.f1_score,
                    "test_precision" : test_metric.precision_score,
                    "test_recall"    : test_metric.recall_score,
                    "test_roc_auc"   : test_metric.roc_auc_score,
                    "cv_f1"          : best_cv_f1,      #cv_f1: The average F1 score across cross-validation folds. It measures how consistently model performs on different subsets of training data during hyperparameter tuning.
                })

                # Log model to MLflow
                if model_name == "XGBClassifier":
                    mlflow.xgboost.log_model(
                        best_model, "model"
                    )
                else:
                    mlflow.sklearn.log_model(
                        best_model, "model"
                    )

                logger.info(
                    f"{model_name} logged to MLflow | "
                    f"Test F1: {test_metric.f1_score:.4f} | "
                    f"Test Recall: {test_metric.recall_score:.4f}"
                )

            return {
                "model"        : best_model,
                "best_params"  : best_params,
                "cv_f1"        : best_cv_f1,
                "train_metric" : train_metric,
                "test_metric"  : test_metric
            }

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Main method — trains LR + RF + XGBoost with GridSearchCV,
        logs all runs to MLflow via DagsHub, selects best model
        by test F1-score, saves it as model.pkl.
        """
        try:
            logger.info("Starting Model Trainer component")

            # ── Load transformed arrays ────────────────────────
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact
                .transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact
                .transformed_test_file_path
            )

            # Split features and target
            # Last column is Label (0=BENIGN, 1=ATTACK)
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

            logger.info(
                f"Train: {X_train.shape} | "
                f"Test: {X_test.shape}"
            )
            logger.info(
                f"Train attack ratio: {y_train.mean()*100:.1f}% "
                f"(post-SMOTE — should be ~50%)"
            )
            logger.info(
                f"Test attack ratio: {y_test.mean()*100:.1f}% "
                f"(original distribution — should be ~35%)"
            )

            # ── Setup MLflow ───────────────────────────────────
            self._setup_mlflow()

            # ── Train all 3 models ─────────────────────────────
            models, params = self._get_models_and_params()
            results = {}

            for model_name, model in models.items():
                logger.info(f"{'='*50}")
                logger.info(f"Training: {model_name}")
                logger.info(f"{'='*50}")

                result = self._train_with_grid_search(
                    model_name=model_name,
                    model=model,
                    params=params[model_name],
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test
                )
                results[model_name] = result

            # ── Model comparison summary ───────────────────────
            logger.info("=" * 60)
            logger.info("MODEL COMPARISON SUMMARY")
            logger.info("=" * 60)
            for name, result in results.items():
                logger.info(
                    f"{name:<25} | "
                    f"Train F1: {result['train_metric'].f1_score:.4f} | "
                    f"Test F1: {result['test_metric'].f1_score:.4f} | "
                    f"Test Recall: {result['test_metric'].recall_score:.4f}"
                )
            logger.info("=" * 60)

            # ── Select best model by test F1 ───────────────────
            best_model_name = max(
                results,
                key=lambda k: results[k]["test_metric"].f1_score
            )
            best_result      = results[best_model_name]
            best_model       = best_result["model"]
            best_test_metric = best_result["test_metric"]

            logger.info(
                f"Best model: {best_model_name} | "
                f"Test F1: {best_test_metric.f1_score:.4f}"
            )

            # ── Check against expected threshold ───────────────
            if best_test_metric.f1_score < MODEL_TRAINER_EXPECTED_SCORE:
                raise Exception(
                    f"Best model {best_model_name} F1 "
                    f"({best_test_metric.f1_score:.4f}) is below "
                    f"threshold ({MODEL_TRAINER_EXPECTED_SCORE}). "
                    f"Check data quality or increase training data."
                )

            # ── Save best model ────────────────────────────────
            os.makedirs(
                os.path.dirname(
                    self.model_trainer_config.trained_model_file_path
                ),
                exist_ok=True
            )
            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logger.info(
                f"Best model saved: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            # ── Log best model summary to MLflow ──────────────
            with mlflow.start_run(run_name=f"BEST_{best_model_name}"):
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_params(best_result["best_params"])
                mlflow.log_metrics({
                    "best_test_f1"       : best_test_metric.f1_score,
                    "best_test_precision": best_test_metric.precision_score,
                    "best_test_recall"   : best_test_metric.recall_score,
                    "best_test_roc_auc"  : best_test_metric.roc_auc_score,
                })

            # ── Return artifact ────────────────────────────────
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=(
                    self.model_trainer_config.trained_model_file_path
                ),
                train_metric_artifact=best_result["train_metric"],
                test_metric_artifact=best_test_metric
            )

            logger.info(
                f"Model Trainer complete | "
                f"Best: {best_model_name} | "
                f"F1: {best_test_metric.f1_score:.4f}"
            )
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)