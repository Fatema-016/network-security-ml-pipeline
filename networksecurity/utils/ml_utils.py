import os
import sys
import numpy as np
from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def get_classification_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> ClassificationMetricArtifact:
    """
    Compute classification metrics for binary classification.
    Uses F1, Precision, Recall, ROC-AUC — NOT accuracy.
    This is critical for imbalanced datasets like CICIDS 2017.
    zero_division=0 handles edge case where model predicts
    zero attacks for a batch (common with highly imbalanced
    Thursday WebAttacks file — only 872 attack records out of 67274)
    """
    try:
        model_f1_score = f1_score(y_true, y_pred, average='binary')
        model_precision_score = precision_score(y_true, y_pred, average='binary', zero_division=0)
        model_recall_score = recall_score(y_true, y_pred, average='binary', zero_division=0)
        model_roc_auc_score = roc_auc_score(y_true, y_pred)

        logger.info(
            f"Classification Metrics  "
            f"F1: {model_f1_score:.4f} | "
            f"Precision: {model_precision_score:.4f} | "
            f"Recall: {model_recall_score:.4f} | "
            f"ROC-AUC: {model_roc_auc_score:.4f}"
        )
            # Warn if Recall is low — may need threshold tuning
        if model_recall_score < 0.80:
            logger.warning(
                f"Recall is below 0.80 ({model_recall_score:.4f}). "
                f"Consider lowering PREDICTION_THRESHOLD from 0.5 to 0.3 "
                
            )

        return ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            roc_auc_score=model_roc_auc_score
        )
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: dict,
) -> dict:
    """
    Train and evaluate multiple models.
    Returns a report dict with test F1 score for each model.
    Used in Model Trainer to compare LR, RF, XGBoost.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logger.info(f"Training model: {model_name}")
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred  = model.predict(X_test)

            train_metric = get_classification_score(y_train, y_train_pred)
            test_metric  = get_classification_score(y_test, y_test_pred)

            report[model_name] = {
                "model": model,
                "train_metric": train_metric,
                "test_metric": test_metric,
            }

            logger.info(
                f"{model_name}  "
                f"Train F1: {train_metric.f1_score:.4f} | "
                f"Test F1: {test_metric.f1_score:.4f}"
            )

        return report
    except Exception as e:
        raise NetworkSecurityException(e, sys)