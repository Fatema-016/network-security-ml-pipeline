import os
import sys
import numpy as np
import pandas as pd

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.main_utils import load_object
from networksecurity.constants.training_pipeline import (
    SAVED_MODEL_DIR,
    MODEL_FILE_NAME,
    PREPROCESSOR_FILE_NAME,
    SHAP_EXPLAINER_FILE_NAME
)


class PredictionPipeline:
    """
    Loads the production model, preprocessor, and SHAP explainer
    ONCE at startup. Used by FastAPI for /predict requests.

    Note : uses preprocessor.transform() NOT fit_transform()
    to avoid data leakage / incorrect scaling on production data.
    """

    FEATURE_NAMES = [
        "Flow Duration", "Total Fwd Packets",
        "Total Backward Packets", "Flow Bytes/s",
        "Flow Packets/s", "Flow IAT Mean",
        "Fwd PSH Flags", "Bwd Packet Length Max",
        "Fwd Packet Length Mean", "Packet Length Mean"
    ]

    def __init__(self):
        try:
            model_path        = os.path.join(SAVED_MODEL_DIR, MODEL_FILE_NAME)
            preprocessor_path = os.path.join(SAVED_MODEL_DIR, PREPROCESSOR_FILE_NAME)
            shap_path          = os.path.join(SAVED_MODEL_DIR, SHAP_EXPLAINER_FILE_NAME)

            self.model        = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
            self.explainer    = load_object(shap_path)

            logger.info(
                f"PredictionPipeline initialized | "
                f"Model: {type(self.model).__name__}"
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, input_data: dict) -> dict:
        """
        Takes a dict of 10 network flow features, returns:
        - prediction: BENIGN or ATTACK
        - confidence: probability of predicted class
        - top_features: top 5 SHAP-driven features (only if ATTACK)
        """
        try:
            # ── Build dataframe in correct feature order ───────
            input_df = pd.DataFrame(
                [[input_data[f] for f in self.FEATURE_NAMES]],
                columns=self.FEATURE_NAMES
            )

            # ── Transform using SAVED preprocessor ─────────────
        
            scaled_input = self.preprocessor.transform(input_df)

            # ── Predict ─────────────────────────────────────────
            prediction_raw = self.model.predict(scaled_input)[0]
            prediction_proba = self.model.predict_proba(scaled_input)[0]

            prediction_label = "ATTACK" if prediction_raw == 1 else "BENIGN"
            confidence = float(prediction_proba[int(prediction_raw)])

            result = {
                "prediction": prediction_label,
                "confidence": round(confidence, 4)
            }

            # ── SHAP explanation — only meaningful for ATTACK ──
            if prediction_raw == 1:
                shap_values = self.explainer.shap_values(scaled_input)

                if isinstance(shap_values, list):
                    values = shap_values[1][0]
                elif len(np.array(shap_values).shape) == 3:
                    values = shap_values[0, :, 1]
                else:
                    values = shap_values[0]

                feature_impact = list(zip(self.FEATURE_NAMES, values))
                feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

                top_features = [
                    {
                        "feature"   : name,
                        "shap_value": float(round(value, 4))
                    }
                    for name, value in feature_impact[:5]
                ]
                result["top_features"] = top_features

                logger.info(
                    f"Prediction: ATTACK (confidence: {confidence:.4f}) | "
                    f"Top feature: {top_features[0]['feature']}"
                )
            else:
                result["top_features"] = []
                logger.info(
                    f"Prediction: BENIGN (confidence: {confidence:.4f})"
                )

            return result

        except Exception as e:
            raise NetworkSecurityException(e, sys)