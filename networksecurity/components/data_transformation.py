import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)
from networksecurity.utils.main_utils import (
    save_numpy_array_data,
    save_object
)
from networksecurity.constants.training_pipeline import (
    DATA_TRANSFORMATION_TRAIN_FILE_NAME,
    DATA_TRANSFORMATION_TEST_FILE_NAME
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig
    ):
        try:
            self.data_validation_artifact   = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def get_data_transformer_object() -> Pipeline:
        """
        Build and return the preprocessing pipeline.

        Pipeline steps:
        1. SimpleImputer(strategy='median')
           — fills NaN with median value per feature
           — median is robust to outliers in network traffic data
           — KNN imputer was considered but too slow on 435K rows

        2. RobustScaler()
           — scales using median and IQR instead of mean/std
           — critical for network data with extreme outliers


        This pipeline is fitted on X_train ONLY, then used to
        transform both X_train and X_test — preventing data leakage.
        """
        try:
            preprocessor = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  RobustScaler())
            ])
            logger.info(
                "Preprocessing pipeline created: "
                "SimpleImputer(median) - RobustScaler"
            )
            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(
        self
    ) -> DataTransformationArtifact:
        """
        Main method — full transformation flow:
        1. Read validated train/test CSVs
        2. Separate features (X) and target (y)
        3. Fit preprocessor on X_train, transform both sets
        4. Apply SMOTE on training set ONLY
        5. Concatenate features + target - numpy arrays
        6. Save train.npy, test.npy, preprocessor.pkl
        7. Return artifact
        """
        try:
            logger.info("Starting Data Transformation component")

            # ── Read validated data ────────────────────────────
            train_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = DataTransformation.read_data(
                self.data_validation_artifact.valid_test_file_path
            )
            logger.info(
                f"Train shape: {train_df.shape} | "
                f"Test shape: {test_df.shape}"
            )

            # ── Separate features and target ───────────────────
            target_column = "Label"

            # Training
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            # Testing
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            logger.info(
                f"Train class distribution before SMOTE - "
                f"BENIGN: {(y_train==0).sum():,} | "
                f"ATTACK: {(y_train==1).sum():,}"
            )

            # ── Build and fit preprocessor ─────────────────────
            # fit on X_train ONLY 
            # This prevents data leakage into the test set
            preprocessor = self.get_data_transformer_object()

            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled  = preprocessor.transform(X_test)

            logger.info(
                "Preprocessing complete — "
                "fitted on train, transformed both sets"
            )

            # ── Apply SMOTE on training set ONLY ──────────────
            # SMOTE: Synthetic Minority Over-sampling Technique
            # Creates synthetic attack samples to balance classes
            # Applied AFTER scaling — SMOTE works in feature space
            # NEVER applied to test set — would inflate metrics
            #
            # Thursday WebAttacks: only 872 attacks out of 68146 rows
            # Without SMOTE, model would ignore minority attacks
            logger.info("Applying SMOTE to training set...")
            logger.info(
                f"Before SMOTE - X_train: {X_train_scaled.shape} | "
                f"Attack ratio: "
                f"{y_train.mean()*100:.1f}%"
            )

            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_scaled, y_train
            )

            logger.info(
                f"After SMOTE - X_train: {X_train_resampled.shape} | "
                f"Attack ratio: "
                f"{y_train_resampled.mean()*100:.1f}%"
            )
            logger.info(
                f"SMOTE added "
                f"{len(X_train_resampled) - len(X_train_scaled):,} "
                f"synthetic attack samples"
            )

            # ── Concatenate features + target ──────────────────
            # np.c_ joins arrays column-wise
            # Result: [feature_1, feature_2, ..., feature_10, Label]
            train_arr = np.c_[
                X_train_resampled,
                np.array(y_train_resampled)
            ]
            test_arr = np.c_[
                X_test_scaled,
                np.array(y_test)
            ]

            logger.info(
                f"Final arrays - "
                f"Train: {train_arr.shape} | "
                f"Test: {test_arr.shape}"
            )

            # ── Save numpy arrays ──────────────────────────────
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            # ── Save preprocessor object ───────────────────────
            # compress=3 reduces file size 
            # This same object is loaded in FastAPI for predictions
            
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor
            )

            logger.info(
                f"Preprocessor saved: "
                f"{self.data_transformation_config.transformed_object_file_path}"
            )

            # ── Return artifact ────────────────────────────────
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=(
                    self.data_transformation_config
                    .transformed_object_file_path
                ),
                transformed_train_file_path=(
                    self.data_transformation_config
                    .transformed_train_file_path
                ),
                transformed_test_file_path=(
                    self.data_transformation_config
                    .transformed_test_file_path
                )
            )

            logger.info(
                f"Data Transformation complete | "
                f"Artifact: {data_transformation_artifact}"
            )
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)