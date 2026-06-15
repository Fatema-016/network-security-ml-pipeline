import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from networksecurity.utils.main_utils import (
    read_yaml_file,
    write_yaml_file
)
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config  = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(
        self, dataframe: pd.DataFrame
    ) -> bool:
        """
        Check if all columns defined in schema.yaml
        exist in the dataframe.
        """
        try:
            schema_columns = list(
                self._schema_config["columns"].keys()
            )
            dataframe_columns = list(dataframe.columns)

            missing_columns = [
                col for col in schema_columns
                if col not in dataframe_columns
            ]

            if missing_columns:
                logger.warning(
                    f"Missing columns: {missing_columns}"
                )
                return False

            logger.info(
                f"All {len(schema_columns)} schema columns "
                f"present in dataframe"
            )
            return True

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def is_numerical_column_exist(
        self, dataframe: pd.DataFrame
    ) -> bool:
        """
        Check if all numerical columns defined in schema.yaml
        exist in the dataframe and are numeric dtype.
        """
        try:
            numerical_columns = self._schema_config[
                "numerical_columns"
            ]
            missing_numerical = [
                col for col in numerical_columns
                if col not in dataframe.columns
            ]

            if missing_numerical:
                logger.warning(
                    f"Missing numerical columns: "
                    f"{missing_numerical}"
                )
                return False

            logger.info(
                f"All {len(numerical_columns)} numerical "
                f"columns present"
            )
            return True

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05
    ) -> bool:
        """
        Detect data drift between train and test sets
        using the Kolmogorov-Smirnov test.

        KS test checks if two samples come from the same
        distribution. p-value < threshold means drift detected.

        threshold=0.05 is standard — 5% significance level.
        In cybersecurity, some drift is expected between
        train (Wednesday) and test (mixed days) data.
        """
        try:
            numerical_columns = self._schema_config[
                "numerical_columns"
            ]
            drift_report = {}
            drift_detected = False

            for column in numerical_columns:
                base_data    = base_df[column].dropna()
                current_data = current_df[column].dropna()

                # KS test — checks distribution similarity
                ks_stat, p_value = ks_2samp(
                    base_data, current_data
                )

                is_drift = p_value < threshold
                if is_drift:
                    drift_detected = True

                drift_report[column] = {
                    "ks_statistic" : float(round(ks_stat, 4)),
                    "p_value"      : float(round(p_value, 4)),
                    "drift_detected": bool(is_drift)
                }

                logger.info(
                    f"{column:<30} | "
                    f"KS: {ks_stat:.4f} | "
                    f"p-value: {p_value:.4f} | "
                    f"Drift: {is_drift}"
                )

            # Save drift report to yaml
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=drift_report,
                replace=True
            )
            logger.info(
                f"Drift report saved: "
                f"{self.data_validation_config.drift_report_file_path}"
            )

            return drift_detected

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Main method — orchestrates full data validation:
        1. Read train/test CSVs
        2. Validate column count
        3. Validate numerical columns exist
        4. Detect dataset drift
        5. Save valid/invalid data
        6. Return artifact
        """
        try:
            logger.info("Starting Data Validation component")

            train_file_path = (
                self.data_ingestion_artifact.trained_file_path
            )
            test_file_path = (
                self.data_ingestion_artifact.test_file_path
            )

            # Read data
            train_df = DataValidation.read_data(train_file_path)
            test_df  = DataValidation.read_data(test_file_path)
            logger.info(
                f"Train shape: {train_df.shape} | "
                f"Test shape: {test_df.shape}"
            )

            # ── Validate number of columns ─────────────────────
            train_col_status = self.validate_number_of_columns(
                train_df
            )
            test_col_status  = self.validate_number_of_columns(
                test_df
            )
            logger.info(
                f"Column validation  "
                f"Train: {train_col_status} | "
                f"Test: {test_col_status}"
            )

            # ── Validate numerical columns ─────────────────────
            train_num_status = self.is_numerical_column_exist(
                train_df
            )
            test_num_status  = self.is_numerical_column_exist(
                test_df
            )
            logger.info(
                f"Numerical column validation  "
                f"Train: {train_num_status} | "
                f"Test: {test_num_status}"
            )

            # ── Overall validation status ──────────────────────
            validation_status = (
                train_col_status and
                test_col_status  and
                train_num_status and
                test_num_status
            )

            # ── Save valid/invalid data ────────────────────────
            os.makedirs(
                os.path.dirname(
                    self.data_validation_config.valid_train_file_path
                ),
                exist_ok=True
            )
            os.makedirs(
                os.path.dirname(
                    self.data_validation_config.invalid_train_file_path
                ),
                exist_ok=True
            )

            if validation_status:
                # Save valid data
                train_df.to_csv(
                    self.data_validation_config.valid_train_file_path,
                    index=False
                )
                test_df.to_csv(
                    self.data_validation_config.valid_test_file_path,
                    index=False
                )
                logger.info("Valid data saved successfully")
            else:
                # Save invalid data for inspection
                train_df.to_csv(
                    self.data_validation_config.invalid_train_file_path,
                    index=False
                )
                test_df.to_csv(
                    self.data_validation_config.invalid_test_file_path,
                    index=False
                )
                logger.warning(
                    "Validation failed — data saved to invalid dir"
                )

            # ── Detect dataset drift ───────────────────────────
            drift_detected = False
            if validation_status:
                drift_detected = self.detect_dataset_drift(
                    base_df=train_df,
                    current_df=test_df
                )
                logger.info(
                    f"Dataset drift detected: {drift_detected}"
                )

            # ── Return artifact ────────────────────────────────
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=(
                    self.data_validation_config.valid_train_file_path
                ),
                valid_test_file_path=(
                    self.data_validation_config.valid_test_file_path
                ),
                invalid_train_file_path=(
                    self.data_validation_config.invalid_train_file_path
                ),
                invalid_test_file_path=(
                    self.data_validation_config.invalid_test_file_path
                ),
                drift_report_file_path=(
                    self.data_validation_config.drift_report_file_path
                )
            )

            logger.info(
                f"Data Validation complete | "
                f"Status: {validation_status} | "
                f"Drift: {drift_detected}"
            )
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)