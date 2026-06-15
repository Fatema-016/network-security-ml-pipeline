import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from urllib.parse import quote_plus
from dotenv import load_dotenv

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.constants.training_pipeline import (
    DATABASE_NAME,
    COLLECTION_NAME
)

load_dotenv()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def _get_mongo_client(self) -> MongoClient:
        """Create and return authenticated MongoDB client."""
        try:
            import certifi
            username = quote_plus(os.getenv("MONGO_USERNAME"))
            password = quote_plus(os.getenv("MONGO_PASSWORD"))
            cluster  = os.getenv("MONGO_CLUSTER")
            app_name = os.getenv("MONGO_APP_NAME")

            uri = (
                f"mongodb+srv://{username}:{password}"
                f"@{cluster}/?appName={app_name}"
            )
            client = MongoClient(
                uri,
                server_api=ServerApi("1"),
                tlsCAFile=certifi.where()  
            )
            return client
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Pull all records from MongoDB Atlas collection
        and return as a pandas DataFrame.
        Excludes the auto-generated MongoDB _id field.
        """
        try:
            logger.info(
                f"Connecting to MongoDB — "
                f"DB: {DATABASE_NAME} | "
                f"Collection: {self.data_ingestion_config.collection_name}"
            )
            client     = self._get_mongo_client()
            collection = client[DATABASE_NAME][
                self.data_ingestion_config.collection_name
            ]

            
            cursor = collection.find({}, {"_id": 0})
            df     = pd.DataFrame(list(cursor))

            logger.info(
                f"Exported {len(df):,} records from MongoDB | "
                f"Shape: {df.shape}"
            )

            # Replace any remaining NA placeholders
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(
        self, dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Save the raw DataFrame pulled from MongoDB to the
        feature store as a CSV file.
        This is the raw data before any transformation.
        """
        try:
            feature_store_file_path = (
                self.data_ingestion_config.feature_store_file_path
            )
            
            os.makedirs(
                os.path.dirname(feature_store_file_path),
                exist_ok=True
            )
            dataframe.to_csv(
                feature_store_file_path,
                index=False,
                header=True
            )
            logger.info(
                f"Raw data saved to feature store: "
                f"{feature_store_file_path}"
            )
            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(
        self, dataframe: pd.DataFrame
    ) -> DataIngestionArtifact:
        """
        Split DataFrame into train and test sets.
        Uses stratified split to preserve BENIGN/ATTACK ratio
        in both sets — critical for imbalanced CICIDS 2017 data.
        """
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42,
                stratify=dataframe["Label"]  # preserve attack ratio
            )

            logger.info(
                f"Stratified train/test split complete | "
                f"Train: {len(train_set):,} | "
                f"Test: {len(test_set):,}"
            )
            logger.info(
                f"Train attack ratio: "
                f"{train_set['Label'].mean()*100:.1f}%"
            )
            logger.info(
                f"Test attack ratio: "
                f"{test_set['Label'].mean()*100:.1f}%"
            )

            # Save train and test CSVs
            os.makedirs(
                os.path.dirname(
                    self.data_ingestion_config.training_file_path
                ),
                exist_ok=True
            )
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )
            logger.info(
                f"Train data saved: "
                f"{self.data_ingestion_config.training_file_path}"
            )
            logger.info(
                f"Test data saved: "
                f"{self.data_ingestion_config.testing_file_path}"
            )

            # Return artifact with paths
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Main method — orchestrates the full data ingestion flow:
        1. Pull from MongoDB Atlas
        2. Save to feature store
        3. Stratified train/test split
        4. Return artifact with file paths
        """
        try:
            logger.info("Starting Data Ingestion component")

            # Step 1 — Pull from MongoDB
            dataframe = self.export_collection_as_dataframe()

            # Step 2 — Save to feature store
            dataframe = self.export_data_into_feature_store(dataframe)

            # Step 3 — Stratified split
            data_ingestion_artifact = self.split_data_as_train_test(
                dataframe
            )

            logger.info(
                f"Data Ingestion complete | "
                f"Artifact: {data_ingestion_artifact}"
            )
            return data_ingestion_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)