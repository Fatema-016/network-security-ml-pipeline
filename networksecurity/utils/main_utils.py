import os
import sys
import yaml
import numpy as np
import pickle
from networksecurity.logging.logger import logger
from networksecurity.exception.exception import NetworkSecurityException


def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.
    Used for reading schema.yaml and model.yaml
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(
    file_path: str,
    content: object,
    replace: bool = False
) -> None:
    """
    Write a dictionary to a YAML file.
    Used for saving drift reports and evaluation reports.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Save a numpy array to a .npy file.
    Used for saving transformed train/test arrays.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logger.info(f"Numpy array saved at: {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load a numpy array from a .npy file.
    Used for loading transformed train/test arrays in model trainer.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj, allow_pickle=True)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    """
    Save any Python object as a pickle file using joblib compression.
    Used for saving preprocessor.pkl, model.pkl, shap_explainer.pkl
    compress=3 reduces file size  — important for t2.micro RAM
    """
    try:
        import joblib
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path, compress=3)
        logger.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load a pickle file saved with joblib.
    Used for loading preprocessor, model, and SHAP explainer.
    """
    try:
        import joblib
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")
        return joblib.load(file_path)
    except Exception as e:
        raise NetworkSecurityException(e, sys)