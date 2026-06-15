from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """
    Output of Data Ingestion component.
    Paths to train and test CSV files.
    """
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    """
    Output of Data Validation component.
    Validation status and paths to valid/invalid data and drift report.
    """
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    """
    Output of Data Transformation component.
    Paths to transformed numpy arrays and saved preprocessor.
    """
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricArtifact:
    """
    Stores classification metrics for one model.
    Used by Model Trainer and Model Evaluation components.
    """
    f1_score: float
    precision_score: float
    recall_score: float
    roc_auc_score: float


@dataclass
class ModelTrainerArtifact:
    """
    Output of Model Trainer component.
    Path to best trained model and its metrics.
    """
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelEvaluationArtifact:
    """
    Output of Model Evaluation component.
    Whether model is accepted and performance comparison.
    """
    is_model_accepted: bool
    improved_accuracy: float
    best_model_path: str
    trained_model_path: str
    train_model_metric_artifact: ClassificationMetricArtifact
    best_model_metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelPusherArtifact:
    """
    Output of Model Pusher component.
    Paths where final model is saved locally and on S3.
    """
    saved_model_path: str
    model_pusher_dir_path: str