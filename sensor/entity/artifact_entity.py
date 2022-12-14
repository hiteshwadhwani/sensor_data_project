from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str
    train_file_path:str
    test_file_path:str

@dataclass
class DataValidationArtifact:
    report_file_path:str

@dataclass
class DataTransformationArtifact:
    tranform_object_path:str
    tranform_train_path:str
    transform_test_path:str
    target_encoder_path:str

@dataclass
class ModelTrainerArtifact:
    model_trainer:str
    f1_train_score:float
    f1_test_score:float

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    improved_accuracy:float

@dataclass
class ModelPusherArtifact:
    model_pusher_dir:str
    saved_model_dir:str