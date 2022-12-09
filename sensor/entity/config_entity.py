import os,sys
from sensor.logger import logging
from sensor.exception import SensorException
from datetime import datetime

FILE_NAME='sensor.csv'
TEST_FILE_NAME='test.csv'
TRAIN_FILE_PATH='train.csv'
MODEL_FILE_NAME = "model.pkl"
TRANSFORM_FILE_NAME = "transform.pkl"
ENCODER_FILE_NAME = "taget_encoder.pkl"


class TrainingPipelineConfig:    
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m:%d:%Y__%H:%M:%S')}")
        except Exception as e:
            SensorException(e, sys)
    def to_dict(self,):
        try:
            return self.__dict__
        except Exception as e:
            raise SensorException(e, sys)

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.database_name = 'aps'
            self.collection_name = 'sensor'
            self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, 'data_ingestion')  # artifact/date_time/data_ingestion
            self.feature_store_file_path = os.path.join(self.data_ingestion_dir, 'feature_store', FILE_NAME) # artifact/date_time/data_ingestion/feature_store/sensor.csv
            self.test_file_path = os.path.join(self.data_ingestion_dir, 'dataset', TEST_FILE_NAME)           # artifact/date_time/data_ingestion/dataset/test.csv
            self.train_file_path = os.path.join(self.data_ingestion_dir, 'dataset', TRAIN_FILE_PATH)         # artifact/date_time/data_ingestion/dataset/train.csv
            self.test_size = 0.2
        except Exception as e:
            SensorException(e, sys)

    def to_dict(self,):
        try:
            return self.__dict__
        except Exception as e:
            raise SensorException(e, sys)


class DataValidationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_validaiton_dir = os.path.join(training_pipeline_config.artifact_dir, 'data_validation') # artifact/date_time/data_validation
            self.report_file_path = os.path.join(self.data_validaiton_dir, 'report.yaml')                   # artifact/date_time/data_validation/report.yaml
            self.missing_threshold:float = 0.7
            self.base_file_path = os.path.join("aps_failure_training_set1.csv")
        except Exception as e:
            raise SensorException(e, sys)
    
    def to_dict(self):
        try:
            return self.to_dict
        except Exception as e:
            raise SensorException(e, sys)


class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, 'data_transformation')
            self.tranformed_obj_path = os.path.join(self.data_transformation_dir,'transformer', TRANSFORM_FILE_NAME)
            self.transformed_train_path = os.path.join(self.data_transformation_dir, 'transformed', 'train.npz')
            self.transformed_test_path = os.path.join(self.data_transformation_dir, 'transformed', 'test.npz')
            self.target_encoder_path = os.path.join(self.data_transformation_dir, 'taget_encoder', ENCODER_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        try:
            self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, 'model_trainer')
            self.model_path = os.path.join(self.model_trainer_dir, 'model', MODEL_FILE_NAME)
            self.expected_score = 0.7
            self.overfitting_threshold = 0.1
        except Exception as e:
            raise SensorException(e, sys)



class ModelEvaluationConfig:
    def __init__(self):
        self.change_threshold = 0.01

        
class ModelPusherConfig:...
