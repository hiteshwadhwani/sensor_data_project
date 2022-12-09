from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity, artifact_entity
from sensor.predictor import ModelResolver
import sys
import os

class ModelEvaluation:
    def __init__(self, model_evaluation_config:config_entity.ModelEvaluationConfig,
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                data_tranformation_artifact:artifact_entity.DataTransformationArtifact,
                model_trainer_artifact:artifact_entity.ModelTrainerArtifact):

                self.model_evaluation_config = model_evaluation_config
                self.data_ingestion_artifact = data_ingestion_artifact
                self.data_tranformation_artifact = data_tranformation_artifact
                self.model_trainer_artifact = model_trainer_artifact
                self.model_resolver = ModelResolver()

    def initiate_model_evaluation(self):
        try:
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_accuracy=None)
                return model_evaluation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    
        