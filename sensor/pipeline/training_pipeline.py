from sensor.logger import logging
from sensor.exception import SensorException
from sensor.utils import get_collection_as_dataframe
import sys,os
from sensor.entity import config_entity
from sensor.components.data_ingestion import DataIngestion
from sensor.components.data_validation import Datavalidation
from sensor.components.data_transformation import DataTranformation
from sensor.components.model_trainer import ModelTrainer
from sensor.components.model_evaluation import ModelEvaluation
from sensor.components.model_pusher import ModelPusher

def start_training_pipeline():
    try:
          training_pipeline_config = config_entity.TrainingPipelineConfig()
          data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
          print(data_ingestion_config.to_dict())
          data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
          data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

          # Data validation
          data_validation_config = config_entity.DataValidationConfig(training_pipeline_config= training_pipeline_config)
          data_validation = Datavalidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
          data_validation_artifact = data_validation.initiate_data_validation()
          print(data_validation_artifact)

          # Data transformation
          data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
          data_transformation = DataTranformation(data_tranformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact)
          data_tranformation_artifact = data_transformation.initiate_data_transformation()
          print(data_tranformation_artifact)

          # model training
          model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
          model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_tranformation_artifact=data_tranformation_artifact)
          model_trainer_artifact = model_trainer.initiate_model_training()
          print(model_trainer_artifact)

          # model evaluation 
          model_evaluation_config = config_entity.ModelEvaluationConfig()
          model_evaluation = ModelEvaluation(model_evaluation_config=model_evaluation_config, 
                                             data_ingestion_artifact=data_ingestion_artifact, 
                                             data_tranformation_artifact=data_tranformation_artifact, 
                                             model_trainer_artifact=model_trainer_artifact)

          model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
          print(model_evaluation_artifact)

          # model pusher
          model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config=training_pipeline_config)
          model_pusher = ModelPusher(data_transformation_artifact=data_tranformation_artifact, 
                                        model_trainer_artifact=model_trainer_artifact, 
                                        model_pusher_config=model_pusher_config)

          model_pusher_artifact = model_pusher.initiate_model_pusher()
          print(model_pusher_artifact)
    except Exception as e:
        raise SensorException(e, sys)