from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity, artifact_entity
from sensor.predictor import ModelResolver
import sys
import os
from sensor import utils
import pandas as pd
from sensor.config import TARGET_COLUMN
from sklearn.metrics import f1_score

class ModelEvaluation:
    def __init__(self, model_evaluation_config:config_entity.ModelEvaluationConfig,
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                data_tranformation_artifact:artifact_entity.DataTransformationArtifact,
                model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
                logging.info(f"{'>>' * 20} DATA EVALUATION {'<<' * 20}")
                self.model_evaluation_config = model_evaluation_config
                self.data_ingestion_artifact = data_ingestion_artifact
                self.data_tranformation_artifact = data_tranformation_artifact
                self.model_trainer_artifact = model_trainer_artifact
                self.model_resolver = ModelResolver()

    def initiate_model_evaluation(self):
        try:
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                logging.info(f"------No saved models available--------")
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, improved_accuracy=None)
                logging.info(f"Model_evaluation_artifact : {model_evaluation_artifact}")
                return model_evaluation_artifact

            logging.info("Importing previously trained model, tranformer and encoder")
            transformer = utils.load_object(self.model_resolver.get_latest_tranformer_path())
            encoder = utils.load_object(self.model_resolver.get_latest_encoder_path())
            model = utils.load_object(self.model_resolver.get_latest_model_path())

            logging.info("Importting currently trained model, tranformer and encoder")
            current_tranformer = utils.load_object(self.data_tranformation_artifact.tranform_object_path)
            current_encoder = utils.load_object(self.data_tranformation_artifact.target_encoder_path)
            current_model = utils.load_object(self.model_trainer_artifact.model_trainer)

            logging.info("Importing test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]
            

            logging.info("calculating accuracy using previous trained model")
            input_feature_name = list(transformer.feature_names_in_)
            input_array = transformer.transform(test_df[input_feature_name])
            y_true = encoder.transform(target_df)
            y_pred = model.predict(input_array)
            previous_model_score = f1_score(y_pred, y_true)
            logging.info("f1_score using previous trained model", previous_model_score)

            # accuracy using currently trained model
            input_feature_name = list(current_tranformer.feature_names_in_)
            input_array = current_tranformer.transform(test_df[input_feature_name])
            y_true = current_encoder.transform(target_df)
            y_pred = current_model.predict(input_array)
            current_model_score = f1_score(y_pred, y_true)
            logging.info("f1_score using current trained model", current_model_score)

            if current_model_score <= previous_model_score:
                raise Exception(f"current model accurracy {current_model_score} is not greater than previous model accuracy {previous_model_score}")

            model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, 
                                                                                improved_accuracy=current_model_score - previous_model_score)
            
            logging.info(f"Model_evaluation_artifact : {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys)

    
        