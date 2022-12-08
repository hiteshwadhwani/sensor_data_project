from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity, artifact_entity
from sensor import utils
import os
import sys
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

class ModelTrainer:
    def __init__(self, model_trainer_config:config_entity.ModelTrainerConfig, data_tranformation_artifact:artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_tranformation_artifact = data_tranformation_artifact
        except Exception as e:
            raise SensorException(e, sys)
    
    def train_model(self, X, y):
        try:
            xgb_classifier = XGBClassifier()
            xgb_classifier.fit(X, y)
            return xgb_classifier
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_training(self)->artifact_entity.ModelTrainerArtifact:
        try:
            train_arr = utils.load_numpy_array(self.data_tranformation_artifact.tranform_train_path)
            test_arr = utils.load_numpy_array(self.data_tranformation_artifact.transform_test_path)

            x_train, y_train = train_arr[:,:-1], train_arr[:, -1]
            x_test , y_test = test_arr[:, :-1], test_arr[:, -1]

            model = self.train_model(x_train, y_train)
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_train, yhat_train)

            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_test, yhat_test)

            # Checking underfitting
            if f1_train_score < self.model_trainer_config.expected_score:
                raise Exception(f"model is not good as it is able to give, expected accuracy {self.model_trainer_config.expected_score}")

            # Checking for overfitting
            if abs(f1_test_score - f1_train_score) > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"model is overfitted, difference between train and test score is {abs(f1_test_score - f1_train_score)}")
            

            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_trainer=self.model_trainer_config.model_path, 
                                                                            f1_train_score=f1_train_score, 
                                                                            f1_test_score=f1_test_score)

            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e, sys)
