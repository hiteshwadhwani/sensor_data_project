import os
import sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity.config_entity import ENCODER_FILE_NAME, TRANSFORM_FILE_NAME, MODEL_FILE_NAME
from glob import glob
from typing import Optional


class ModelResolver:
    def __init__(self, model_registery:str='saved_models',
                tranform_dir_name:str='transformer',
                target_encoder_dir_name:str='encoder',
                model_dir_name:str='model'):
        
        self.model_registery = model_registery
        os.makedirs(self.model_registery, exist_ok=True)
        self.tranform_dir_name= tranform_dir_name
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name
        
    def get_latest_dir_path(self)->Optional[str]:
        try:
            dir_names = os.listdir(self.model_registery)
            if len(dir_names) == 0:
                return None
            
            dir_names = list(map(int, dir_names))
            latest_dir_name = max(dir_names)
            return os.path.join(self.model_registery, f"{latest_dir_name}")

        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_tranformer_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("tranformer folder is not available")
            return os.path.join(latest_dir, self.tranform_dir_name, TRANSFORM_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_encoder_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("encoer folder is not available")
            return os.path.join(latest_dir, self.target_encoder_dir_name, ENCODER_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_model_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("model folder is not available")
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_save_dir_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.model_registery, f"{0}")
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registery, f"{latest_dir_num+1}")
        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_save_tranformer_path(self):
        try:
            return os.path.join(self.get_latest_save_dir_path(), self.tranform_dir_name, TRANSFORM_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_save_encoder_path(self):
        try:
            return os.path.join(self.get_latest_save_dir_path(), self.target_encoder_dir_name, ENCODER_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)

    def get_latest_save_model_path(self):
        try:
            return os.path.join(self.get_latest_save_dir_path(), self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)



class Predictor:
    def __init__(self, model_resolver:ModelResolver):
        self.model_resolver = model_resolver