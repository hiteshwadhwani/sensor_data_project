import os,sys
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity import config_entity, artifact_entity
from sensor.predictor import ModelResolver
from sensor import utils

class ModelPusher:
    def __init__(self, data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                        model_trainer_artifact:artifact_entity.ModelTrainerArtifact,
                        model_pusher_config:config_entity.ModelPusherConfig):
        try:
            logging.info(f"{'>>' * 20} MODEL PUSHER {'<<' * 20}")
            self.model_pusher_config = model_pusher_config
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_resolver = ModelResolver(model_registery=self.model_pusher_config.saved_model_dir)
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_pusher(self):
        try:

            logging.info("importing all the latest model, encoder and transformer")
            model = utils.load_object(file_path=self.model_trainer_artifact.model_trainer)
            encoder = utils.load_object(file_path=self.data_transformation_artifact.target_encoder_path)
            transformer = utils.load_object(file_path=self.data_transformation_artifact.tranform_object_path)


            logging.info("Saving latest  model, encoder and transformer to artifact folder")
            utils.save_object(file_path=self.model_pusher_config.model_pusher_model_path, obj=model)
            utils.save_object(file_path=self.model_pusher_config.model_pusher_encoder_path, obj=encoder)
            utils.save_object(file_path=self.model_pusher_config.model_pusher_transformer_path, obj=transformer)

            logging.info("Getting transformer_path , encoder_path  and model_path")
            transformer_path = self.model_resolver.get_latest_save_tranformer_path()
            encoder_path = self.model_resolver.get_latest_save_encoder_path()
            model_path = self.model_resolver.get_latest_save_model_path()

            logging.info("saving latest model, encoder and transformer to saved_models dir")
            utils.save_object(file_path=model_path, obj=model)
            utils.save_object(file_path=encoder_path, obj=encoder)
            utils.save_object(file_path=transformer_path, obj=transformer)

            model_pusher_artifact = artifact_entity.ModelPusherArtifact(model_pusher_dir=self.model_pusher_config.model_pusher_dir, 
            saved_model_dir=self.model_pusher_config.saved_model_dir)
            logging.info(f"Model_pusher_artifact : {model_pusher_artifact}")
            
            return model_pusher_artifact

        except Exception as e:
            raise SensorException(e, sys)