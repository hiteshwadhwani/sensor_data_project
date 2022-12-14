from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity, artifact_entity
from sensor import utils
import os
import sys
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sensor.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder

# Over-sampling using SMOTE and cleaning using Tomek links. Combine over- and under-sampling using SMOTE and Tomek links.
from imblearn.combine import SMOTETomek


class DataTranformation:
    def __init__(self, data_tranformation_config:config_entity.DataTransformationConfig, data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'>>' * 20} DATA TRANSFORMATION {'<<' * 20}")
            self.data_tranformation_config = data_tranformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_transformation_object(cls):
        try:
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler = RobustScaler()

            pipeline = Pipeline(steps=[('Imputer', simple_imputer), ('Robust_scaler', robust_scaler)]) 
            return pipeline

        except Exception as e:
            raise SensorException(e, sys)
    
    def initiate_data_transformation(self):
        # we will apply robustScaler (for scaling values -> why robust ? -> becasue of outliers), simpleimputer (to replace null values with mean)
        # we will balance our data with SMOTE
        
        try:
            logging.info("Read test and train file")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info("Selecting input feature for train and test dataframe")
            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            logging.info("Selecting output feature for train and test dataframe")
            logging.info(f"TARGET COLUMN : {TARGET_COLUMN}")
            output_feature_train_df = train_df[TARGET_COLUMN]
            output_feature_test_df = test_df[TARGET_COLUMN]

            label_encoder = LabelEncoder()
            label_encoder.fit(output_feature_train_df)

            logging.info("Label Encoding output feature")
            output_feature_train_arr = label_encoder.transform(output_feature_train_df)
            output_feature_test_arr = label_encoder.transform(output_feature_test_df)

            logging.info("Making Transformation pipeline")
            tranformation_pipeline = DataTranformation.get_transformation_object()
            tranformation_pipeline.fit(input_feature_train_df)

            logging.info("Tranforming input features")
            input_feature_train_arr = tranformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = tranformation_pipeline.transform(input_feature_test_df)

            logging.info("appling SMOTE oversampling")
            smt = SMOTETomek(random_state=42)
            input_feature_train_arr, output_feature_train_arr = smt.fit_resample(input_feature_train_arr, output_feature_train_arr)
            input_feature_test_arr, output_feature_test_arr = smt.fit_resample(input_feature_test_arr, output_feature_test_arr)


            logging.info("merging train and test np.array")
            train_arr = np.c_[input_feature_train_arr, output_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, output_feature_test_arr]


            logging.info("saving final train and test array")
            utils.save_numpy_array(file_path=self.data_tranformation_config.transformed_train_path , array=train_arr)
            utils.save_numpy_array(file_path=self.data_tranformation_config.transformed_test_path , array=train_arr)

            utils.save_object(file_path=self.data_tranformation_config.tranformed_obj_path , obj=tranformation_pipeline)
            utils.save_object(file_path=self.data_tranformation_config.target_encoder_path , obj=label_encoder)

            data_tranformation_artifact = artifact_entity.DataTransformationArtifact(
                                                                                    tranform_object_path=self.data_tranformation_config.tranformed_obj_path, 
                                                                                    tranform_train_path=self.data_tranformation_config.transformed_train_path, 
                                                                                    transform_test_path=self.data_tranformation_config.transformed_test_path, 
                                                                                    target_encoder_path=self.data_tranformation_config.target_encoder_path)
            logging.info(f"Data_transformation_artifact : {data_tranformation_artifact}")
            return data_tranformation_artifact
        except Exception as e:
            raise SensorException(e, sys)















