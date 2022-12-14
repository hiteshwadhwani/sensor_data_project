import os,sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity, artifact_entity
from sensor import utils
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:

    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>' * 20} DATA INGESTION {'<<' * 20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys)
    

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info("Importing data from database")
            df:pd.DataFrame = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Dataset imported of shape {df.shape}")



            logging.info(f"replacing na values {df.isna().sum()} to np.NAN dtype")     
            df.replace(to_replace='na', value=np.NAN, inplace=True)

            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            
            os.makedirs(feature_store_dir, exist_ok=True)

            logging.info("saving dataframe")
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path, index=False, header=True)

            logging.info("splitting dataframe in train and test")
            train_df, test_df= train_test_split(df, test_size=self.data_ingestion_config.test_size, random_state=42)
            logging.info(f"train : {train_df.shape}, test : {test_df.shape}")

            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)

            logging.info("saving train and test df")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path, 
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path=self.data_ingestion_config.test_file_path)
            logging.info(f"data_ingestion_artifact : {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)

    