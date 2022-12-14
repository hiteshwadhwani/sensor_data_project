# we can validate the number of columns, missing values threshold, name of columns, etc
# refer to test/validaiton.ipynb for further info
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity, artifact_entity
from scipy.stats import ks_2samp
from sensor import utils
import os
import sys
from typing import Optional
import pandas as pd
import numpy as np
from sensor.config import TARGET_COLUMN


class Datavalidation:
    def __init__(self, data_validation_config:config_entity.DataValidationConfig, data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f'{">>" * 20} DATA VALIDATION {"<<" * 20}')
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise SensorException(e, sys)

    def is_required_columns_exist(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str)->bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns

            missing_columns = []

            for base_column in base_columns:
                if base_column not in current_columns:
                    missing_columns.append(base_column)

            if len(missing_columns) > 0:
                self.validation_error[report_key_name] = missing_columns
                return False
            return True
        except Exception as e:
            raise SensorException(e, sys)

    def drop_missing_columns(self, df:pd.DataFrame, report_key_name:str)->Optional[pd.DataFrame]:
        """
        This function drops columns from dataframe which have missing values greater than 30% by default

        df: accepts a pandas dataframe
        threshold: percentage criteria for dropping a dataframe column
        =================================================================================
        returns pandas dataframe if atleast a single column is available
        """

        try:
            threshold = self.data_validation_config.missing_threshold

            # Finding percentage of null values in each columns
            missing_percentage = df.isnull().sum().div(df.shape[0])

            # finding columns with null_percentage > threshold
            missing_percentage = missing_percentage[missing_percentage.values > threshold]

            missing_percentage_names = missing_percentage.index

            self.validation_error['report_key_name'] = list(missing_percentage_names)

            # Dropping columns
            df.drop(missing_percentage_names, axis=1, inplace=True)

            if len(df.columns) == 0:
                return None
            return df

        except Exception as e:
            raise SensorException(e, sys)
        
    def data_drift(self,base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str):
        try:
            drift_report = dict()

            base_columns = base_df.columns
            current_columns = current_df.columns

            for base_column in base_columns:
                base_data, current_data = base_df[base_column], current_df[base_column]

                #Null hypothesis = base_data and current_data have same distribution

                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue > 0.05:
                    # We are accepting Null hypothesis
                    drift_report[base_column] = {
                        'p_value':float(same_distribution.pvalue),
                        'same_distribution':True
                    }
                    # same distribution
                else:
                    drift_report[base_column] = {
                        'p_value':float(same_distribution.pvalue),
                        'same_distribution':False
                    }
                    # different distribution
            self.validation_error[report_key_name] = drift_report
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        try:
            logging.info("importing Base Dataset for validation")
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace({'na':np.NAN}, inplace=True)

            base_df = self.drop_missing_columns(df=base_df, report_key_name='missing_values_within_base_dataset')

            logging.info("Importing train and test dataframe for validation")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = self.drop_missing_columns(df=train_df, report_key_name='missing_values_within_train_dataaset')
            test_df = self.drop_missing_columns(df=test_df, report_key_name='missing_values_within_test_dataset')

            logging.info(f"convert columns type to float excluding target column : {TARGET_COLUMN}")
            exclude_columns= [TARGET_COLUMN]
            base_df = utils.convert_columns_float(df=base_df, exclude_columns=exclude_columns)
            train_df = utils.convert_columns_float(df=train_df, exclude_columns=exclude_columns)
            test_df = utils.convert_columns_float(df=test_df, exclude_columns=exclude_columns)

            logging.info("Checking if base_df and current_df has same number of columns")
            train_df_column_status = self.is_required_columns_exist(base_df=base_df, current_df=train_df, report_key_name='missing_columns_within_train_dataset')
            test_df_column_status = self.is_required_columns_exist(base_df=base_df, current_df=test_df, report_key_name='missing_columns_within_test_dataset')

            logging.info("Checking data drift within train and test dataset")
            if test_df_column_status:
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name='data_drift_within_test_dataset')
            
            if train_df_column_status:
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name='data_drift_within_train_dataset')
            
            logging.info("Writing report into yaml file")
            utils.create_yaml_file_from_dict(file_path=self.data_validation_config.report_file_path, data=self.validation_error)

            logging.info("Ceating artifact of data validation")
            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)

            logging.info(f"data_validation_artifact : {data_validation_artifact}")

            # Returning data_validation_artifact
            return data_validation_artifact
        except Exception as e:
            raise SensorException(e, sys)



