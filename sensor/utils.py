import pandas as pd
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.config import client
import os,sys
import yaml
import numpy as np
import dill

def get_collection_as_dataframe(database_name, collection_name)->pd.DataFrame:
    """
    Description : This function return collection as dataframe
    ==================================================================
    params:
    database_name : name of database
    collection_name : name of collection
    ==================================================================
    returns : pandas dataframe of collection
    """
    try:
        logging.info(f"Reading data from database {database_name} and collection {collection_name}")
        df = pd.DataFrame(list(client[database_name][collection_name].find()))
        logging.info(f"Found columns {df.columns}")
        if '_id' in df.columns:
            df.drop('_id', axis=1, inplace=True)
        logging.info(f"Rows and columns of data {df.shape}")
        return df
    except Exception as e:
        logging.error(e)
        raise SensorException(e, sys)

def create_yaml_file_from_dict(file_path, data:dict):
    try:
        file_dir = os.path.dirname(file_path)

        os.makedirs(file_dir, exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(data, file)
    except Exception as e:
        raise SensorException(e, sys)

def convert_columns_float(df:pd.DataFrame, exclude_columns:list):
    try:
        for column in df.columns:
            if column not in exclude_columns:
                df[column] = df[column].astype('float')
        return df
    except Exception as e:
        raise SensorException(e, sys)

def save_object(file_path:str, obj:object):
    """
    Description : This function save object to file
    ==================================================================
    params:
    file_path : path of file
    obj : object to save
    """
    try:
        logging.info("Entered the save_obj method of utils")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info("exited from the save_obj method of utils")
    except Exception as e:
        raise SensorException(e, sys)
    
def load_object(file_path:str)->object:
    """
    Description : This function return object from file
    ==================================================================
    params:
    file_path : path of file
    ==================================================================
    returns : object
    """
    try:
        if not os.path.exists(file_path):
            raise Exception("file path {file_path} does not exists")
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise SensorException(e, sys)

def save_numpy_array(file_path:str, array:np.array)->str:
    """
    Description : This function save numpy array to file
    ==================================================================
    params:
    file_path : file path
    array : numpy array
    """
    try:
        logging.info("Entered the save_numpy_array method of utils")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file, array)
    except Exception as e:
        raise SensorException(e, sys)

def load_numpy_array(file_path:str)->np.array:
    """
      load numpy array from file path
      ===============================
      params:
      file_path : file path
    """
    try:
        if not os.path.exists(file_path):
            raise Exception("file path {file_path} does not exists")
        with open(file_path, 'rb') as file:
            return np.load(file)
    except  Exception as e:
        raise SensorException(e, sys)
    

