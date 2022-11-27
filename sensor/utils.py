import pandas as pd
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.config import client
import os,sys


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
