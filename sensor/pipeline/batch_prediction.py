from sensor.exception import SensorException
import os,sys
from sensor.logger import logging
from sensor.predictor import ModelResolver
import pandas as pd
import numpy as np
from sensor.utils import load_object
from datetime import datetime

PREDICTION_DIR_NAME = 'prediction'
def start_batch_prediction(input_file_path:str):
    try:
        os.makedirs(PREDICTION_DIR_NAME, exist_ok=True)
        logging.info("Making object of ModelResolver")
        model_resolver = ModelResolver()
        logging.info(f"importing dataset from {input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace(to_replace='na', value=np.NAN, inplace=True)

        logging.info("importing transformer,  encoder and model")
        transformer = load_object(file_path=model_resolver.get_latest_tranformer_path())
        encoder = load_object(file_path=model_resolver.get_latest_encoder_path())
        model = load_object(file_path=model_resolver.get_latest_model_path())

        input_feature_names = transformer.feature_names_in_
        input_array = transformer.transform(df[input_feature_names])
        
        logging.info("Predicting target feature")
        prediction = model.predict(input_array)
        prediction = encoder.inverse_transform(prediction)

        df['prediction'] = prediction

        prediction_file_name = os.path.basename(input_file_path).replace('.csv', f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR_NAME , prediction_file_name)

        logging.info(f"saving prediction file on path : {prediction_file_path}")
        df.to_csv(prediction_file_path, index=False, header=True)
        return prediction_file_path
        
    except Exception as e:
        raise SensorException(e, sys)