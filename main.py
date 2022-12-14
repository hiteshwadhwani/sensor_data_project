from sensor.pipeline import training_pipeline
from sensor.pipeline.batch_prediction import start_batch_prediction

INPUT_FILE = 'aps_failure_training_set1.csv'

if __name__=="__main__":
     print(start_batch_prediction(input_file_path=INPUT_FILE))
     # training_pipeline.start_training_pipeline()