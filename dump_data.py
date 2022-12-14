import pymongo
import pandas as pd
import certifi
import json

import os
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')

# Create connection
client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
print(client)

DATA_FILE_PATH = '/config/workspace/aps_failure_training_set1.csv'
DATABASE_NAME = 'aps'
COLLECTION_NAME = 'sensor'

db = client[DATABASE_NAME] #database
col = db[COLLECTION_NAME]  #collection

if __name__ == '__main__':
    df = pd.read_csv(DATA_FILE_PATH)
    print("Rows and column", df.shape)

    df.reset_index(drop=True, inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    col.insert_many(json_record)


