import pymongo
import pandas as pd
import certifi
import json
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class EnvironmentVariables:
    MONGO_DB_URL = os.getenv('MONGO_DB_URL')
    
env_var = EnvironmentVariables()

# Create connection
client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
TARGET_COLUMN = 'class'
