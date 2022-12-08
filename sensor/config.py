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
    mognoDB_pass = os.getenv("mongoDB_pass")

env_var = EnvironmentVariables()

# Create connection
client = pymongo.MongoClient(
    f"mongodb+srv://hiteshwadhwani1403:{env_var.mognoDB_pass}@ineuron.xskip.mongodb.net/?retryWrites=true&w=majority",
    tlsCAFile=certifi.where())

TARGET_COLUMN = 'class'
