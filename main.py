from sensor.utils import get_collection_as_dataframe
from sensor.config import client

df = get_collection_as_dataframe('aps', 'sensor')
print(df.head())


