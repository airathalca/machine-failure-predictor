import pandas as pd
import numpy as np
import sys
from typing import Optional

from machine_failure.configuration.mongodb_connection import MongoDBClient
from machine_failure.exception.custom_exception import CustomException
from machine_failure.logger.custom_logging import logging
from machine_failure.utils.main_utils import read_yaml_file

params = read_yaml_file("config/param.yaml")

class MongoDataset:
  """
  Class Name  : Dataset
  Description : This class is used to export collection as dataframe
  Output      : pd.DataFrame
  On Failure  : Raise Exception
  """
  def __init__(self):
    try:
      self.mongo_client = MongoDBClient(database_name=params["database"]["name"])
    except Exception as e:
      logging.error(f"Error in cls MongoDataset method __init__: {e}")
      raise CustomException(e,sys)
      
  def export_collection_to_dataframe(self, collection_name:str, database_name:Optional[str] = None)-> pd.DataFrame:
    logging.info(f"Entered the export_collection_to_dataframe method of MongoDataset class")
    try:
      """
      export entire collection as dataframe:
      return pd.DataFrame
      """
      if database_name is None:
        collection = self.mongo_client.database[collection_name]
      else:
        collection = self.mongo_client[database_name][collection_name]
      df = pd.DataFrame(list(collection.find()))
      if "_id" in df.columns.to_list():
        df = df.drop(columns=["_id"], axis=1)
      df.replace({"na":np.nan},inplace=True)
      logging.info(f"Exiting the export_collection_to_dataframe method of MongoDataset class")
      return df
    except Exception as e:
      logging.error(f"Error in class MongoDataset method export_collection_to_dataframe: {e}")
      raise CustomException(e,sys)