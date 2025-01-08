import os
import sys
import pymongo
import certifi
from dotenv import load_dotenv

from machine_failure.exception.custom_exception import CustomException
from machine_failure.logger.custom_logging import logging
from machine_failure.constants import DATABASE_NAME

ca = certifi.where()
load_dotenv()

class MongoDBClient:
  """
  Class Name  : MongoDBClient
  Description : This method is used to connect to the mongodb database
  Output      : connection to mongodb database
  On Failure  : Raise Exception
  """
  client = None
  def __init__(self, database_name=DATABASE_NAME) -> None:
    logging.info("Entered the __init__ method of MongoDBClient class")
    try:
      if MongoDBClient.client is None:
        mongo_db_url = os.getenv("MONGODB_URL_KEY")
        if mongo_db_url is None:
          logging.error("Environment key: MONGODB_URL_KEY is not set.")
          raise Exception(f"Environment key: MONGODB_URL_KEY is not set.")
        MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
      self.client = MongoDBClient.client
      self.database = self.client[database_name]
      self.database_name = database_name
      logging.info("Exiting the __init__ method of MongoDBClient class. Connection established successfully")
    except Exception as e:
      logging.error(f"Error in cls MongoDBClient method __init__: {e}")
      raise CustomException(e,sys)
    
if __name__ == "__main__":
  try:
    mongo_client = MongoDBClient()
    print(mongo_client)
  except Exception as e:
    print(e)