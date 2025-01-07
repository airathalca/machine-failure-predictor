import os
import sys
import pandas as pd

from machine_failure.exception.custom_exception import CustomException
from machine_failure.logger.custom_logging import logging
from machine_failure.entity.config_entity import DataIngestionConfig
from machine_failure.entity.artifact_entity import DataIngestionArtifact
from machine_failure.configuration.mongodb_data_access import MongoDataset
from machine_failure.utils.main_utils import write_csv

from sklearn.model_selection import train_test_split


class DataIngestion:
  def __init__(self):
    self.config = DataIngestionConfig()

  def export_raw_data(self) -> pd.DataFrame:
    logging.info("Entered the export_raw_data method")
    try:
      logging.info('Exporting data from MongoDB')
      mongo_data = MongoDataset()
      df = mongo_data.export_collection_to_dataframe(self.config.collection_name)
      logging.info(f'Shape of the data: {df.shape}')

      dir_path = os.path.dirname(self.config.raw_file_path)
      os.makedirs(dir_path,exist_ok=True)
      logging.info(f'Saving raw data')
      write_csv(self.config.raw_file_path, df)
      logging.info('Raw data saved successfully')

      return df
    except Exception as e:
      logging.info("Error in exporting raw data: {e}")
      raise CustomException(e, sys)
    
  def split_data(self, df: pd.DataFrame) -> None:
    logging.info("Entered the split_data method")
    try:
      logging.info('Splitting data into train and test')
      train_set, test_set = train_test_split(df, test_size=self.config.train_test_split_ratio)
      logging.info(f'Data split successfully. Shape of train data: {train_set.shape} and test data: {test_set.shape}')

      dir_path = os.path.dirname(self.config.training_file_path)
      os.makedirs(dir_path,exist_ok=True)
      logging.info(f'Saving train data and test data')
      write_csv(self.config.training_file_path, train_set)
      write_csv(self.config.testing_file_path, test_set)
      logging.info('Train and Test data saved successfully')

    except Exception as e:
      logging.info("Error in splitting data: {e}")
      raise CustomException(e, sys)
    
  def read_data(self) -> DataIngestionArtifact:
    logging.info("Entered the read_data method")
    try:
      df = self.export_raw_data()
      logging.info('Data exported successfully')

      self.split_data(df)
      logging.info('Data splited successfully')

      data_ingestion_artifact = DataIngestionArtifact(train_file_path=self.config.training_file_path, 
                                                      test_file_path=self.config.testing_file_path)

      return data_ingestion_artifact
    except Exception as e:
      logging.info("Data ingestion failed")
      raise CustomException(e, sys)
    
if __name__ == "__main__":
  data_ingestion = DataIngestion()
  data_ingestion.read_data()