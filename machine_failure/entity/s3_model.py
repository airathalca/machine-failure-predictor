import sys
import pandas as pd

from machine_failure.configuration.s3_storage import S3Storage
from machine_failure.exception.custom_exception import CustomException
from machine_failure.logger.custom_logging import logging
from machine_failure.entity.model import MachineFailureModel

class MachineFailureS3Model:
  def __init__(self, bucket_name:str, model_path:str):
    self.bucket_name = bucket_name
    self.s3 = S3Storage()
    self.model_path = model_path
    self.loaded_model: MachineFailureModel = None

  def is_model_present(self, model_path):
    try:
      return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
    except CustomException as e:
      logging.info(f"Error in cls MachineFailureS3Model method is_model_present: {e}")
      return False

  def load_model(self) -> MachineFailureModel:
    return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)

  def predict(self,df: pd.DataFrame):
    try:
      if self.loaded_model is None:
        self.loaded_model = self.load_model()
      return self.loaded_model.predict(df)
    except Exception as e:
      logging.error(f"Error in cls MachineFailureS3Model method predict: {e}")
      raise CustomException(e, sys)
    
  def predict_proba(self,df: pd.DataFrame):
    try:
      if self.loaded_model is None:
        self.loaded_model = self.load_model()
      return self.loaded_model.predict_proba(df)
    except Exception as e:
      logging.error(f"Error in cls MachineFailureS3Model method predict_proba: {e}")
      raise CustomException(e, sys)