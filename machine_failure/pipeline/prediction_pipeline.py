import os
import sys
import numpy as np
import pandas as pd

from machine_failure.entity.config_entity import ModelBucketConfig
from machine_failure.entity.s3_model import MachineFailureS3Model
from machine_failure.exception.custom_exception import CustomException

class MachineData:
  def __init__(self, product_id: int, type: str, air_temperature: float, 
               process_temperature: float, rotational_speed: int, torque: float, 
               tool_wear: int, TWF: int, HDF: int, PWF: int, OSF: int) -> None:
    try:
      self.product_id = product_id
      self.type = type
      self.air_temperature = air_temperature
      self.process_temperature = process_temperature
      self.rotational_speed = rotational_speed
      self.torque = torque
      self.tool_wear = tool_wear
      self.TWF = TWF
      self.HDF = HDF
      self.PWF = PWF
      self.OSF = OSF
    except Exception as e:
      raise CustomException(e, sys)

  def convert_to_pandas(self)-> pd.DataFrame:
    try:
      dict = self.convert_to_dict()
      return pd.DataFrame(dict)
    
    except Exception as e:
      raise CustomException(e, sys)

  def convert_to_dict(self):
    try:
      input_data = {
        'Product ID': [self.product_id],
        'Type': [self.type],
        'Air temperature [K]': [self.air_temperature],
        'Process temperature [K]': [self.process_temperature],
        'Rotational speed [rpm]': [self.rotational_speed],
        'Torque [Nm]': [self.torque],
        'Tool wear [min]': [self.tool_wear],
        'TWF': [self.TWF],
        'HDF': [self.HDF],
        'PWF': [self.PWF],
        'OSF': [self.OSF]
      }
      return input_data
    except Exception as e:
      raise CustomException(e, sys)

class MachineClassifier:
  def __init__(self) -> None:
    try:
        self.config = ModelBucketConfig()
        self.model =  MachineFailureS3Model(self.config.bucket_name, self.config.s3_model_key_path)
    except Exception as e:
        raise CustomException(e, sys)

  def predict(self, df: pd.DataFrame) -> str:
    try:
      result =  self.model.predict(df)
      return result
    except Exception as e:
      raise CustomException(e, sys)