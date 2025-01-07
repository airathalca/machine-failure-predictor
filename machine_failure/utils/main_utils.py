import pandas as pd
import numpy as np
import os
import sys
import dill
import yaml

from machine_failure.exception.custom_exception import CustomException
from machine_failure.logger.custom_logging import logging

def read_yaml_file(file_path: str) -> dict:
  """
  Read yaml file
  file_path: str location of file to read
  """
  logging.info("Entered the read_yaml_file method of utils")
  try:
    with open(file_path, "rb") as yaml_file:
      return yaml.safe_load(yaml_file)
    logging.info(f"Exited the read_yaml_file. {file_path} read successfully")
  except Exception as e:
    logging.error(f"Error in read_yaml_file: {e}")
    raise CustomException(e, sys)
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
  """
  Write content to yaml file
  file_path: str location of file to write
  content: object data to write
  replace: bool replace file if exists
  """
  logging.info("Entered the write_yaml_file method of utils")
  try:
    if replace:
      if os.path.exists(file_path):
        os.remove(file_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
      yaml.dump(content, file)
    logging.info(f"Exited the write_yaml_file. {file_path} written successfully")
  except Exception as e:
    logging.error(f"Error in write_yaml_file: {e}")
    raise CustomException(e, sys)
  
def save_object(file_path: str, obj: object) -> None:
  """
  Save object to file
  file_path: str location of file to save
  obj: object data to save
  """
  logging.info("Entered the save_object method of utils")
  try:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
      dill.dump(obj, file_obj)
    logging.info(f"Exited the save_object. {file_path} saved successfully")
  except Exception as e:
    logging.error(f"Error in save_object: {e}")
    raise CustomException(e, sys)
    
def load_object(file_path: str) -> object:
  """
  Load object from file
  file_path: str location of file to load
  """
  logging.info("Entered the load_object method of utils")
  try:
    with open(file_path, "rb") as file_obj:
      obj = dill.load(file_obj)
    logging.info(f"Exited the load_object. {file_path} loaded successfully")
    return obj
  except Exception as e:
    logging.error(f"Error in load_object: {e}")
    raise CustomException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array):
  """
  Save numpy array data to file
  file_path: str location of file to save
  array: np.array data to save
  """
  logging.info("Entered the save_numpy_array_data method of utils")
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as file_obj:
      np.save(file_obj, array)
    logging.info(f"Exited the save_numpy_array_data. {file_path} saved successfully")
  except Exception as e:
    logging.error(f"Error in save_numpy_array_data: {e}")
    raise CustomException(e, sys)
        
def load_numpy_array_data(file_path: str) -> np.array:
  """
  load numpy array data from file
  file_path: str location of file to load
  return: np.array data loaded
  """
  logging.info("Entered the load_numpy_array_data method of utils")
  try:
    with open(file_path, 'rb') as file_obj:
      return np.load(file_obj)
    logging.info(f"Exited the load_numpy_array_data. {file_path} loaded successfully")
  except Exception as e:
    logging.error(f"Error in load_numpy_array_data: {e}")
    raise CustomException(e, sys)
  
def read_csv(file_path: str) -> pd.DataFrame:
  """
  Read csv file
  file_path: str location of file to read
  return: pandas DataFrame
  """
  logging.info("Entered the read_csv method of utils")
  try:
    df = pd.read_csv(file_path)
    logging.info(f"Exited the read_csv. {file_path} read successfully")
    return df
  except Exception as e:
    logging.error(f"Error in read_csv: {e}")
    raise CustomException(e, sys)
  
def write_csv(file_path: str, df: pd.DataFrame) -> None:
  """
  Write csv file
  file_path: str location of file to write
  df: pandas DataFrame
  """
  logging.info("Entered the write_csv method of utils")
  try:
    df.to_csv(file_path, index=False, header=True)
    logging.info(f"Exited the write_csv. {file_path} written successfully")
  except Exception as e:
    logging.error(f"Error in write_csv: {e}")
    raise CustomException(e, sys)

def drop_columns(df: pd.DataFrame, cols: list)-> pd.DataFrame:
  """
  drop the columns form a pandas DataFrame
  df: pandas DataFrame
  cols: list of columns to be dropped
  """
  logging.info("Entered the drop_columns method of utils")
  try:
    df = df.drop(columns=cols, axis=1)
    logging.info(f"Exited the drop_columns. Columns {cols} dropped successfully")
    return df
  except Exception as e:
    logging.error(f"Error in drop_columns: {e}")
    raise CustomException(e, sys)