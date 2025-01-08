from machine_failure.logger.custom_logging import logging
from machine_failure.exception.custom_exception import CustomException
import sys

def write_yaml_file() -> None:
  logging.info("Entered the write_yaml_file method of utils")
  try:
    raise ValueError("xadasdas")
  except Exception as e:
    logging.error(f"Error in write_yaml_file: {e}")
    raise CustomException("Error in write_yaml_file", sys)

def detect_dataset_drift() -> bool:
  logging.info("Entered the detect_dataset_drift method")
  try:
    write_yaml_file()
    return True
  except Exception as e:
    logging.error(f"Error in detect_dataset_drift: {e}")
    raise CustomException("Error in detect_dataset_drift", sys)
  
if __name__ == "__main__":
  detect_dataset_drift()