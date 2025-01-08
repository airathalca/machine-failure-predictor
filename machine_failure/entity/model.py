import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from machine_failure.logger.custom_logging import logging
from machine_failure.exception.custom_exception import CustomException

class MachineFailureModel:
  def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
    self.preprocessing_object = preprocessing_object
    self.trained_model_object = trained_model_object

  def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
    logging.info("Entered predict method of MachineFailureModel class")
    try:
      transformed_feature = self.preprocessing_object.transform(dataframe)
      logging.info("Exiting the predict method of MachineFailureModel class")
      return self.trained_model_object.predict(transformed_feature)
    except Exception as e:
      logging.error(f"Error in cls MachineFailureModel method predict: {e}")
      raise CustomException(e, sys)
    
  def predict_proba(self, dataframe: pd.DataFrame) -> pd.DataFrame:
    logging.info("Entered predict_proba method of MachineFailureModel class")
    try:
      transformed_feature = self.preprocessing_object.transform(dataframe)
      logging.info("Exiting the predict_proba method of MachineFailureModel class")
      return self.trained_model_object.predict_proba(transformed_feature)
    except Exception as e:
      logging.error(f"Error in cls MachineFailureModel method predict_proba: {e}")
      raise CustomException(e, sys)

  def __repr__(self):
    return f"{type(self.trained_model_object).__name__}()"

  def __str__(self):
    return f"{type(self.trained_model_object).__name__}()"
    