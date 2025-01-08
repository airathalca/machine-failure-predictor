import os
import sys
import pandas as pd

from machine_failure.exception.custom_exception import CustomException
from machine_failure.logger.custom_logging import logging
from machine_failure.utils.main_utils import read_yaml_file, write_yaml_file, read_csv
from machine_failure.entity.config_entity import DataValidationConfig
from machine_failure.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from machine_failure.constants import CONFIG_DIR, SCHEMA_FILE_PATH

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class DataValidation:
  def __init__(self, data_ingestion_artifact: DataIngestionArtifact):
    self.config = DataValidationConfig()
    self.ingestion_artifact = data_ingestion_artifact
    self.schema = read_yaml_file(os.path.join(CONFIG_DIR, SCHEMA_FILE_PATH))

  def validate_total_columns(self, df: pd.DataFrame) -> bool:
    logging.info("Entered the validate_total_columns method of DataValidation class")
    try:
      total_cols = len(df.columns) == len(self.schema['columns'])
      logging.info(f'Is total number of columns in the data frame equal to the schema? {total_cols}')
      logging.info(f"Exiting the validate_total_columns method of DataValidation class")
      return total_cols
    except Exception as e:
      logging.error(f"Error in cls DataValidation method validate_total_columns: {e}")
      raise CustomException(e, sys)

  def validate_columns(self, df: pd.DataFrame) -> bool:
    logging.info("Entered the validate_columns method of DataValidation class")
    try:
      missing_num_col = []
      missing_cat_col = []
      all_exist = True

      for col in self.schema['columns']:
        col_name, col_type = next(iter(col.items()))
        if col_name not in df.columns:
          all_exist = False
          if col_type == 'category':
            missing_cat_col.append(col_name)
          else:
            missing_num_col.append(col_name)
      
      if len(missing_num_col) > 0:
        logging.info(f'Missing numerical columns: {missing_num_col}')
      if len(missing_cat_col) > 0:
        logging.info(f'Missing categorical columns: {missing_cat_col}')
      logging.info(f"Exiting the validate_columns method of DataValidation class")
      return all_exist                          
    except Exception as e:
      logging.error(f"Error in cls DataValidation method validate_columns: {e}")
      raise CustomException(e, sys)
    
  def detect_dataset_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    logging.info("Entered the detect_dataset_drift method of DataValidation class")
    try:
      # Create a data drift report using the new Evidently API
      data_drift_report = Report(metrics=[DataDriftPreset()])
      data_drift_report.run(reference_data=train_df, current_data=test_df)

      # Save the report as a YAML and HTML file
      report_json = data_drift_report.as_dict()
      write_yaml_file(file_path=self.config.drift_report_yaml_file_path, content=report_json)
      data_drift_report.save_html(self.config.drift_report_html_file_path)

      # Extract drift status from the report
      drift_status = report_json["metrics"][0]["result"]["dataset_drift"]
      n_features = report_json["metrics"][0]["result"]["number_of_columns"]
      n_drifted_features = report_json["metrics"][0]["result"]["number_of_drifted_columns"]

      logging.info(f"{n_drifted_features}/{n_features} drift detected.")
      logging.info(f"Exiting the detect_dataset_drift method of DataValidation class")
      return drift_status
    except Exception as e:
      logging.error(f"Error in cls DataValidation method detect_dataset_drift: {e}")
      raise CustomException(e, sys)

  def validate_data(self) -> DataValidationArtifact:
    logging.info("Entered the validate_data method")
    try:
      error_msg = ""
      train_df = read_csv(self.ingestion_artifact.train_file_path)
      test_df = read_csv(self.ingestion_artifact.test_file_path)
      
      logging.info("Validating training data")
      total_col_train = self.validate_total_columns(train_df)
      if not total_col_train:
        error_msg += "Total number of columns in the training data does not match the schema\n"
      all_exist_train = self.validate_columns(train_df)
      if not all_exist_train:
        error_msg += "All columns in the training data do not match the schema\n"

      logging.info("Validating testing data")
      total_col_test = self.validate_total_columns(test_df)
      if not total_col_test:
        error_msg += "Total number of columns in the testing data does not match the schema\n"
      all_exist_test = self.validate_columns(test_df)
      if not all_exist_test:
        error_msg += "All columns in the testing data do not match the schema\n"

      if len(error_msg) > 0:
        logging.info(f"Error in validating data: {error_msg}")
      else:
        drift_status = self.detect_dataset_drift(train_df, test_df)
        if drift_status:
          error_msg += "Dataset drift detected\n"
          logging.info("Dataset drift detected")
        else:
          logging.info("No dataset drift detected. Data is valid")
      logging.info("Exiting the validate_data method of DataValidation class")
      return DataValidationArtifact(
        validation_status=(len(error_msg) == 0),
        message=error_msg,
        drift_report_file_path=self.config.drift_report_yaml_file_path
      )
    except Exception as e:
      logging.error(f"Error in cls DataValidation method validate_data: {e}")
      raise CustomException(e, sys)