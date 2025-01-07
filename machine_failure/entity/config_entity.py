import os
from machine_failure.constants import *
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
  data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)
  raw_file_path: str = os.path.join(data_ingestion_dir, RAW_FILE_NAME)
  training_file_path: str = os.path.join(data_ingestion_dir,  DATA_SPLIT_DIR_NAME, TRAIN_FILE_NAME)
  testing_file_path: str = os.path.join(data_ingestion_dir, DATA_SPLIT_DIR_NAME, TEST_FILE_NAME)
  train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
  collection_name:str = DATA_INGESTION_COLLECTION_NAME

@dataclass
class DataValidationConfig:
  data_validation_dir: str = os.path.join(ARTIFACT_DIR, DATA_VALIDATION_DIR_NAME)
  drift_report_yaml_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_YAML_FILE_NAME)
  drift_report_html_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_HTML_FILE_NAME)

@dataclass
class DataTransformationConfig:
  data_transformation_dir: str = os.path.join(ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME)
  transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_DATA_DIR_NAME, TRAIN_FILE_NAME.replace("csv", "npy"))
  transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_DATA_DIR_NAME, TEST_FILE_NAME.replace("csv", "npy"))
  transformed_object_file_path: str = os.path.join(data_transformation_dir, PREPROCESSOR_FILE_NAME)