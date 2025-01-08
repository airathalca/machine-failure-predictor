import os
from machine_failure.utils.main_utils import read_yaml_file
from dataclasses import dataclass

params = read_yaml_file("config/param.yaml")

@dataclass
class DataIngestionConfig:
  data_ingestion_dir: str = os.path.join(params["artifact"]["dir"], params["data_ingestion"]["dir_name"])
  raw_file_path: str = os.path.join(data_ingestion_dir, params["artifact"]["raw_file_name"])
  training_file_path: str = os.path.join(data_ingestion_dir, params["data_ingestion"]["split_dir_name"], params["artifact"]["train_file_name"])
  testing_file_path: str = os.path.join(data_ingestion_dir, params["data_ingestion"]["split_dir_name"], params["artifact"]["test_file_name"])
  train_test_split_ratio: float = params["data_ingestion"]["train_test_split_ratio"]
  collection_name: str = params["database"]["collection"]

@dataclass
class DataValidationConfig:
  data_validation_dir: str = os.path.join(params["artifact"]["dir"], params["data_validation"]["dir_name"])
  drift_report_yaml_file_path: str = os.path.join(data_validation_dir, params["data_validation"]["report_yaml_file_name"])
  drift_report_html_file_path: str = os.path.join(data_validation_dir, params["data_validation"]["report_html_file_name"])

@dataclass
class DataTransformationConfig:
  data_transformation_dir: str = os.path.join(params["artifact"]["dir"], params["data_transformation"]["dir_name"])
  transformed_train_file_path: str = os.path.join(data_transformation_dir, params["data_transformation"]["data_dir_name"], params["artifact"]["train_file_name"].replace("csv", "npy"))
  transformed_test_file_path: str = os.path.join(data_transformation_dir, params["data_transformation"]["data_dir_name"], params["artifact"]["test_file_name"].replace("csv", "npy"))
  transformed_object_file_path: str = os.path.join(data_transformation_dir, params["artifact"]["preprocessor_file_name"])

@dataclass
class ModelTrainerConfig:
  model_trainer_dir: str = os.path.join(params["artifact"]["dir"], params["model_trainer"]["dir_name"])
  trained_model_file_path: str = os.path.join(model_trainer_dir, params["artifact"]["model_file_name"])
  expected_metric: float = params["model_trainer"]["expected_roc_score"]
  model_config_file_path: str = params["config"]["model_trainer_config_file_path"]

@dataclass
class ModelBucketConfig:
  bucket_name: str = params["cloud"]["model_bucket_name"]
  s3_model_key_path: str = params["artifact"]["model_file_name"]