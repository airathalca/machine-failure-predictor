import os
from datetime import date

DATABASE_NAME = "machine-data"
COLLECTION_NAME = "machine_data_failure"

ARTIFACT_DIR: str = "artifact"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
FILE_NAME: str = "data.csv"
MODEL_FILE_NAME = "model.pkl"

CONFIG_DIR: str = "config"
SCHEMA_FILE_PATH: str = "schema.yaml"

DATA_INGESTION_COLLECTION_NAME: str = "machine_data_failure"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_SPLIT_DIR_NAME: str = "split"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_YAML_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_REPORT_HTML_FILE_NAME: str = "report.html"