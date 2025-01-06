import os
from datetime import date

DATABASE_NAME = "machine-data"
COLLECTION_NAME = "machine_data_failure"

ARTIFACT_DIR: str = "artifact"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
FILE_NAME: str = "data.csv"
MODEL_FILE_NAME = "model.pkl"

DATA_INGESTION_COLLECTION_NAME: str = "machine_data_failure"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_SPLIT_DIR_NAME: str = "split"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2