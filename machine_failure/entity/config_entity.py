import os
from machine_failure.constants import *
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(ARTIFACT_DIR, DATA_INGESTION_DIR_NAME)
    raw_file_path: str = os.path.join(data_ingestion_dir, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir,  DATA_SPLIT_DIR_NAME, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_SPLIT_DIR_NAME, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME