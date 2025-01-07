import os
import sys
from typing import Optional
from urllib.parse import urlparse
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score
from machine_failure.constants import CONFIG_DIR, SCHEMA_FILE_PATH
from machine_failure.entity.s3_model import MachineFailureS3Model
from machine_failure.utils.main_utils import drop_columns, read_yaml_file
from machine_failure.logger.custom_logging import logging
from machine_failure.exception.custom_exception import CustomException
from machine_failure.entity.config_entity import ModelBucketConfig
from machine_failure.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, ModelEvaluationArtifact

import mlflow
import mlflow.sklearn

class ModelEvaluation:
  def __init__(self, model_trainer_artifact: ModelTrainerArtifact, data_ingestion_artifact: DataIngestionArtifact):
    self.config = ModelBucketConfig()
    self.model_trainer_artifact = model_trainer_artifact
    self.ingestion_artifact = data_ingestion_artifact
    self.schema = read_yaml_file(os.path.join(CONFIG_DIR, SCHEMA_FILE_PATH))

  def get_bucket_model(self) -> Optional[MachineFailureS3Model]:
    logging.info('Entered the get_bucket_model method')
    try:
      bucket_name = self.config.bucket_name
      model_path=self.config.s3_model_key_path
      model_bucket = MachineFailureS3Model(bucket_name, model_path)
      if model_bucket.is_model_present(model_path=model_path):
        return model_bucket
      logging.error('Model not found in the bucket')
      return None

    except Exception as e:
      raise CustomException(e, sys)

  def evaluate_model(self) -> ModelEvaluationArtifact:
    try:
      test_df = pd.read_csv(self.ingestion_artifact.test_file_path)
      X_test = test_df.drop(columns=[self.schema['target']], axis=1)
      y_test = test_df[self.schema['target']]
      X_test['Product ID'] = X_test['Product ID'].str.replace('M','').str.replace('L','').str.replace('H','').astype(int)
      X_test = drop_columns(X_test, self.schema['drop_columns'])
      logging.info('Model evaluation started')

      logging.info('Getting the model from the bucket')
      model = self.get_bucket_model()
      if model is not None:
        tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
          pred_proba = model.predict_proba(X_test)[:, 1]
          roc_auc = roc_auc_score(y_test, pred_proba)
          pred = model.predict(X_test)
          f1 = f1_score(y_test, pred)

          mlflow.log_metric("roc_auc", roc_auc)
          mlflow.log_metric("f1_score", f1)
          if tracking_url_type != "file":
            mlflow.sklearn.log_model(model.loaded_model.trained_model_object, "model", registered_model_name="ml_model")
          else:
            mlflow.sklearn.log_model(model.loaded_model.trained_model_object, "model")
      else:
        roc_auc = 0
        f1 = 0

      is_model_accepted = self.model_trainer_artifact.metric_artifact.roc_auc_score > roc_auc
      logging.info('Model evaluation completed')
      return ModelEvaluationArtifact(model_accepted=is_model_accepted, s3_model_path=self.config.s3_model_key_path, 
                                     trained_model_path=self.model_trainer_artifact.trained_model_file_path)
    except Exception as e:
      raise CustomException(e, sys)