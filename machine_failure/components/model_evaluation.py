import os
import sys
from typing import Optional
from urllib.parse import urlparse
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score
from machine_failure.entity.s3_model import MachineFailureS3Model
from machine_failure.utils.main_utils import load_numpy_array_data, load_object
from machine_failure.logger.custom_logging import logging
from machine_failure.exception.custom_exception import CustomException
from machine_failure.entity.config_entity import ModelBucketConfig
from machine_failure.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact

import mlflow
import mlflow.sklearn

class ModelEvaluation:
  def __init__(self, model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact):
    self.config = ModelBucketConfig()
    self.model_trainer_artifact = model_trainer_artifact
    self.transformation_artifact = data_transformation_artifact

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
      test_np = load_numpy_array_data(self.transformation_artifact.transformed_test_file_path)
      X_test, y_test = test_np[:, :-1], test_np[:, -1]
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
            mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
          else:
            mlflow.sklearn.log_model(model, "model")
      else:
        roc_auc = 0
        f1 = 0

      is_model_accepted = self.model_trainer_artifact.metric_artifact.roc_auc_score > roc_auc
      logging.info('Model evaluation completed')
      return ModelEvaluationArtifact(model_accepted=is_model_accepted, s3_model_path=self.config.s3_model_key_path, 
                                     trained_model_path=self.model_trainer_artifact.trained_model_file_path)
    except Exception as e:
      raise CustomException(e, sys)