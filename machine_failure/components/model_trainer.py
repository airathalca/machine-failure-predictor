from importlib import import_module
import os
import sys
from typing import Any, Dict, Tuple
import numpy as np

from machine_failure.entity.model import MachineFailureModel
from machine_failure.exception.custom_exception import CustomException
from machine_failure.logger.custom_logging import logging
from machine_failure.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object
from machine_failure.entity.config_entity import ModelTrainerConfig
from machine_failure.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.ensemble import VotingClassifier
import optuna

params = read_yaml_file("config/param.yaml")

class ModelTrainer:
  def __init__(self, data_transformation_artifact: DataTransformationArtifact):
    self.config = ModelTrainerConfig()
    self.transformation_artifact = data_transformation_artifact
    self.schema = read_yaml_file(os.path.join(params["config"]["dir"], params["config"]["model_trainer_config_file_path"]))

  def _get_model_class(self, model_config: Dict[str, Any]) -> Any:
    logging.info("Entered the _get_model_class method of ModelTrainer class")
    try:
      module_name = model_config["module"]
      class_name = model_config["class"]
      module = import_module(module_name)
      model_class = getattr(module, class_name)
      logging.info(f"Exiting the _get_model_class method of ModelTrainer class")
      return model_class
    except Exception as e:
      logging.error(f"Error in cls ModelTrainer method _get_model_class: {e}")
      raise CustomException(e, sys)
    
  def _optimize_model(self, model_name: str, model_config: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    logging.info("Entered the _optimize_model method of ModelTrainer class")
    try:
      model_class = self._get_model_class(model_config)
      def objective(trial):
        params = {}
        for param, space in model_config["search_space"].items():
          if space["type"] == "int":
            params[param] = trial.suggest_int(f"{param}", space["low"], space["high"])
          elif space["type"] == "float":
            if space.get("log", False):
              params[param] = trial.suggest_float(f"{param}", space["low"], space["high"], log=True)
            else:
              params[param] = trial.suggest_float(f"{param}", space["low"], space["high"])
          elif space["type"] == "categorical":
            if "None" in space["choices"]:
              choices = [None if choice == "None" else choice for choice in space["choices"]]
              params[param] = trial.suggest_categorical(f"{param}", choices)
            else:
              params[param] = trial.suggest_categorical(f"{param}", space["choices"])

        model = model_class(**model_config["params"], **params)
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        return roc_auc

      study = optuna.create_study(
        direction=self.model_schema["optuna"]["direction"],
        sampler=optuna.samplers.TPESampler(),
        study_name=f"{model_name}_tuning",
      )
      study.optimize(objective, n_trials=self.model_schema["optuna"]["n_trials"])
      logging.info(f"Best ROC AUC for model {model_name}: {study.best_value}")
      logging.info(f"Best parameters for model {model_name}: {study.best_params}")
      logging.info(f"Exiting the _optimize_model method of ModelTrainer class")
      return study.best_params
    except Exception as e:
      logging.error(f"Error in cls ModelTrainer method _optimize_model: {e}")
      raise CustomException(e, sys)

  def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[object, ClassificationMetricArtifact]:
    logging.info("Entered the train_model method of ModelTrainer class")
    try:
      # Optimize each model individually
      best_models = {}
      for model_name, model_config in self.model_schema["models"].items():
        logging.info(f"Optimizing model: {model_name}")
        best_params = self._optimize_model(model_name, model_config, X_train, y_train, X_test, y_test)

        model_class = self._get_model_class(model_config)
        best_models[model_name] = model_class(**model_config["params"], **best_params)

      # Create VotingClassifier with the best models
      logging.info("Creating VotingClassifier")
      voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting=self.model_schema["voting_classifier"]["params"]["voting"],
        weights=self.model_schema["voting_classifier"]["params"]["weights"]
      )
      voting_clf.fit(X_train, y_train)

      y_pred_proba = voting_clf.predict_proba(X_test)[:, 1]
      y_pred = voting_clf.predict(X_test)

      roc_auc = roc_auc_score(y_test, y_pred_proba)
      f1 = f1_score(y_test, y_pred)
      cm = confusion_matrix(y_test, y_pred)
      logging.info(f"Confusion Matrix: {cm}")

      metric_artifact = ClassificationMetricArtifact(
        roc_auc_score=roc_auc,
        f1_score=f1,
      )
      logging.info(f"ROC AUC: {roc_auc}, F1 Score: {f1}")
      logging.info("Exiting the train_model method of ModelTrainer class")
      return voting_clf, metric_artifact
    except Exception as e:
      logging.error(f"Error in cls ModelTrainer method train_model: {e}")
      raise CustomException(e, sys)

  def get_model(self) -> ModelTrainerArtifact:
    logging.info("Entered the get_model method of ModelTrainer class")
    try:
      logging.info("Loading train transformed data")
      train_arr = load_numpy_array_data(self.transformation_artifact.transformed_train_file_path)
      logging.info("Loading test transformed data")
      test_arr = load_numpy_array_data(self.transformation_artifact.transformed_test_file_path)

      X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
      X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

      logging.info("Train the model")
      model, metric_artifact = self.train_model(X_train, y_train, X_test, y_test)
      if metric_artifact.roc_auc_score < self.config.expected_metric:
        raise CustomException(f"ROC AUC score is less than the expected metric: {metric_artifact.roc_auc_score}", sys)

      preprocessing_obj = load_object(file_path=self.transformation_artifact.transformed_object_file_path)
      save_file = MachineFailureModel(preprocessing_obj, model)
      save_object(self.config.trained_model_file_path, save_file)
      logging.info(f"Model saved at: {self.config.trained_model_file_path}")

      logging.info("Exiting the get_model method of ModelTrainer class")
      return ModelTrainerArtifact(
        trained_model_file_path=self.config.trained_model_file_path,
        metric_artifact=metric_artifact
      )
    except Exception as e:
      logging.error(f"Error in cls ModelTrainer method get_model: {e}")
      raise CustomException(e, sys)

if __name__ == "__main__":
  transformation_artifact = DataTransformationArtifact(
    'artifact/data_transformation/preprocessor.pkl',
    'artifact/data_transformation/data/train.npy',
    'artifact/data_transformation/data/test.npy'
  )
  model_trainer = ModelTrainer(transformation_artifact)
  model_artifact = model_trainer.get_model()