import os
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

from machine_failure.exception.custom_exception import CustomException
from machine_failure.logger.custom_logging import logging
from machine_failure.utils.main_utils import drop_columns, read_yaml_file, read_csv, save_numpy_array_data, save_object
from machine_failure.entity.config_entity import DataTransformationConfig
from machine_failure.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from machine_failure.constants import CONFIG_DIR, SCHEMA_FILE_PATH

class FeatureGenerator(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self

  def transform(self, X):
    data = X.copy()
    # Create new features
    data['Power'] = data['Torque [Nm]'] * data['Rotational speed [rpm]']
    data['Temp Ratio'] = data['Process temperature [K]'] / data['Air temperature [K]']
    data['Torque X Tool wear'] = data['Torque [Nm]'] * data['Tool wear [min]']
    max_tool_wear = data['Tool wear [min]'].max()
    data['Tool Wear Rate'] = data['Tool wear [min]'] / max_tool_wear
    return data

class DataTransformation:
  def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact):
    self.config = DataTransformationConfig()
    self.ingestion_artifact = data_ingestion_artifact
    self.validation_artifact = data_validation_artifact
    self.schema = read_yaml_file(os.path.join(CONFIG_DIR, SCHEMA_FILE_PATH))

  def create_preprocessor(self) -> Pipeline:
    logging.info("Entered the create_preprocessor method")
    try:
      categorical_cols = self.schema['cat_columns']
      numerical_cols = self.schema['num_columns']
      logging.info("Initialize numerical pipeline")
      num_pipeline = Pipeline([
        ('scaler', StandardScaler())
      ])

      logging.info("Initialize categorical pipeline")
      cat_pipeline = Pipeline([
        ('label_encoder', OrdinalEncoder()),
        ('std_scaler', StandardScaler())
      ])

      logging.info("Initialize preprocessor")
      preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
      ])

      logging.info("Creating Full Pipeline")
      full_preprocessor = Pipeline([
        ('feature_generator', FeatureGenerator()),
        ('preprocessor', preprocessor)
      ])
      logging.info('Preprocessor created successfully')

      return full_preprocessor
    except Exception as e:
      logging.error("Error in cls DataTransformation method create_preprocessor: {e}")
      raise CustomException(e, sys)
    
  def transform_data(self) -> DataTransformationArtifact:
    logging.info("Entered the transform_data method")
    try:
      if not self.validation_artifact.validation_status:
        raise CustomException("Data validation failed", sys)
      
      logging.info("Starting data transformation")
      preprocessor = self.create_preprocessor()
      logging.info("Preprocessor object obtained")

      train_df = read_csv(self.ingestion_artifact.train_file_path)
      test_df = read_csv(self.ingestion_artifact.test_file_path)

      feature_train_df = train_df.drop(columns=[self.schema['target']], axis=1)
      target_train_df = train_df[self.schema['target']]
      logging.info('Feature and target dataframes created for training data')

      feature_test_df = test_df.drop(columns=[self.schema['target']], axis=1)
      target_test_df = test_df[self.schema['target']]
      logging.info('Feature and target dataframes created for testing data')

      feature_train_df['Product ID'] = feature_train_df['Product ID'].str.replace('M','').str.replace('L','').str.replace('H','').astype(int)
      feature_test_df['Product ID'] = feature_test_df['Product ID'].str.replace('M','').str.replace('L','').str.replace('H','').astype(int)
      logging.info('Product ID column transformed from categorical to numerical')

      feature_train_df = drop_columns(feature_train_df, self.schema['drop_columns'])
      feature_test_df = drop_columns(feature_test_df, self.schema['drop_columns'])
      logging.info('Necessary columns dropped from feature dataframes')

      logging.info('Fitting and transforming feature dataframes')
      feature_train_df = preprocessor.fit_transform(feature_train_df)
      feature_test_df = preprocessor.transform(feature_test_df)
      logging.info('Data transformation completed')

      train_arr = np.concatenate((feature_train_df, target_train_df.values.reshape(-1,1)), axis=1)
      test_arr = np.concatenate((feature_test_df, target_test_df.values.reshape(-1,1)), axis=1)

      logging.info('Saving transformed data to numpy files')
      save_numpy_array_data(self.config.transformed_train_file_path, train_arr)
      save_numpy_array_data(self.config.transformed_test_file_path, test_arr)

      logging.info('Saving preprocessor object')
      save_object(self.config.transformed_object_file_path, preprocessor)

      return DataTransformationArtifact(
        transformed_train_file_path=self.config.transformed_train_file_path,
        transformed_test_file_path=self.config.transformed_test_file_path,
        transformed_object_file_path=self.config.transformed_object_file_path
      )
    except Exception as e:
      logging.error(f"Error in cls DataTransformation method transform_data: {e}")
      raise CustomException(e, sys)