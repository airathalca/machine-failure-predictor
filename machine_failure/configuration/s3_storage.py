import boto3
import os
import sys
from io import StringIO
from typing import Union, List
from mypy_boto3_s3.service_resource import Bucket
from botocore.exceptions import ClientError
import pandas as pd
import pickle
from machine_failure.configuration.aws_connection import AWSConnection
from machine_failure.logger.custom_logging import logging
from machine_failure.exception.custom_exception import CustomException

class S3Storage:
  def __init__(self):
    s3_client = AWSConnection()
    self.s3_resource = s3_client.resource
    self.s3_client = s3_client.client

  def s3_key_path_available(self, bucket_name, s3_key) -> bool:
    logging.info("Entered the s3_key_path_available method of S3Storage class")
    try:
      logging.info(f"Checking if {s3_key} path is available in {bucket_name} bucket")
      bucket = self.get_bucket(bucket_name)
      file_objects = [file_object for file_object in bucket.objects.filter(Prefix=s3_key)]
      if len(file_objects) > 0:
          return True
      else:
          return False
    except Exception as e:
      logging.error(f"Error in cls S3Storage method s3_key_path_available: {e}")
      raise CustomException(e, sys)
      
  @staticmethod
  def read_object(object_name: str, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:
    logging.info("Entered the read_object method of S3Operations class")
    try:
      func = (
          lambda: object_name.get()["Body"].read().decode()
          if decode is True
          else object_name.get()["Body"].read()
      )
      conv_func = lambda: StringIO(func()) if make_readable is True else func()
      logging.info("Exited the read_object method of S3Operations class")
      return conv_func()
    except Exception as e:
      logging.error(f"Error in cls S3Storage method read_object: {e}")
      raise CustomException(e, sys)

  def get_bucket(self, bucket_name: str) -> Bucket:
    logging.info("Entered the get_bucket method of S3Operations class")
    try:
      bucket = self.s3_resource.Bucket(bucket_name)
      logging.info("Exited the get_bucket method of S3Operations class")
      return bucket
    except Exception as e:
      logging.error(f"Error in cls S3Storage method get_bucket: {e}")
      raise CustomException(e, sys)

  def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
    logging.info("Entered the get_file_object method of S3Operations class")
    try:
      bucket = self.get_bucket(bucket_name)
      file_objects = [file_object for file_object in bucket.objects.filter(Prefix=filename)]
      func = lambda x: x[0] if len(x) == 1 else x
      file_objs = func(file_objects)
      logging.info("Exited the get_file_object method of S3Operations class")
      return file_objs
    except Exception as e:
      logging.error(f"Error in cls S3Storage method get_file_object: {e}")
      raise CustomException(e, sys)

  def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
    logging.info("Entered the load_model method of S3Operations class")
    try:
      func = (lambda: model_name if model_dir is None else model_dir + "/" + model_name)
      model_file = func()
      file_object = self.get_file_object(model_file, bucket_name)
      model_obj = self.read_object(file_object, decode=False)
      model = pickle.loads(model_obj)
      logging.info("Exited the load_model method of S3Operations class")
      return model
    except Exception as e:
      logging.error(f"Error in cls S3Storage method load_model: {e}")
      raise CustomException(e, sys)

  def create_folder(self, folder_name: str, bucket_name: str) -> None:
    logging.info("Entered the create_folder method of S3Operations class")
    try:
      self.s3_resource.Object(bucket_name, folder_name).load()
    except ClientError as e:
      if e.response["Error"]["Code"] == "404":
        folder_obj = folder_name + "/"
        self.s3_client.put_object(Bucket=bucket_name, Key=folder_obj)
      else:
        pass
      logging.info("Exited the create_folder method of S3Operations class")

  def upload_file(self, local_filename: str, bucket_filename: str,  bucket_name: str,  remove: bool = True):
    logging.info("Entered the upload_file method of S3Operations class")
    try:
      logging.info(
        f"Uploading {local_filename} file to {bucket_filename} file in {bucket_name} bucket"
      )
      self.s3_resource.meta.client.upload_file(
        local_filename, bucket_name, bucket_filename
      )
      logging.info(
        f"Uploaded {local_filename} file to {bucket_filename} file in {bucket_name} bucket"
      )
      if remove is True:
        os.remove(local_filename)
        logging.info(f"Remove is set to {remove}, deleted the file")
      else:
        logging.info(f"Remove is set to {remove}, not deleted the file")
      logging.info("Exited the upload_file method of S3Operations class")
    except Exception as e:
      logging.error(f"Error in cls S3Storage method upload_file: {e}")
      raise CustomException(e, sys)

  def upload_df_as_csv(self,df: pd.DataFrame, local_filename: str, bucket_filename: str, bucket_name: str) -> None:
    logging.info("Entered the upload_df_as_csv method of S3Operations class")
    try:
      df.to_csv(local_filename, index=None, header=True)
      self.upload_file(local_filename, bucket_filename, bucket_name)
      logging.info("Exited the upload_df_as_csv method of S3Operations class")
    except Exception as e:
      logging.error(f"Error in cls S3Storage method upload_df_as_csv: {e}")
      raise CustomException(e, sys)

  def get_df_from_object(self, object_: object) -> pd.DataFrame:
    logging.info("Entered the get_df_from_object method of S3Operations class")
    try:
      content = self.read_object(object_, make_readable=True)
      df = pd.read_csv(content, na_values="na")
      logging.info("Exited the get_df_from_object method of S3Operations class")
      return df
    except Exception as e:
      logging.error(f"Error in cls S3Storage method get_df_from_object: {e}")
      raise CustomException(e, sys)
  
  def read_csv(self, filename: str, bucket_name: str) -> pd.DataFrame:
    logging.info("Entered the read_csv method of S3Operations class")
    try:
      csv_obj = self.get_file_object(filename, bucket_name)
      df = self.get_df_from_object(csv_obj)
      logging.info("Exited the read_csv method of S3Operations class")
      return df
    except Exception as e:
      logging.error(f"Error in cls S3Storage method read_csv: {e}")
      raise CustomException(e, sys)