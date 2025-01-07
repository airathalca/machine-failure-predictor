import boto3
import os
from dotenv import load_dotenv
from machine_failure.logger.custom_logging import logging

load_dotenv()

class AWSConnection:
  client = None
  resource = None
  region_name = os.getenv("REGION_NAME", "ap-southeast-2")
  def __init__(self, region_name=region_name):
    if AWSConnection.resource == None or AWSConnection.client == None:
      logging.info("AWS Connection has not been established. Establishing connection")
      access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
      secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
      if access_key_id is None:
        raise Exception(f"Environment variable: AWS_ACCESS_KEY_ID is not not set.")
      if secret_access_key is None:
        raise Exception(f"Environment variable: AWS_SECRET_ACCESS_KEY is not set.")
      AWSConnection.resource = boto3.resource('s3', aws_access_key_id_id=access_key_id, 
                                         aws_secret_access_key_id=secret_access_key, region_name=region_name)
      AWSConnection.client = boto3.client('s3', aws_access_key_id_id=access_key_id, 
                                     aws_secret_access_key_id=secret_access_key, region_name=region_name)
    self.resource = AWSConnection.resource
    self.client = AWSConnection.client