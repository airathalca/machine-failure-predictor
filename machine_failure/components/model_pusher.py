import sys
from machine_failure.configuration.s3_storage import S3Storage
from machine_failure.exception.custom_exception import CustomException
from machine_failure.logger.custom_logging import logging
from machine_failure.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from machine_failure.entity.config_entity import ModelBucketConfig
from machine_failure.entity.s3_model import MachineFailureS3Model
class ModelPusher:
  def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact):
    self.s3 = S3Storage()
    self.model_evaluation_artifact = model_evaluation_artifact
    self.config = ModelBucketConfig()

  def push_model(self) -> ModelPusherArtifact:
    logging.info("Entered push_model method of ModelPusher class")
    try:
      logging.info("Uploading artifacts folder to s3 bucket")
      self.s3.upload_file(self.model_evaluation_artifact.trained_model_path, self.config.s3_model_key_path,
                          self.config.bucket_name, remove=False)
      model_pusher_artifact = ModelPusherArtifact(bucket_name=self.config.bucket_name,
                                                  s3_model_path=self.config.s3_model_key_path)
      logging.info("Uploaded artifacts folder to s3 bucket")
      logging.info(f"Model pusher artifact: [{model_pusher_artifact}]")
      return model_pusher_artifact
    except Exception as e:
      raise CustomException(e, sys)