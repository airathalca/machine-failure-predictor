import sys
from machine_failure.exception.custom_exception import CustomException
from machine_failure.logger.custom_logging import logging
from machine_failure.components.data_ingestion import DataIngestion
from machine_failure.components.data_validation import DataValidation
from machine_failure.components.data_transformation import DataTransformation
from machine_failure.components.model_trainer import ModelTrainer
from machine_failure.components.model_evaluation import ModelEvaluation
from machine_failure.components.model_pusher import ModelPusher
from machine_failure.entity.artifact_entity import (ClassificationMetricArtifact, DataIngestionArtifact,
                                            DataValidationArtifact,
                                            DataTransformationArtifact,
                                            ModelTrainerArtifact,
                                            ModelEvaluationArtifact,
                                            ModelPusherArtifact)

class TrainPipeline:
  def __init__(self):
    pass

  def start_data_ingestion(self) -> DataIngestionArtifact:
    logging.info("Entered the start_data_ingestion method of TrainPipeline class")
    try:
      data_ingestion = DataIngestion()
      data_ingestion_artifact = data_ingestion.read_data()
      logging.info("Exiting the start_data_ingestion method of TrainPipeline class")
      return data_ingestion_artifact
    except Exception as e:
      logging.error(f"Error in cls TrainPipeline method start_data_ingestion: {e}")
      raise CustomException(e, sys)

  def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
    logging.info("Entered the start_data_validation method of TrainPipeline class")
    try:
      data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact)
      data_validation_artifact = data_validation.validate_data()
      logging.info("Exiting the start_data_validation method of TrainPipeline class")
      return data_validation_artifact
    except Exception as e:
      logging.error(f"Error in cls TrainPipeline method start_data_validation: {e}")
      raise CustomException(e, sys)

  def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
    logging.info("Entered the start_data_transformation method of TrainPipeline class")
    try:
      data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                data_validation_artifact=data_validation_artifact)
      data_transformation_artifact = data_transformation.transform_data()
      logging.info("Exiting the start_data_transformation method of TrainPipeline class")
      return data_transformation_artifact
    except Exception as e:
      logging.error(f"Error in cls TrainPipeline method start_data_transformation: {e}")
      raise CustomException(e, sys)

  def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
    logging.info("Entered the start_model_trainer method of TrainPipeline class")
    try:
      model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact)
      model_trainer_artifact = model_trainer.get_model()
      logging.info("Exiting the start_model_trainer method of TrainPipeline class")
      return model_trainer_artifact
    except Exception as e:
      logging.error(f"Error in cls TrainPipeline method start_model_trainer: {e}")
      raise CustomException(e, sys)

  def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact, model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
    logging.info("Entered the start_model_evaluation method of TrainPipeline class")
    try:
      model_evaluation = ModelEvaluation(model_trainer_artifact=model_trainer_artifact,
                                          data_ingestion_artifact=data_ingestion_artifact)
      model_evaluation_artifact = model_evaluation.evaluate_model()
      logging.info("Exiting the start_model_evaluation method of TrainPipeline class")
      return model_evaluation_artifact
    except Exception as e:
      logging.error(f"Error in cls TrainPipeline method start_model_evaluation: {e}")
      raise CustomException(e, sys)
  
  def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
    logging.info("Entered the start_model_pusher method of TrainPipeline class")
    try:
      model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact)
      model_pusher_artifact = model_pusher.push_model()
      logging.info("Exiting the start_model_pusher method of TrainPipeline class")
      return model_pusher_artifact
    except Exception as e:
      logging.error(f"Error in cls TrainPipeline method start_model_pusher: {e}")
      raise CustomException(e, sys)

  def run_pipeline(self) -> None:
    logging.info("Entered the run_pipeline method of TrainPipeline class")
    try:
      data_ingestion_artifact = self.start_data_ingestion()
      data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
      data_transformation_artifact = self.start_data_transformation(
          data_ingestion_artifact=data_ingestion_artifact, data_validation_artifact=data_validation_artifact)
      model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
      model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                              model_trainer_artifact=model_trainer_artifact)
      if not model_evaluation_artifact.model_accepted:
          logging.info(f"Model not accepted. Exiting the run_pipeline method of TrainPipeline class without pushing the model")
          return None
      self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
      logging.info("Exiting the run_pipeline method of TrainPipeline class")
      return None
    except Exception as e:
      logging.error(f"Error in cls TrainPipeline method run_pipeline: {e}")
      raise CustomException(e, sys)

if __name__ == '__main__':
  train_pipeline = TrainPipeline()
  train_pipeline.run_pipeline()
