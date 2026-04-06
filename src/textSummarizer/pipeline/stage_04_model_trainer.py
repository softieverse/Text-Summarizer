from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.logging import logger
import os

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        
        # Skip if model already trained
        model_path = os.path.join(model_trainer_config.root_dir, "pegasus-samsum-model")
        if os.path.exists(model_path):
            logger.info("Model already trained, skipping training stage!")
            return
        
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()