import os
from textSummarizer.logging import logger
from textSummarizer.entity import DataValidationConfig

class DataValidation:
    def __init__(self, config):
        self.config = config

    def validate_all_files_exist(self):
        try:
            validation_status = True
            data_dir = os.path.join("artifacts", "data_ingestion")
            all_files = os.listdir(data_dir)
            for required_file in self.config.ALL_REQUIRED_FILES:
                if required_file not in all_files:
                    validation_status = False
                    break
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")
            return validation_status
        except Exception as e:
            raise e