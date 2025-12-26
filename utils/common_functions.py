import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml


logger=get_logger(__name__)

def read_yaml(file_path:str):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at path {file_path} does not exist.")
        
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            logger.info(f"Successfully read YAML file at {file_path}")
            return content
    except Exception as e:
        logger.error(f"Error reading YAML file at {file_path}: {e}")
        raise CustomException(e, sys)
    
    
def load_data(path):
    try:
        logger.info(f"Loading data from {path}")
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error loading data from {path}: {e}")
        raise CustomException(e, sys)
    