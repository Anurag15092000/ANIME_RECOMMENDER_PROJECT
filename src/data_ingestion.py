import os
import sys
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger=get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config=config["data_ingestion"]
        self.bucket_name=self.config["bucket_name"]
        self.file_names=self.config["bucket_file_names"]
        
        os.makedirs(RAW_DIR,exist_ok=True)
        logger.info(f"Created directory for raw data at {RAW_DIR}")
        
        
    def download_csv_with_gcp(self):
        try:
            client=storage.Client()
            bucket=client.bucket(self.bucket_name)
            for file_name in self.file_names:
                file_path=os.path.join(RAW_DIR,file_name)
                if file_name=="animelist.csv":
                    blob=bucket.blob(file_name)
                    blob.download_to_filename(file_path)
                    
                    data=pd.read_csv(file_path,nrows=5000000)
                    data.to_csv(file_path,index=False)
                    logger.info(f"Downloaded and limited {file_name} to 5 million rows at {file_path}")
                else:
                    blob=bucket.blob(file_name)
                    blob.download_to_filename(file_path)
                    logger.info(f"Downloaded {file_name} from bucket {self.bucket_name} to {file_path}")
        except Exception as e:
            logger.error(f"Error occurred while downloading files from GCP: {e}")
            raise CustomException(e,sys)
        
    def run(self):
        try:   
            logger.info("Starting data ingestion process")
            self.download_csv_with_gcp()
            logger.info("Data ingestion process completed")
        except Exception as e:
            logger.error(f"Error in data ingestion run method: {e}")
            raise CustomException(e,sys)
        finally:
            logger.info("Exiting DataIngestion run method")
       
       
if __name__=="__main__":
    dataingestion=DataIngestion(read_yaml(CONFIG_PATH)) 
    dataingestion.run()
