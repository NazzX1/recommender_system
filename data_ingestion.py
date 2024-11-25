import os
import sys
from utils.exception import CustomException
from utils.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from data_transformation import DataTransformation
from data_transformation import DataTransformationConfig

from model_trainer import ModelTrainerConfig
from model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('data', "train_data.csv")
    test_data_path: str=os.path.join('data', "test_data.csv")
    raw_data_path: str=os.path.join('data', "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion ...")
        try:
            df=pd.read_csv('data/Sales_Amazon_Cleaned_final.csv')
            logging.info('Read the dataset as dataframe')
            
            user_id_counts = df['user_id'].value_counts()
            unique_user_ids = user_id_counts[user_id_counts == 1].index.tolist()
            df = df[~df['user_id'].isin(unique_user_ids)]
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df, test_size=0.2, random_state=0)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))