# it's all about reading the data from the source and writing it to the destination
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig




@dataclass
class dataIngestionConfig:
    # this class is used to store the configuration of the data ingestion
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__ (self):
        self.ingestion_config = dataIngestionConfig()
        # this will create an object of the dataIngestionConfig class
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            df = pd.read_csv('notebook/data/stud.csv')     # we can read data from any source here
            logging.info('Read the dataset as dataframe')
            
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # To make dir
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Train Test split initiated.')

            train_set,test_set = train_test_split(df,test_size=0.2, random_state=42)
            # train_test_split will split the data into train and test set
            logging.info('Train Test split completed')
            logging.info(f"Train set shape: {train_set.shape}")
            logging.info(f"Test set shape: {test_set.shape}")
            # this will log the shape of the train and test set
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path, # path to the train data for next step data transformation
                self.ingestion_config.test_data_path # path to the test data for next step data transformation
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data =obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)

        

