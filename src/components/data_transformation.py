import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # for pipline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging 
# from src.components import data_ingestion
from src.utils import save_object # Function to save the preprocessor object






@dataclass
class DataTransformationConfig: # Configuration for data transformation
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() # Initialize the configuration

    
    
    
    def get_data_transformer_object(self): # Create a data transformer object
        """""This function creates a preprocessor object that applies transformations to the dataset.
        It includes both numerical and categorical transformations using pipelines.
        """
   
   
        try:
            numerical_columns = ['writing_score', 'reading_score'] # Numerical columns to be transformed
            categorical_columns =  [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",   
                "test_preparation_course"

            ] # Categorical columns to be transformed



# Numerical pipeline
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),   # Impute missing values with median
                ('scaler', StandardScaler())                     # Scale numerical features
            ])



# Categorical pipeline
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # Impute missing values with most frequent
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

           
           
            # Combine numerical and categorical pipelines into a ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipline', numerical_pipeline, numerical_columns), # Apply numerical pipeline to numerical columns
                    ('cat_pipline', categorical_pipeline, categorical_columns)  # Apply categorical pipeline to categorical columns
                ]
            )
           
           
           
            logging.info("Preprocessor object created successfully.")
            
            return preprocessor # Return the preprocessor object
       
       
        except Exception as e:
            raise CustomException(e, sys)
        


    def initiate_data_transformation(self, train_path, test_path): # Initiate data transformation   
        


        try:
            
            train_df = pd.read_csv(train_path) # Read training data
            test_df = pd.read_csv(test_path)   # Read testing data

            logging.info("Read train and test data completed successfully.")
            logging.info("Obtaining preprocessor object.")  

            preprocessor_obj = self.get_data_transformer_object() # Get the preprocessor object

            logging.info("Applying preprocessor object on training and testing data.")
            
            target_column_name = 'math_score' # Target column for prediction
            numerical_columns = ['writing_score', 'reading_score'] # Numerical columns
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1) # Drop target column from training data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1) # Drop target column from testing data
            
            target_feature_train_df = train_df[target_column_name] # Target feature for training data
            target_feature_test_df = test_df[target_column_name] # Target feature for testing data
            # Apply transformations to training and testing data
            
            logging.info("Applying preprocessor object on training data.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df) # Fit and transform training data
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df) # Transform testing data 

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] # Combine input features and target for training data
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)] # Combine input features and target for testing data
            logging.info("Applying preprocessor object on testing data.")   

            
            # Save the preprocessor object to a file(pkl file))
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                 obj=preprocessor_obj
                ) # Save the preprocessor object using the utility function
            
            
            logging.info("Preprocessor object saved successfully.")
            logging.info("Data transformation completed successfully.")
            
            
            
            
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path, # pkl file path for the preprocessor object
                preprocessor_obj
            
            ) # Return transformed training and testing data along with the preprocessor object
       
        except Exception as e:
            raise CustomException(e, sys)
      




