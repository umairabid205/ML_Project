import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model
# from src.utils import evaluate_model # Function to evaluate models is imported from utils.py



@dataclass
class ModelTrainerConfig: # Configuration for model training
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')



class ModelTrainer: # Class for training machine learning models
    def __init__(self):
            self.model_trainer_config = ModelTrainerConfig() # Initialize the configuration

    
    def initiate_model_trainer(self, train_array, test_array): # Method to initiate model training
         


         try:
              logging.info("Splitting training and testing data")
              X_train, y_train, X_test, y_test = (
                    train_array[:,:-1], 
                    train_array[:,-1], 
                    test_array[:,:-1], 
                    test_array[:,-1]
                ) # Split the data into features and target variable
              
              models = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    'KNeighborsRegressor': KNeighborsRegressor(),
    'RandomForestRegressor': RandomForestRegressor(),
    'GradientBoostingRegressor': GradientBoostingRegressor(),
    'AdaBoostRegressor': AdaBoostRegressor(),
    'XGBRegressor': XGBRegressor(),
    'CatBoostRegressor': CatBoostRegressor(verbose=False)
} # Dictionary of models to be trained
            


              params = {
                         "DecisionTreeRegressor": {
                                                 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        # 'splitter': ['best', 'random'],
        # 'max_features': ['sqrt', 'log2'],
    },
    "RandomForestRegressor": {
        # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        # 'max_features': ['sqrt', 'log2', None],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "GradientBoostingRegressor": {
        # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        # 'criterion': ['squared_error', 'friedman_mse'],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "LinearRegression": {},
    "XGBRegressor": {
        'learning_rate': [0.1, 0.01, 0.05, 0.001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },
    "CatBoostRegressor": {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [30, 50, 100]
    },
    "AdaBoostRegressor": {
        'learning_rate': [0.1, 0.01, 0.5, 0.001],
        # 'loss': ['linear', 'square', 'exponential'],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    }
}















              model_report: dict = evaluate_model(X_train,y_train,X_test,y_test,models,params) # this function is in utils.py file
              logging.info(f"Model Report: {model_report}") # Log the model report

              # for best model score
              best_model_score = max(sorted(model_report.values())) # Get the best model score
              best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] # Get the name of the best model

              best_model = models[best_model_name] # Get the best model instance  

              logging.info(f"Best model found: {best_model_name} with score: {best_model_score}") # Log the best model and its score
              
              if best_model_score < 0.6:
                  raise CustomException("No best model found with sufficient score", sys)
              
              save_object(
                  file_path=self.model_trainer_config.trained_model_file_path, 
                  obj=best_model
              ) # Save the best model to a file
              logging.info(f"Model saved at {self.model_trainer_config.trained_model_file_path}") # Log the model saving

              predictions = best_model.predict(X_test) # Make predictions using the best model
              r2_square = r2_score(y_test, predictions) # Calculate R-squared score for the predictions
              return r2_square # Return the R-squared score

         except Exception as e:
              raise CustomException(e, sys)

