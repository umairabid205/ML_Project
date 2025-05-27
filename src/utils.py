# let say I wanna read dataset from a database , I can create my mongo client here
# If i want to save my model into the clould, I can write the code here

import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV





def save_object(file_path, obj):
    """
    Save an object to a file using pickle.

    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.

    Raises:
    - CustomException: If there is an error during saving the object.
    """
    try:
        import pickle
        
        dir_path = os.path.dirname(file_path)
       
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create directory if it doesn't exist       
        
        with open(file_path, 'wb') as file_obj: # Open the file in binary write mode
            dill.dump(obj, file_obj)            # Use dill to serialize the object and write it to the file
        logging.info(f"Object saved successfully at {file_path}")
    
    except Exception as e:
        raise CustomException(e,sys)
    

    
def evaluate_model(X_train, y_train, X_test, y_test, models,params):
    from sklearn.metrics import r2_score
    import numpy as np

    model_report = {}
    for model_name, model in models.items():
        try:
            #model.fit(X_train, y_train)
            para = params.get(model_name, {})
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=2) if para else model
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_) if para else None  # Set best parameters if available
            model.fit(X_train, y_train)  # Fit the model on training data
            logging.info(f"Model {model_name} trained with parameters: {model.get_params()}")
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            model_report[model_name] = score
        except Exception as e:
            # Log or print the error for debugging
            print(f"Model {model_name} failed: {e}")
            model_report[model_name] = None  # or np.nan
    return model_report
  