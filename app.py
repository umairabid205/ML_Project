from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

print("Starting imports...")

try:
    from src.pipline.predict_pipline import CustomData, PredictPipeline
    print("Successfully imported CustomData and PredictPipeline")
except Exception as e:
    print(f"Error importing: {e}")

print("Creating Flask app...")
application = Flask(__name__)
app = application

## Route for a home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Print form data for debugging
            print("Form data received:")
            for key, value in request.form.items():
                print(f"{key}: {value}")
            
            # Check if all required fields are present
            required_fields = ['gender', 'ethnicity', 'parental_level_of_education', 
                             'lunch', 'test_preparation_course', 'reading_score', 'writing_score']
            
            for field in required_fields:
                if not request.form.get(field):
                    return render_template('home.html', error=f"Missing field: {field}")
            
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            
            pred_df = data.get_data_as_data_frame()
            print("DataFrame created:", pred_df)
            print("Before Prediction")

            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print("Prediction results:", results)
            print("After Prediction")
            
            return render_template('home.html', results=results[0])
            
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return render_template('home.html', error=f"Error: {str(e)}")

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="127.0.0.1", port=5001, debug=True)  # Use a free port, e.g., 5001