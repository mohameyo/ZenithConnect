#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import joblib
import flask
import json
from flask import Flask, request, jsonify
from scripts import cleansing
from scripts import preprocessing

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('models/logitmodel.joblib')
scaler = joblib.load('models/scaler.joblib')

with open('data/default_values.json', 'r') as json_file:
    REQUIRED_KEYS = json.load(json_file)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    
    # Ensure all required keys are present
    for key, default_value in REQUIRED_KEYS.items():
        if key not in data:
            data[key] = default_value

    # Convert JSON data to DataFrame
    df = pd.DataFrame([data])
    
    # Perform data cleansing
    cleansed_df = cleansing.clean_data(df.copy())
    
    # Perform data preprocessing
    preprocessed_data = preprocessing.apply_preprocessing(cleansed_df.copy())
    ml_data = preprocessed_data.drop('customerID', axis=1)
    
    # Apply the scaler
    ml_data[['tenure', 'MonthlyCharges']] = scaler.transform(ml_data[['tenure', 'MonthlyCharges']])
    
    # Predict churn and calculate probabilities
    X = ml_data.drop('Churn', axis=1)
    y_pred = model.predict(X)
    churn_probability = model.predict_proba(X)[:, 1]
    
    # Create response
    response = {
        'Predicted': int(y_pred[0]),
        'ChurnScore': round(churn_probability[0] * 100, 0)  # Convert to a score out of 100
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)


# In[ ]:




