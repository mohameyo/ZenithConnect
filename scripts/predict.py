#!/usr/bin/env python
# coding: utf-8

# In[2]:


from scripts import cleansing, preprocessing
import logging
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
import pandas as pd


# In[4]:


def fit(file_path, model, scaler):
    
    data = cleansing.read_csv_file(file_path)
    cleansed_data = cleansing.clean_data(data.copy())
    preprocessed_data = preprocessing.apply_preprocessing(cleansed_data.copy())
    
    idx = preprocessed_data['customerID']
    ml_data = preprocessed_data.drop('customerID', axis=1).copy()
    
    
    ml_data[['tenure', 'MonthlyCharges']] = scaler.transform(ml_data[['tenure', 'MonthlyCharges']])
    logging.info(f"Successfully scaled new  data")
        
    X = ml_data.drop('Churn', axis = 1)
    y_actual = ml_data['Churn']
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred) 
    f1 = f1_score(y_actual, y_pred)
    report = classification_report(y_actual, y_pred)
    
    # Log metrics
    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1}")
    logging.info("\nClassification Report:\n" + report)
    logging.info('Done!')
    
    return pd.DataFrame({'customerID': idx, 'Predicted': y_pred, 'Probability': y_proba})

