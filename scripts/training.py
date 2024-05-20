#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import logging
import argparse
import joblib
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_logistic_regression(X_train, y_train, full = False):
    """
    Train a logistic regression model using grid search.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training labels.

    Returns:
    LogisticRegression: The best logistic regression model.
    """
    try:
        
        ## Do scaler
        
        # Standardize the numerical features
        scaler = StandardScaler()
        X_train[['tenure', 'MonthlyCharges']] = scaler.fit_transform(X_train[['tenure', 'MonthlyCharges']])
        
        logging.info("Data Scaling completed successfully")
        
        logistic_regression = LogisticRegression(max_iter=1500)
        params = {
            'C': [0.01, 0.1, 0.3, 0.5, 1, 10],
            'penalty': ['l1'],
            'solver': ['liblinear'],
            'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 4}],
            'fit_intercept': [True]
        }

        grid_search = GridSearchCV(logistic_regression, {k: v for k, v in params.items()},
                                   scoring='recall', cv=5, n_jobs=-1, error_score=np.nan)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        
        logging.info("Logistic regression training completed successfully")
        return best_model, scaler
    except Exception as e:
        logging.error(f"Error training logistic regression model: {e}")
        raise
        
def train_full_df(df, model_parameters):
    
    df = df.drop('customerID', axis = 1)
    scaler = StandardScaler()
    X_train = df.drop('Churn', axis = 1)
    y_train = df['Churn']
    X_train[['tenure', 'MonthlyCharges']] = scaler.fit_transform(X_train[['tenure', 'MonthlyCharges']])
        
    logging.info("Data Scaling completed successfully")
    model = LogisticRegression(**model_parameters)
    model.fit(X_train, y_train)
    
    logging.info("Fit on all dataset")
    
    return model, scaler


def train_and_return(X_train, y_train, X_test, y_test):
    """
    Function to train the model and return the model and scaler along with evaluation metrics.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    tuple: The trained model, scaler, and evaluation metrics.
    """
    best_model, scaler = train_logistic_regression(X_train, y_train)
    X_test[['tenure', 'MonthlyCharges']] = scaler.transform(X_test[['tenure', 'MonthlyCharges']])

    # Evaluate the model on the test set
    y_test_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    logging.info(f"Test Accuracy: {accuracy}")
    logging.info(f"Test Recall: {recall}")
    logging.info(f"Test Precision: {precision}")
    logging.info(f"Test F1 Score: {f1}")

    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1
    }

    return best_model, metrics, scaler


def main(input_file1, inputfile2):
    """
    Main function to load data, preprocess, train the model, and save the model and scaler.

    Parameters:
    input_file (str): Path to the preprocessed data file (CSV).
    model_output (str): Path to save the trained model (joblib).
    """
    try:
        training_data = pd.read_csv(input_file1)
        testing_data = pd.read_csv(input_file2)
        best_model, metrics, scaler = train_and_return(training_data.drop('Churn', axis =1),
                                                       training_data['Churn'],
                                                       testing_data.drop('Churn', axis =1),
                                                       testing_data['Churn'])
        logging.info(f"Training completed and model saved to {model_output}, scaler saved to {scaler_output}")
    except Exception as e:
        logging.error(f"Error in main training process: {e}")
        raise

if __name__ == '__main__':
    if 'ipykernel' in sys.modules or hasattr(sys, 'ps1'):
        # Interactive mode (e.g., Jupyter notebook)
        input_file1 = input("Please enter the path to the input CSV file: ")
        input_file2 = input("Please enter the path to the input CSV file: ")
        main(input_file1, input_file2)
    else:
        # Command-line execution
        parser = argparse.ArgumentParser(description="Model Training Script")
        parser.add_argument('input_file1', help='Path to the preprocessed data file (CSV)')
        parser.add_argument('input_file2', help='Path to the preprocessed data file (CSV)')
        args = parser.parse_args()

        # Execute the training process
        main(args.input_file1, args.input_file2)

