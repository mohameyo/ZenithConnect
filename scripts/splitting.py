#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import logging
import argparse
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_data(df):
    """
    Split the data into training and test sets and standardize numerical features.

    Parameters:
    input_file (str): Path to the input data file (CSV).
    output_train_file (str): Path to save the training data (CSV).
    output_test_file (str): Path to save the test data (CSV).
    scaler_file (str): Path to save the scaler (joblib).

    Returns:
    None
    """
    try:
        df = df.drop('customerID', axis=1)
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # Combine X and y for saving to CSV
        train_data = X_train.copy()
        train_data['Churn'] = y_train
        test_data = X_test.copy()
        test_data['Churn'] = y_test


        logging.info("Data splitting completed successfully")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error splitting and scaling data: {e}")
        raise

def split_data_test(input_file, output_train_file, output_test_file):
    """
    Split the data into training and test sets and standardize numerical features.

    Parameters:
    input_file (str): Path to the input data file (CSV).
    output_train_file (str): Path to save the training data (CSV).
    output_test_file (str): Path to save the test data (CSV).
    scaler_file (str): Path to save the scaler (joblib).

    Returns:
    None
    """
    try:
        df = pd.read_csv(input_file)
        df = df.drop('customerID', axis=1)
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        # Combine X and y for saving to CSV
        train_data = X_train.copy()
        train_data['Churn'] = y_train
        test_data = X_test.copy()
        test_data['Churn'] = y_test

        # Save the split and scaled data to CSV files
        train_data.to_csv(output_train_file, index=False)
        test_data.to_csv(output_test_file, index=False)

        logging.info("Data splitting completed successfully")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logging.error(f"Error splitting and scaling data: {e}")
        raise

if __name__ == '__main__':
    if 'ipykernel' in sys.modules or hasattr(sys, 'ps1'):
        # Interactive mode (e.g., Jupyter notebook)
        input_file = input("Please enter the path to the input CSV file: ")
        output_train_file = input("Please enter the path to save the training data CSV file: ")
        output_test_file = input("Please enter the path to save the test data CSV file: ")
        split_data_test(input_file, output_train_file, output_test_file)
    else:
        # Command-line execution
        parser = argparse.ArgumentParser(description="Data Splitting and Scaling Script")
        parser.add_argument('input_file', help='Path to the input data file (CSV)')
        parser.add_argument('output_train_file', help='Path to save the training data (CSV)')
        parser.add_argument('output_test_file', help='Path to save the test data (CSV)')
        args = parser.parse_args()

        # Execute the data splitting and scaling process
        split_data_test(args.input_file, args.output_train_file, args.output_test_file)

