#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import pandas as pd
import argparse
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define mappings
gender_mapping = {'Male': 1, 'Female': 2}
yes_no_mapping = {'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}
internet_service_map = {'Fiber optic': 2, 'DSL': 1, 'No': 0}
contract_map = {'Month-to-month': 1, 'One year': 2, 'Two year': 3}
payment_method_map = {'Credit card (automatic)': 4, 'Bank transfer (automatic)': 3, 'Electronic check': 2, 'Mailed check': 1}

# Function to log and handle unmapped values
def handle_unmapped_values(column, mapping):
    def mapper(value):
        if value not in mapping:
            logging.warning(f"Unmapped value detected in column '{column}': {value}")
            return -1  # Assign a default value for unmapped entries
        return mapping[value]
    return mapper

def apply_preprocessing(df):
    """
    Apply mappings to the DataFrame and log any unmapped values.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with mappings applied.
    """
    try:
        df['gender'] = df['gender'].map(handle_unmapped_values('gender', gender_mapping))
        for column in ['SeniorCitizen','Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']:
            df[column] = df[column].map(handle_unmapped_values(column, yes_no_mapping))

        df['InternetService'] = df['InternetService'].map(handle_unmapped_values('InternetService', internet_service_map))
        df['Contract'] = df['Contract'].map(handle_unmapped_values('Contract', contract_map))
        df['PaymentMethod'] = df['PaymentMethod'].map(handle_unmapped_values('PaymentMethod', payment_method_map))
        
        ## features to be used
        cols = cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                'MonthlyCharges', 'Churn']
        
        df = df[cols]

        logging.info("Successfully applied mappings to the DataFrame")
        return df
    except Exception as e:
        logging.error(f"Error applying mappings: {e}")
        raise

def execute_data_preprocessing(input_file, output_file):
    """
    Executes the data preprocessing steps.
    
    Parameters:
    input_file (str): The path to the input CSV file.
    output_file (str): The path to the output CSV file.
    
    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    try:
        df = pd.read_csv(input_file)
        preprocessed_df = apply_mappings(df)
        preprocessed_df.to_csv(output_file, index=False)
        logging.info(f"Data preprocessing process completed successfully and saved to {output_file}")
        return preprocessed_df
    except Exception as e:
        logging.error(f"Error in data preprocessing process: {e}")
        raise

if __name__ == '__main__':
    if 'ipykernel' in sys.modules or hasattr(sys, 'ps1'):
        # Interactive mode (e.g., Jupyter notebook)
        input_file = input("Please enter the path to the input CSV file: ")
        output_file = input("Please enter the path to the output CSV file: ")
        execute_data_preprocessing(input_file, output_file)
    else:
        # Command-line execution
        parser = argparse.ArgumentParser(description="Data Preprocessing Script")
        parser.add_argument('input_file', help='Path to the input CSV file')
        parser.add_argument('output_file', help='Path to the output CSV file')
        args = parser.parse_args()
        execute_data_preprocessing(args.input_file, args.output_file)

