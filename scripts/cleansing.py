#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import logging
import argparse
import json
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_csv_file(file_path):
    """
    Reads a CSV file and returns a DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Successfully read the file: {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"No data: {file_path}")
        raise
    except pd.errors.ParserError:
        logging.error(f"Parsing error: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading file: {file_path} - {e}")
        raise
        
# Function to calculate the most repeated value or average this will be used in the final script only
def most_repeated_or_average(column):
    if column.dtype == 'object':  # Check if the column is categorical
        return column.mode()[0]  # Return the most frequent value
    else:  # Numeric column
        return column.mean()  # Return the average value

def clean_data(data):
    """
    Fills missing values in the DataFrame.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """
    try:
        data['SeniorCitizen'] = data['SeniorCitizen'].map({0: 'No', 1 : 'Yes'})
        data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        data = data.replace('', np.nan)
        data = data.replace(np.nan, 0)
        logging.info("Successfully filled missing values")
        
        return data
    except Exception as e:
        logging.error(f"Error filling missing values: {e}")
        raise

def clean(input_file):
    """
    Executes the data cleansing steps: reading the CSV file and filling missing values.

    Parameters:
    input_file (str): The path to the input CSV file.

    Returns:
    pd.DataFrame: The cleansed DataFrame.
    """
    try:
        data = read_csv_file(input_file)
        data = fill_missing_values(data)
        logging.info("Data cleaning completed successfully")
        return data
    except Exception as e:
        logging.error(f"Error in data cleansing process: {e}")
        raise

def main(input_file, output_file):
    """
    Main function to clean data and save to an output file.

    Parameters:
    input_file (str): The path to the input CSV file.
    output_file (str): The path to the output CSV file.
    """
    cleaned_data = clean(input_file)
    cleaned_data.to_csv(output_file, index=False)
    logging.info(f"Cleansed data saved to {output_file}")

if __name__ == '__main__':
    if 'ipykernel' in sys.modules or hasattr(sys, 'ps1'):
        # Interactive mode (e.g., Jupyter notebook)
        input_file = input("Please enter the path to the input CSV file: ")
        output_file = input("Please enter the path to the output CSV file: ")
        main(input_file, output_file)
    else:
        # Command-line execution
        parser = argparse.ArgumentParser(description="Data Cleansing Script")
        parser.add_argument('input_file', help='Input data file (CSV)')
        parser.add_argument('output_file', help='Output cleansed data file (CSV)')
        args = parser.parse_args()
        main(args.input_file, args.output_file)

