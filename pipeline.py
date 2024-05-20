#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scripts import cleansing, preprocessing, splitting, training, predict
import logging
import yaml
import json
import joblib


# In[2]:


# Set up logging
logging.basicConfig(level=logging.INFO)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Step 1: Load Training Data
    data = cleansing.read_csv_file(config['train_data_path'])
    
    # Step 2: Data Cleansing
    logging.info("Starting data cleansing...")
    cleansed_data = cleansing.clean_data(data.copy())
    # save most repeated to be default in our app
    default_values = {col: cleansing.most_repeated_or_average(cleansed_data[col]) for col in cleansed_data.columns}
    with open('data/default_values.json', 'w') as json_file:
        json.dump(default_values, json_file)
    
    # Step 3: Data Preprocessing
    logging.info("Starting data preprocessing...")
    preprocessed_data = preprocessing.apply_preprocessing(cleansed_data.copy())
    
    # Step 4: Data Splitting
    logging.info("Starting data splitting...")
    X_train, y_train, X_test, y_test = splitting.split_data(preprocessed_data.copy())
    
    # Step 5: Training & Testing the Model & get the best parameters
    logging.info("Starting model training...")
    model, metrics, _ = training.train_and_return(X_train, y_train, X_test, y_test)
    
    # save the model parameters
    model_params = model.get_params()
    config['model_parameters'] = model_params

    # Step 5: Train on the whole dataset and save the scaler
    model, scaler = training.train_full_df(preprocessed_data.copy(), model_params)
    
    joblib.dump(model, config['model_path'])
    joblib.dump(scaler, config['scaler_path'])
    
    logging.info("Finished Saved Model and scaler")
    
    logging.info("Predict on new data")
    final_predictions = predict.fit(config['test_data_path'], model, scaler)
    
    return final_predictions


# In[3]:


if __name__ == "__main__":

    config_path = 'config.yaml'
    final_results = main(config_path)
    final_results.to_csv('data/model_results.csv', index = False)

