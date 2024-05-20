# ZenithConnect AI-Powered Marketing Agent

## Purpose

The purpose of this project is to create an AI-powered Large Language Model (LLM) for ZenithConnect that interacts with a churn model API. This innovative solution empowers the marketing team to interact with the model without needing to specify exact values. The power of LLM and semantics will map inputs to the right definitions and obtain model predictions. Additionally, we have integrated extra tools for the marketing team to interact with the LLM, receive marketing advice based on ZenithConnect's business, create business plans, and explore Saudi Arabia's telecom market trends. This comprehensive agent provides a full suite of marketing functionalities.

## Repository Overview

### Data Folder

The `data` folder contains all the training and testing data, as well as the top churn clients based on their Customer Lifetime Value (CLTV) and probability to churn. This data is essential for ZenithConnect to conduct extensive marketing campaigns with optimized costs.

### Scripts Folder

The `scripts` folder includes production-level scripts for data cleansing, preprocessing, training, and testing. Additionally, it contains a statistics file used in our notebooks for hypothesis testing and visualizations.

### Models Folder

The `models` folder holds our final machine learning model, specifically a logistic regression model with its parameters and the scaler used. These models are integral to our main application files.

### Notebooks Folder

The `notebooks` folder is a comprehensive storytelling resource for all projects:
- **ML Part:** Details the machine learning process.
- **LLM Part:** Demonstrates outputs from the chatbot.
- **Visualization:** Shows the process of creating HTML files for visualizations.

### Visualization Folder

The `visualization` folder provides outputs that enable the marketing team to visualize customer demographics, data distribution, and churn contributors. Additionally, it offers a 360-degree overview of ZenithConnect's clients, allowing comparison of individual churn scores and CLTV against client averages. This helps in determining whether the cost to retain a customer is justified.

### Main Scripts

- **pipeline.py:** This script handles the end-to-end machine learning cycle, from data preprocessing to training, evaluating, and saving the model and scalers. It ensures that new data can be processed and predictions can be made.
  
- **app.py:** This script creates the ML API endpoint using Flask. The API is used to serve predictions from the trained model.

- **marketing_chatbot.py:** This script runs and interacts with the Llama 3 open-source model through a Gradio interface chatbot. The chatbot allows marketing interactions and predictions, providing an interface for marketing-specific tasks.

## Getting Started

Follow these instructions to set up and run the project on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed on your machine:
- Python 3.6 or higher
- `pip` (Python package installer)

### Installation

1. **Clone the repository:**

   ```sh
   git clone https://github.com/mohameyo/ZenithConnect.git
   cd ZenithConnect

Certainly! Here are the complete steps for setting up and running your project, including using the `--no-cache-dir` option for installing dependencies. You can copy and paste this content into your `README.md` file.

2. **Create a virtual environment:**

   For Unix or macOS:

   ```sh
   python3 -m venv env
   source env/bin/activate
   ```

   For Windows:

   ```sh
   python -m venv env
   .\env\Scripts\activate
   ```

3. **Install the dependencies:**

   ```sh
   pip install --no-cache-dir -r requirements.txt
   ```

### Running the Application

1. **Run the end-to-end ML pipeline:**

   ```sh
   python pipeline.py
   ```

2. **Start the ML API Endpoint:**

   ```sh
   python app.py
   ```
Open a new Command Prompt and

3. **Run the Marketing Chatbot:**

   ```sh
   python marketing_chatbot.py
   ```
   Copy the link that appears in the command prompt to open the Gradio library and happy LLMing!

### Additional Information

- **Deactivate the virtual environment:** When you are done, you can deactivate the virtual environment by running:

  ```sh
  deactivate
  ```

- **Updating dependencies:** If you add new dependencies, update the `requirements.txt` file by running:

  ```sh
  pip freeze > requirements.txt
  ```

### Troubleshooting

- If you encounter issues with package installation, ensure that your `pip` is up to date:

  ```sh
  pip install --upgrade pip
  ```

- If you experience network-related issues, try using a different PyPI mirror:

  ```sh
  pip install -r requirements.txt -i https://pypi.org/simple
  ```

This is the end of the project!
