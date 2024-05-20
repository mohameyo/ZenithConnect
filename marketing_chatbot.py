#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
from groq import Groq
import requests
import gradio as gr
import time
import logging
from langchain.schema import AIMessage, HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize OpenAI client
client = Groq(api_key = "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")
MODEL = 'llama3-70b-8192'

# Define the churn prediction function
def predict_churn(data: dict) -> dict:
    url = 'http://127.0.0.1:5000/predict'
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        return {
            "error": f"Request failed: {str(e)}",
            "details": response.text if response else "No response text available"
        }

# Initialize chat history
chat_history = []

# Define the provide_marketing_advice function
def provide_marketing_advice(data: dict) -> dict:
    # This is a placeholder function that returns a generic marketing advice response
    # In a real-world scenario, I would implement actual logic to provide advice maybre RAG arch or fine-tuned model.
    logging.info("Providing marketing advice.")
    return {
        "advice": "Based on the latest trends in the telecom industry in Saudi, we recommend focusing on personalized customer experiences, improving customer service, and offering competitive pricing and promotions to retain customers."
    }

# Define the conversation function
def run_conversation(message):
    global chat_history

    # General marketing prompt
    pre_prompt = """
    You are a marketing assistant. You work for Zenith Connect. Your role is to assist the marketing team by providing data-driven insights to help understand customer behavior and improve customer retention. You can answer questions about customer churn prediction, provide marketing advice, greet the team, or handle any general marketing-related queries. Make sure to provide personalized and helpful responses.

    For customer churn prediction, the marketing team can provide:
    - customerID:
    - gender: Male, Female
    - SeniorCitizen: 0, 1
    - Partner: Yes, No
    - Dependents: Yes, No
    - tenure: integer (number of months) 
    - PhoneService: Yes, No
    - MultipleLines: Yes, No, No phone service
    - InternetService: DSL, Fiber optic, No
    - OnlineSecurity: Yes, No, No internet service
    - OnlineBackup: Yes, No, No internet service
    - DeviceProtection: Yes, No, No internet service
    - TechSupport: Yes, No, No internet service
    - StreamingTV: Yes, No, No internet service
    - StreamingMovies: Yes, No, No internet service
    - Contract: Month-to-month, One year, Two year
    - PaperlessBilling: Yes, No
    - PaymentMethod: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) (default = Electronic check)
    - MonthlyCharges: Float (monthly charge amount)
    - TotalCharges: Float (total charge amount)
    - Churn: Yes, No

    If the user is asking about customer churn, output the data as a JSON object to be used as input for a machine learning model. Otherwise, provide general marketing advice or handle any other marketing-related queries.
    """

    messages = [
        {"role": "system", "content": pre_prompt},
        {"role": "user", "content": message['input']}
    ]

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "predict_churn",
                "description": "Get the churn score and prediction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "The customer data for churn prediction",
                        }
                    },
                    "required": ["data"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "provide_marketing_advice",
                "description": "Provide general marketing advice",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Any additional data needed for advice",
                        }
                    },
                    "required": ["data"]
                }
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=8000
        )

        response_message = response.choices[0].message
        tool_calls = getattr(response_message, 'tool_calls', [])

        if tool_calls:
            available_functions = {
                "predict_churn": predict_churn,
                "provide_marketing_advice": provide_marketing_advice,
            }
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(data=function_args.get("data"))
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response)
                })
            
            second_response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            chat_history.append((message['input'], second_response.choices[0].message.content))
            return {"output": second_response.choices[0].message.content}
        else:
            chat_history.append((message['input'], response_message.content))
            return {"output": response_message.content}
    except Exception as e:
        logging.error(f"An error occurred during conversation: {e}")
        return {"output": f"An error occurred: {str(e)}"}

# Function to handle prediction and conversation
def predict(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = run_conversation({"input": message})
    
    response_message = ""
    for character in gpt_response["output"]:
        response_message += character
        time.sleep(0.01)
        yield response_message

# Create a Gradio interface
theme = gr.themes.Base(primary_hue="green")
chat_interface = gr.ChatInterface(predict, autofocus=False, theme=theme)

# Enable the queue
chat_interface.queue()

# Launch the Gradio interface
chat_interface.launch()

