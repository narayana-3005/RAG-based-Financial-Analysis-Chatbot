import streamlit as st
from google.cloud import aiplatform
import json
import requests

# Initialize Google Cloud AI Platform
aiplatform.init()

# Your GCP token (Directly included)
GCP_TOKEN = "theta-function-429605-j0-7e0753216ae2.json"

# Define LLM query function with error handling
def query_llm(prompt):
    try:
        # Replace with your model endpoint
        endpoint = "your-model-endpoint"
        headers = {"Authorization": f"Bearer {GCP_TOKEN}"}
        payload = {"instances": [{"prompt": prompt}]}
        response = requests.post(endpoint, headers=headers, json=payload)
        
        # Check if response is successful
        if response.status_code == 200:
            return response.json()["predictions"][0]["text"]
        else:
            # Log the error details
            st.error(f"LLM query failed with status code: {response.status_code}")
            return "Hmm, looks like my brain took a coffee break ☕. Try again in a bit!"
    except Exception as e:
        # Log the exception
        st.error(f"An error occurred: {e}")
        return "Yikes! I tripped over a wire 🤖💥. Please try again soon!"

# Streamlit UI
st.title("Financial News & Stock Price Chatbot")
st.sidebar.header("Chatbot Options")
st.write("👋 Welcome! Ask me about financial news or stock prices.")

user_input = st.text_input("Enter your query:", placeholder="Type here...")
if st.button("Send"):
    if user_input:
        with st.spinner("Processing..."):
            response = query_llm(user_input)
        st.success("Response:")
        st.write(response)
    else:
        st.warning("Please enter a query.")