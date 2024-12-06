import streamlit as st
import requests
import os

# Flask app URL
FLASK_APP_URL = "https://flask-app-184982369838.us-east1.run.app/predict"

# Define the function to query the Flask app
def query_flask_app(prompt):
    try:
        # Payload to send to the Flask app
        payload = {"query": prompt}
        
        # Send POST request
        response = requests.post(FLASK_APP_URL, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Attempt to parse the JSON response
            response_data = response.json()
            if "answer" in response_data:  # Check for 'answer' key in the response
                return response_data["answer"]
            else:
                # Debugging: Show the full response content if 'answer' key is missing
                st.error(f"Unexpected response format: {response_data}")
                return "Oops! I got an unexpected response from the server. 🤔"
        else:
            # Log the status code and response content for debugging
            st.error(f"Flask app query failed with status code: {response.status_code}")
            st.error(f"Response content: {response.text}")
            return "Hmm, the server seems to be snoozing. Try again later! 😴"
    except Exception as e:
        # Log the exception details
        st.error(f"An error occurred: {e}")
        return "Yikes! Something went wrong while contacting the server. 🚨"

# Multi-page navigation
def main():
    # Page state management
    if "page" not in st.session_state:
        st.session_state.page = "home"  # Default to the home page

    # Home Page
    if st.session_state.page == "home":
        st.title("Welcome to Your Financial Assistant Bot! 💸")
        st.write("""
        ### Meet Your Intelligent Financial Assistant:
        🚀 Our financial bot is designed to simplify your interactions with financial data. 
        With cutting-edge AI capabilities, it can:
        - **Analyze trends** in stock prices.
        - **Summarize financial news** for faster decision-making.
        - **Provide market insights** in seconds.

        Whether you're a trader, an investor, or simply curious about the market, this bot is here to assist you.
        
        🤖 **Why this bot?**
        - Always up-to-date with the latest financial trends.
        - Simple, intuitive, and lightning-fast responses.
        - Built on state-of-the-art AI technology to give you the edge in financial insights.
        """)
        st.write("Click below to start exploring!")
        
        # "Try Me" button
        if st.button("Try Me"):
            st.session_state.page = "chatbot"

    # Chatbot Page
    elif st.session_state.page == "chatbot":
        st.title("Financial News & Stock Price Chatbot")
        #st.sidebar.header("Chatbot Options")
        st.write("👋 Welcome! Ask me about financial news or stock prices.")

        # User input
        user_input = st.text_input("Enter your query:", placeholder="Type here...")

        # Handle query
        if st.button("Send"):
            if user_input:
                with st.spinner("Processing..."):
                    response = query_flask_app(user_input)
                st.success("Response:")
                st.write(response)

                # Mock-up feedback buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.button("👍 Thumbs Up")
                with col2:
                    st.button("👎 Thumbs Down")
            else:
                st.warning("Please enter a query.")

# Run the app
if __name__ == "__main__":
    main()
