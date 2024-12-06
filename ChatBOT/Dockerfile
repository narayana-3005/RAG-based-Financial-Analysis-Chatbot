# Use a lightweight Python image as the base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for the Streamlit app
EXPOSE 8080

# Set the entrypoint command to start the Streamlit app
CMD ["streamlit", "run", "chatbotUI.py", "--server.port=8080", "--server.address=0.0.0.0"]
