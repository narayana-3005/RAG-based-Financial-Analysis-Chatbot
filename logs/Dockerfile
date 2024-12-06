# Use a lightweight Python image as the base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set environment variables for Google Cloud authentication
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json"

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the logs.py and requirements.txt files into the container
COPY logs.py /app/logs.py
COPY requirements.txt /app/requirements.txt

# Copy the service account key into the container
COPY ../../Model_Pipeline/scripts/theta-function-429605-j0-7e0753216ae2.json /app/credentials.json

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for the Flask app
EXPOSE 8085

# Set the entrypoint command to start the Flask app
CMD ["python", "logs.py"]
