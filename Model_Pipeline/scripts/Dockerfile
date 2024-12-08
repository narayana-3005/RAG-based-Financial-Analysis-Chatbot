# Use a slim Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json"

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy application code and configuration
COPY ./Model_Pipeline/scripts /app/scripts
COPY ./Model_Pipeline/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the Flask app's listening port
EXPOSE 8085

# Set the entrypoint to run the Flask app directly
CMD ["python", "scripts.flask_app"]
