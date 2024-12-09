# Use a lightweight Python image as the base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json"

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy application files and dependencies
COPY ./Model_Pipeline/scripts /app/scripts
COPY ./Model_Pipeline/scripts/requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Flask app
EXPOSE 8085

# Set the entrypoint command to start the Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8085", "--timeout", "300", "scripts.flask_app:app"]
