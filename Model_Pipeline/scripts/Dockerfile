# Use a slim Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1


# Set environment variables for Google Cloud authentication
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json"

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*


# Set the working directory
WORKDIR /app

# Copy application code and configuration
COPY ./Model_Pipeline/scripts /app/scripts
COPY ./Model_Pipeline/requirements.txt /app/requirements.txt


# Copy the service account key into the container
COPY ./Model_Pipeline/scripts/theta-function-429605-j0-7e0753216ae2.json /app/credentials.json



# Expose the default Flask port
EXPOSE 8080


# Set the entrypoint command to start the Flask app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "300", "scripts.flask_app:app"]
