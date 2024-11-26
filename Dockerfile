# Base image
FROM python:3.9.18-slim

# Set working directory
WORKDIR /mlflow

# Install necessary dependencies
RUN pip install mlflow psycopg2-binary google-cloud-storage

# Environment variable for artifact storage
ENV DEFAULT_ARTIFACT_ROOT="gs://mlflow_connection_bucket/artifacts"

# Expose the MLflow server port
EXPOSE 8080

# Command to run MLflow server
CMD mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 8080
