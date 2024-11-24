#!/bin/bash

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed or not in PATH."
    exit 1
fi

# Check if mlflow is installed
if ! command -v mlflow &> /dev/null; then
    echo "Error: mlflow is not installed or not in PATH."
    exit 1
fi

# Fetch secrets from GCP Secret Manager
POSTGRESQL_URL=$(gcloud secrets versions access latest --secret=mlflow-db-secret)
STORAGE_URL=$(gcloud secrets versions access latest --secret=mlflow-bucket-db)

# Print secrets for debugging (optional, remove in production)
echo "POSTGRESQL_URL: $POSTGRESQL_URL"
echo "STORAGE_URL: $STORAGE_URL"

# Set the default port to 8080 if not provided
PORT=${PORT:-8080}

# Upgrade the MLflow database
mlflow db upgrade "$POSTGRESQL_URL"

# Start the MLflow server
mlflow server \
  --host 0.0.0.0 \
  --port $PORT \
  --backend-store-uri "$POSTGRESQL_URL" \
  --artifacts-destination "$STORAGE_URL"
