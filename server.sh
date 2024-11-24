#!/bin/bash

# Debug: Check gcloud installation
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed or not in PATH."
    exit 1
fi

# Debug: Check mlflow installation
if ! command -v mlflow &> /dev/null; then
    echo "Error: mlflow is not installed or not in PATH."
    exit 1
fi

# Fetch PostgreSQL URI from GCP Secret Manager
POSTGRESQL_URL=$(gcloud secrets versions access latest --secret=mlflow-db-secret)

# Fetch Storage URL from GCP Secret Manager
STORAGE_URL=$(gcloud secrets versions access latest --secret=mlflow-bucket-db)

# Debugging: Print variables (remove in production)
echo "POSTGRESQL_URL: $POSTGRESQL_URL"
echo "STORAGE_URL: $STORAGE_URL"
echo "MLFLOW_SECRET_KEY: $MLFLOW_SECRET_KEY"

# Upgrade the MLflow database
mlflow db upgrade "$POSTGRESQL_URL"

# Start the MLflow server
mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  --backend-store-uri "$POSTGRESQL_URL" \
  --artifacts-destination "$STORAGE_URL"
