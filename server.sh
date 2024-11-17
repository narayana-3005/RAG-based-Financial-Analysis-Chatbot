#!/bin/bash

# Fetch PostgreSQL URI from GCP Secret Manager
POSTGRESQL_URL=$(gcloud secrets versions access latest --secret=mlflow-db-secret)

# Fetch Storage URL from GCP Secret Manager
STORAGE_URL=$(gcloud secrets versions access latest --secret=mlflow-bucket-db)

# Upgrade the MLflow database
mlflow db upgrade "$POSTGRESQL_URL"

# Start the MLflow server
mlflow server \
  --host 0.0.0.0 \  # Use 0.0.0.0 instead of 127.0.0.1 to allow external access in Cloud Run
  --port ${PORT:-5001} \  # Default to 5001 but allow Cloud Run to override with $PORT
  --backend-store-uri "$POSTGRESQL_URL" \
  --artifacts-destination "$STORAGE_URL"
