#!/bin/bash

# Fetch PostgreSQL URI from GCP Secret Manager
POSTGRESQL_URL=$(gcloud secrets versions access latest --secret=mlflow-db-secret)
STORAGE_URL=$(gcloud secrets versions access latest --secret=mlflow-bucket-db)

# Set the default port to 8080 if not provided
PORT=${PORT:-8080}

# Upgrade the MLflow database
mlflow db upgrade "$POSTGRESQL_URL"

# Start the MLflow server on the specified port
mlflow server \
  --host 0.0.0.0 \
  --port $PORT \
  --backend-store-uri "$POSTGRESQL_URL" \
  --artifacts-destination "$STORAGE_URL"
