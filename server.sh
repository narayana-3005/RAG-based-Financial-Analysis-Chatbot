#!/bin/bash

# Fetch PostgreSQL URI from GCP Secret Manager
POSTGRESQL_URL=$(gcloud secrets versions access latest --secret=mlflow-db-secret)

# Fetch Storage URL from GCP Secret Manager
STORAGE_URL=$(gcloud secrets versions access latest --secret=mlflow-bucket-db)

# Use the MLFLOWSECRETKEY from the environment (debugging: remove echo in production)
echo "MLFLOW_SECRET_KEY: $MLFLOW_SECRET_KEY"

# Upgrade the MLflow database
mlflow db upgrade "$POSTGRESQL_URL"

# Start the MLflow server
mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  --backend-store-uri "$POSTGRESQL_URL" \
  --artifacts-destination "$STORAGE_URL"
