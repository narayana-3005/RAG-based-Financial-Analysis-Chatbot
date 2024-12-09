import pandas as pd
from google.cloud import storage
from datetime import datetime
import uuid
import os

def log_predictions(inputs, predictions, true_labels=None):
    # Handle missing true_labels
    true_labels = true_labels if true_labels is not None else "N/A"

    # Prepare data
    data = {
        "inputs": inputs,
        "predictions": predictions,
        "true_labels": true_labels,
        "timestamp": datetime.utcnow()
    }
    df = pd.DataFrame([data])

    # Save to Google Cloud Storage
    bucket_name = os.getenv("MONITORING_BUCKET_NAME", "model-monitoring-logs")  # Dynamic bucket name
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Check if bucket exists
    if not bucket.exists():
        bucket = storage_client.create_bucket(bucket_name)
        print(f"Created new bucket: {bucket_name}")

    # Unique file name
    file_name = f"logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.csv"
    blob = bucket.blob(file_name)

    try:
        # Upload file
        blob.upload_from_string(df.to_csv(index=False), "text/csv")
        print(f"Log successfully uploaded: {file_name}")
    except Exception as e:
        print(f"Error uploading log to GCS: {e}")
