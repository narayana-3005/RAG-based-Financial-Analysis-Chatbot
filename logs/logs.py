import logging
import os
from google.cloud import storage
from datetime import datetime
import threading
import time

# Configure logging
LOCAL_LOG_FILE = "/tmp/application.log"  # Use a temp directory or a configurable path
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "logs_buck")

os.makedirs(os.path.dirname(LOCAL_LOG_FILE), exist_ok=True)  # Ensure directory exists

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOCAL_LOG_FILE),  # Write logs to file
        logging.StreamHandler()  # Output logs to console
    ]
)

logger = logging.getLogger(__name__)

# Function to upload logs to GCS
def upload_logs_to_gcs(local_file_path, bucket_name):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        current_time = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        destination_blob_name = f"logs/{current_time}_application.log"

        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        logger.info(f"Uploaded logs to GCS: {destination_blob_name}")
    except Exception as e:
        logger.error(f"Failed to upload logs to GCS: {e}")

# Periodic log upload
def periodic_log_upload(interval=3600):
    while True:
        upload_logs_to_gcs(LOCAL_LOG_FILE, BUCKET_NAME)
        time.sleep(interval)

# Flask app
from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    logger.info("Health check endpoint accessed.")
    return jsonify({"status": "healthy"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        logger.info("Prediction endpoint accessed.")
        result = {"prediction": "dummy result"}
        logger.info("Prediction successful.")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    try:
        logger.info("Starting Flask application...")
        
        # Validate GCS bucket
        upload_logs_to_gcs(LOCAL_LOG_FILE, BUCKET_NAME)
        
        # Start periodic log uploads in the background
        threading.Thread(target=periodic_log_upload, daemon=True).start()

        # Start Flask app
        app.run(host="0.0.0.0", port=8085)
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
