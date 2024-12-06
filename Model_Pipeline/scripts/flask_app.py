# !pip install transformers datasets faiss-cpu google-cloud-storage
from flask import Flask, request, jsonify
from transformers import (
    DPRContextEncoder,
    DPRQuestionEncoder,
    DPRReader,
    DPRReaderTokenizer,
)
import logging
import torch
import faiss
import numpy as np
import logging
import os
import io
import pyarrow.parquet as pq
from google.cloud import storage
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

def load_documents_data_from_gcs(bucket_name='news_data_bucket'):
  """Loads a news data from Google Cloud Storage.

  Args:
    bucket_name: The name of the GCS bucket.

  Returns:
    The loaded documents_data.
  """
  temp_file = 'documents_data.json'
  storage_client = storage.Client()
  data_blobs = storage_client.list_blobs(bucket_name)
  data_blob = next(data_blobs)
  data_blob.download_to_filename(temp_file)
  with open(temp_file, 'r') as f:
        documents_data = json.load(f)
  os.remove(temp_file)
  return documents_data

def load_index_from_gcs(bucket_name='index_faiss'):
  """Loads a Faiss index from Google Cloud Storage.

  Args:
    bucket_name: The name of the GCS bucket.
    index_file_name: The name of the index file within the bucket.

  Returns:
    The loaded Faiss index.
  """
  temp_file = 'index.faiss'
  storage_client = storage.Client()
  index_blobs = storage_client.list_blobs(bucket_name)
  index_blob = next(index_blobs)
  index_blob.download_to_filename(temp_file)
  index = faiss.read_index(temp_file)
  os.remove(temp_file)
  return index

# Function to retrieve the top-k relevant documents based on the query
def retrieve_context(query, index, documents, k=5):
    """Retrieve top-k documents relevant to the query."""
    logger.info("Method: retrieve_context execution started.")
    # Encode the query with the question encoder
    inputs = reader_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        query_embedding = question_encoder(**inputs).pooler_output.cpu().numpy()

    # Perform the search on the FAISS index
    D, I = index.search(query_embedding, k)  # D = distances, I = indices of retrieved documents

    # Check if indices are valid and retrieve documents
    retrieved_docs = []
    for i in I[0]:
        if i < len(documents):  # Check to avoid out-of-bounds access
            retrieved_docs.append(documents[i])
        else:
            logger.info(f"Warning: Retrieved index {i} is out of range for documents list.")
    logger.info("Method: retrieve_context execution completed.")
    return retrieved_docs

# Function to generate the answer using the DPRReader (Facebook's model for QA)
def generate_answer(query, retrieved_docs):
    """Generate an answer to the query using the DPRReader model."""
    logger.info("Method: generate_answer execution started.")
    # Combine retrieved documents into a context for the model
    context = " ".join(retrieved_docs)

    # Encode the query and context to generate the answer
    inputs = reader_tokenizer(query, context, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = reader_model(**inputs)

    # Get the best answer (usually in the 'start' and 'end' logits)
    start_idx = outputs.start_logits.argmax()
    end_idx = outputs.end_logits.argmax()

    # Decode the answer from the tokens
    answer_tokens = inputs.input_ids[0][start_idx:end_idx + 1]
    answer = reader_tokenizer.decode(answer_tokens, skip_special_tokens=True)
    logger.info("Method: generate_answer execution completed.")
    return answer

# Function to download models from GCP bucket
def download_from_gcp(bucket_name, source_folder, destination_folder):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=source_folder)  # List all files in the source folder
    for blob in blobs:
        # Skip "folder" blobs (those that end with "/")
        if blob.name.endswith("/"):
            continue

        # Create the full local path for each file
        local_path = os.path.join(destination_folder, os.path.relpath(blob.name, source_folder))

        # Create local directories if they don't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the blob to the local file path
        blob.download_to_filename(local_path)

        print(f"Downloaded {blob.name} to {local_path}")

if __name__ != "__main__":
    try:
        # Load models and tokenizers at startup
        logger.info("Preloading models and tokenizers...")
        bucket_name = "fin_rag_model_storage"
        source_folder = "facebook_dpr_model"
        destination_folder = "./downloaded_models"
        os.makedirs(destination_folder, exist_ok=True)
        download_from_gcp(bucket_name, source_folder, destination_folder)

        context_encoder = DPRContextEncoder.from_pretrained(
            os.path.join(destination_folder, "context_encoder")
        )
        question_encoder = DPRQuestionEncoder.from_pretrained(
            os.path.join(destination_folder, "question_encoder")
        )
        reader_model = DPRReader.from_pretrained(
            os.path.join(destination_folder, "reader_model")
        )
        reader_tokenizer = DPRReaderTokenizer.from_pretrained(
            os.path.join(destination_folder, "reader_tokenizer")
        )
        index = load_index_from_gcs()
        documents_data = load_documents_data_from_gcs()

        logger.info("Models and tokenizers preloaded successfully.")
    except Exception as e:
        logger.error(f"Error preloading models or tokenizers: {e}")
        raise e
# Route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the request JSON
        data = request.json

        # Extract query and documents
        query = data.get("query")

        # Get the retrieved context documents
        retrieved_docs = retrieve_context(query, index, documents_data)

        # Generate the answer using the DPRReader model
        answer = generate_answer(query, retrieved_docs)

        logger.info("Prediction successful.")
        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500


# Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8085)
