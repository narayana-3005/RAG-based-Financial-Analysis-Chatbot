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

# Create a storage client
storage_client = storage.Client()

# Config Bucket
config_bucket = storage_client.bucket('fin_rag_config')

# Load ticker_company_map
ticker_company_blob = config_bucket.blob('ticker_company_map.json')
json_string = ticker_company_blob.download_as_string()
ticker_company_map = json.loads(json_string)

# DPR Reader Setup
reader_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# FAISS Index Setup
def build_or_update_faiss_index(documents, index_name):
    """Build FAISS index for the context documents."""
    logger.info("Method: build_faiss_index execution started.")
    context_embeddings = []
    for doc in documents:
        # Tokenize and process each document
        inputs = reader_tokenizer(doc, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embeddings = context_encoder(**inputs).pooler_output
        context_embeddings.append(embeddings.cpu().numpy())

    # Convert the embeddings list to a numpy array
    context_embeddings = np.vstack(context_embeddings)

    # FAISS index for efficient retrieval
    if index_name == None:
        # Create Index
        index_name = 'index.faiss'
        index = faiss.IndexFlatL2(context_embeddings.shape[1])  # Flat L2 index
    else:
        # Load existing index
        index_blobs = storage_client.list_blobs('index_faiss')
        index_blob = next(index_blobs)  # Assuming only one index file in the bucket
        index_blob.download_to_filename('index.faiss')
        index = faiss.read_index('index.faiss')

    # Add the context embeddings to the index
    index.add(context_embeddings)

    # Save Index
    faiss.write_index(index, index_name )
    # Upload the updated index to cloud storage, overwriting the existing file
    bucket = storage_client.bucket('index_faiss')
    blob = bucket.blob(index_name)

    # Delete the existing blob if it exists and upload the new one
    if blob.exists():
        blob.delete()
    blob.upload_from_filename(index_name)

    # Remove the temporary file
    os.remove(index_name)
    logger.info("Method: build_faiss_index execution completed.")
    return index

def list_and_download_parquet_blobs(bucket_name):
    """Lists all the Parquet blobs in the specified bucket and returns their content as a list of raw text."""
    logger.info("Method: list_and_download_parquet_blobs execution started.")
    # List to hold raw content of the Parquet files
    parquet_data = []

    # Initialize the Cloud Storage client
    storage_client = storage.Client()
    try:
        # Retrieve all blobs (objects) in the specified bucket
        blobs = storage_client.list_blobs(bucket_name)

        # Iterate through each blob in the bucket
        for blob in blobs:
            # Check if the blob is a Parquet file
            if blob.name.endswith('.parquet'):
                try:
                    # Download the blob's content as a byte stream
                    parquet_bytes = blob.download_as_string()

                    # Read the Parquet data using pyarrow
                    buffer = io.BytesIO(parquet_bytes)
                    table = pq.read_table(buffer)

                    # Convert the Parquet table to a pandas DataFrame (optional step)
                    df = table.to_pandas()

                    # Append the DataFrame (or just the raw Parquet data) to the list
                    # parquet_data = parquet_data + (df.summary.to_list())
                    for index, row in df.iterrows():
                      # Format the date as "Month Year Day" (e.g., November 2024 02)
                      formatted_date = index.strftime('%B %Y %d')
                      text = f"On date {formatted_date} , for company name {ticker_company_map[row['ticker']]} and Ticker name {row['ticker']} news title is \"{row['title']}\" with summary : \"{row['summary']}\" and sentiment score : \"{row['sentiment']}\""
                      parquet_data.append(text)

                except Exception as e:
                    logger.error(f"Failed to process {blob.name}: {e}")

        # Save the list of raw Parquet data to a file
        with open("news_data.json", "w") as f:
            json.dump(parquet_data, f)

        # Upload the file to GCS
        storage_client = storage.Client()
        bucket_name = 'news_data_bucket'
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob('news_data.json')
        blob.upload_from_filename('news_data.json')

        # Remove the temporary file
        os.remove('news_data.json')
    except Exception as e:
        logger.error(f"Error accessing bucket '{bucket_name}': {e}")
    logger.info("Method: list_and_download_parquet_blobs execution completed.")
    # Return the list of DataFrames or raw Parquet data
    return parquet_data

# Load Market data
index_name = 'index.faiss'
documents_data = list_and_download_parquet_blobs('news_article-bucket_preprocessed')
index = build_or_update_faiss_index(documents_data, index_name)
