import pandas as pd
import os
import json
from google.cloud import storage
from datetime import datetime

# Initialize GCS client
storage_client = storage.Client()

# Bucket configurations
PREPROCESSED_BUCKET_NAME = 'stock_prices-bucket_preprocessed'
TRANSFORMED_BUCKET_NAME = 'stock_prices_transformed'
preprocessed_bucket = storage_client.bucket(PREPROCESSED_BUCKET_NAME)
transformed_bucket = storage_client.bucket(TRANSFORMED_BUCKET_NAME)

def list_parquet_files_for_month(ticker, year, month):
    """List all .parquet files in GCS for a specific ticker, year, and month."""
    prefix = f"{ticker}/year={year}/month={month:02d}/"
    blobs = list(preprocessed_bucket.list_blobs(prefix=prefix))
    return [blob.name for blob in blobs if blob.name.endswith('.parquet')]

def load_parquet_from_gcs(parquet_files, ticker):
    """Download and load parquet files from GCS into a single DataFrame, adding the ticker column."""
    df_list = []

    for file_path in parquet_files:
        blob = preprocessed_bucket.blob(file_path)
        data = blob.download_as_bytes()
        df = pd.read_parquet(pd.io.common.BytesIO(data))
        df['Ticker'] = ticker
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)

def generate_text_from_dataframe(df, company_name):
    """Generate a text summary for each day in the DataFrame."""
    text_data = []

    for _, row in df.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        high = row['High']
        low = row['Low']
        close = row['Close']
        open_ = row['Open']
        volume = row['Volume']
        daily_return = row['Daily Return']

        text_data.append(
            f"On {date_str}, {company_name} company's stock made a high and low of "
            f"{high}$ and {low}$, its closing and opening market prices were "
            f"{close}$ and {open_}$ with an overall volume of {volume} shares traded "
            f"and a return of {daily_return:.6f}% on daily basis."
        )

    return "\n".join(text_data)

def save_text_to_gcs(text, ticker, year, month):
    """Save the generated text to GCS as a .txt file."""
    folder_name = f"{ticker}/year={year}/month={month:02d}"
    filename = f"{ticker}_{year}{month:02d}.txt"

    tmp_file = f"/tmp/{filename}"
    with open(tmp_file, 'w') as f:
        f.write(text)

    blob = transformed_bucket.blob(f"{folder_name}/{filename}")
    blob.upload_from_filename(tmp_file)

    os.remove(tmp_file)
    print(f"Uploaded transformed data for {ticker} to {folder_name}/{filename}")

def process_request(request):
    """Cloud Function entry point for processing stock data."""
    try:
        request_json = request.get_json()

        # Extract tickers, year, and months from request
        tickers = request_json.get('tickers', [])  # List of tickers
        year = int(request_json.get('year'))      # Year
        months = request_json.get('months', [])   # List of months

        if not tickers or not year or not months:
            return ("Missing required parameters: 'tickers', 'year', 'months'", 400)

        if not isinstance(months, list):
            return ("'months' parameter must be a list", 400)

        for ticker in tickers:
            print(f"Processing data for ticker: {ticker}")
            for month in months:
                print(f"Processing data for year {year}, month {month:02d}")

                parquet_files = list_parquet_files_for_month(ticker, year, month)
                if not parquet_files:
                    print(f"No data found for {ticker} in {year}-{month:02d}")
                    continue

                df = load_parquet_from_gcs(parquet_files, ticker)
                company_name = ticker  # Replace with a mapping if company names differ from tickers
                text_summary = generate_text_from_dataframe(df, company_name)

                save_text_to_gcs(text_summary, ticker, year, month)

        return "Processing complete. Check the transformed bucket for results.", 200

    except Exception as e:
        print(f"Error processing request: {e}")
        return f"Error processing request: {e}", 500


if __name__ == "__main__":
    # Example for local testing
    from flask import Request

    class MockRequest(Request):
        def __init__(self, json_data):
            self._json_data = json_data

        def get_json(self):
            return self._json_data

    mock_request = MockRequest({
        "tickers": ["AAPL"],
        "year": 2024,
        "month": 10
    })

    print(process_request(mock_request))
