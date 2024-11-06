import pandas as pd
import json
from google.cloud import storage
from datetime import datetime
import os
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Configure GCS
RAW_BUCKET_NAME = 'stock_prices-bucket'
PREPROCESSED_BUCKET_NAME = 'stock_prices-bucket_preprocessed'
SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Initialize Google Cloud Storage client
storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_JSON)
raw_bucket = storage_client.bucket(RAW_BUCKET_NAME)
preprocessed_bucket = storage_client.bucket(PREPROCESSED_BUCKET_NAME)

def preprocess_stock_data(df):
    """Clean, transform, and add features to stock data DataFrame."""
    df['Daily Return'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    df['5-Day MA'] = df['Close'].rolling(window=5).mean()
    df['20-Day MA'] = df['Close'].rolling(window=20).mean()
    df['5-Day Volatility'] = df['Close'].rolling(window=5).std()
    return df

def load_raw_data_from_gcs(ticker):
    """Load raw data for the given ticker from GCS and convert it to a DataFrame."""
    blobs = raw_bucket.list_blobs(prefix=f"historical/{ticker}/")
    data = []

    for blob in blobs:
        if blob.name.endswith('.json'):
            content = blob.download_as_text()
            try:
                outer_record = json.loads(content)
                record = json.loads(outer_record)
                clean_record = {key.split(",")[0].strip("(' "): value for key, value in record.items()}
                data.append(clean_record)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for blob {blob.name}")
                continue

    if not data:
        print(f"No data found for {ticker}.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df['Adj Close'] = df['Adj Close'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Open'] = df['Open'].astype(float)
    df['Volume'] = df['Volume'].astype(int)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def save_preprocessed_data_to_gcs(df, ticker):
    """Save preprocessed data to GCS in Parquet format with the specified folder structure."""
    if df.empty:
        year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
        folder_name = f"{ticker}/year={year}/month={month:02d}/day={day:02d}"
        filename = f"preprocessed_{ticker}_{year}{month:02d}{day:02d}_empty.parquet"
        pd.DataFrame().to_parquet('/tmp/empty.parquet', index=False)
        blob = preprocessed_bucket.blob(f"{folder_name}/{filename}")
        blob.upload_from_filename('/tmp/empty.parquet')
        os.remove('/tmp/empty.parquet')
        return

    for _, row in df.iterrows():
        date = row['Date']
        year, month, day = date.year, date.month, date.day
        folder_name = f"{ticker}/year={year}/month={month:02d}/day={day:02d}"
        filename = f"preprocessed_{ticker}_{year}{month:02d}{day:02d}.parquet"
        row_df = pd.DataFrame([row])
        row_df.to_parquet('/tmp/temp.parquet', index=False)
        blob = preprocessed_bucket.blob(f"{folder_name}/{filename}")
        blob.upload_from_filename('/tmp/temp.parquet')
        os.remove('/tmp/temp.parquet')

def preprocess_all_data():
    """Load, preprocess, and save data for all specified tickers."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "V", "JNJ", "WMT", "JPM", "PG", "MA", "UNH"]
    for ticker in tickers:
        print(f"Processing data for {ticker}...")
        try:
            df = load_raw_data_from_gcs(ticker)
            if not df.empty:
                df = preprocess_stock_data(df)
            save_preprocessed_data_to_gcs(df, ticker)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

if __name__ == "__main__":
    preprocess_all_data()