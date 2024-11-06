import pandas as pd
import json
from google.cloud import storage
from datetime import datetime
import os
from textblob import TextBlob

# Configure GCS
RAW_BUCKET_NAME = 'news_articles-bucket'
PREPROCESSED_BUCKET_NAME = 'news_articles-bucket_preprocessed'
SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
 
# Initialize Google Cloud Storage client
storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_JSON)
raw_bucket = storage_client.bucket(RAW_BUCKET_NAME)
preprocessed_bucket = storage_client.bucket(PREPROCESSED_BUCKET_NAME)

def preprocess_news_data(df):
    """Clean, transform, and add features to news articles DataFrame."""
    # Example: Apply basic text preprocessing
    df['summary'] = df['summary'].fillna('').str.strip()
    df['content'] = df['content'].fillna('').str.strip()

    def calculate_sentiment(text):
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        return sentiment
    
    # Calculate sentiment if not already calculated (Assuming sentiment is a float between -1 and 1)
    if 'sentiment' not in df.columns:
        df['sentiment'] = df['content'].apply(lambda x: calculate_sentiment(x))
    
    # Convert dates if necessary
    if 'published_utc' in df.columns:
        df['published_utc'] = pd.to_datetime(df['published_utc'])
    
    return df

def load_raw_news_data_from_gcs(ticker):
    """Load raw news data for the given ticker from GCS and convert it to a DataFrame."""
    blobs = raw_bucket.list_blobs(prefix=f"news/{ticker}/")
    data = []

    for blob in blobs:
        if blob.name.endswith('.json'):
            content = blob.download_as_text()
            try:
                record = json.loads(content)
                data.append(record)
            except json.JSONDecodeError:
                print(f"Error decoding JSON for blob {blob.name}")
                continue

    if not data:
        print(f"No data found for {ticker}.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    return df

def save_preprocessed_news_data_to_gcs(df, ticker):
    """Save preprocessed news data to GCS in Parquet format with the specified folder structure."""
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
        date = row['published_utc']
        year, month, day = date.year, date.month, date.day
        folder_name = f"{ticker}/year={year}/month={month:02d}/day={day:02d}"
        filename = f"preprocessed_{ticker}_{year}{month:02d}{day:02d}.parquet"
        row_df = pd.DataFrame([row])
        row_df.to_parquet('/tmp/temp.parquet', index=False)
        blob = preprocessed_bucket.blob(f"{folder_name}/{filename}")
        blob.upload_from_filename('/tmp/temp.parquet')
        os.remove('/tmp/temp.parquet')

def preprocess_all_news_data():
    """Load, preprocess, and save news data for all specified tickers."""
    # List of tickers to process
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "V", "JNJ", "WMT", "JPM", "PG", "MA", "UNH"]
    for ticker in tickers:
        print(f"Processing news data for {ticker}...")
        try:
            df = load_raw_news_data_from_gcs(ticker)
            if not df.empty:
                df = preprocess_news_data(df)
            save_preprocessed_news_data_to_gcs(df, ticker)
        except Exception as e:
            print(f"Error processing news data for {ticker}: {e}")
            continue


if __name__ == "__main__":
    preprocess_all_news_data()
