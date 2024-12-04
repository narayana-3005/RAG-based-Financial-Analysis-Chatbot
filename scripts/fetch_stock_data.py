import os
import yfinance as yf
from google.cloud import storage
import json
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# Configure Google Cloud Storage
STOCK_BUCKET_NAME = 'stock_prices-bucket'
SERVICE_ACCOUNT_JSON = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')  # Environment variable for JSON key path
storage_client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_JSON)
bucket = storage_client.bucket(STOCK_BUCKET_NAME)



# List of tickers to fetch (you can adjust this list as needed)
sp500_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "BRK.B", "V", "JNJ",
    "WMT", "JPM", "PG", "MA", "UNH"]

def upload_to_gcs(data, folder_name, filename):
    """Uploads data to GCS in the specified folder."""
    blob_path = f"{folder_name}/{filename}"
    blob = bucket.blob(blob_path)
    blob.upload_from_string(
        data=json.dumps(data),
        content_type='application/json'
    )
    logger.info(f"Uploaded {filename} to {STOCK_BUCKET_NAME}/{folder_name}")

def fetch_historical_data(ticker):
    """Fetches historical daily data for the ticker for the last year with a 1-day interval."""
    end_date = datetime.now()
    # start_date = end_date - timedelta(days=365)  # One year ago from today
    start_date = end_date - timedelta(days=5)
    data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), interval="1d")

    if not data.empty:
        for date, row in data.iterrows():
            # Date fields from historical data
            year = date.strftime("%Y")
            month = date.strftime("%m")
            day = date.strftime("%d")

            # Current time fields for hour, minute, second
            now = datetime.now()
            hour = now.strftime("%H")
            minute = now.strftime("%M")
            second = now.strftime("%S")

            # Convert the row data to dictionary and add the date
            day_data = row.to_dict()
            day_data['Date'] = date.strftime("%Y-%m-%d")  # Add the date as a string

            # Ensure all keys and values in the dictionary are serializable
            day_data = {str(k): (v if isinstance(v, (str, int, float, bool, type(None))) else str(v)) for k, v in day_data.items()}

            # Convert dictionary to JSON
            try:
                day_json = json.dumps(day_data)
            except TypeError as e:
                logger.error(f"Error serializing data for {ticker} on {date.strftime('%Y-%m-%d')}: {e}")
                continue  # Skip this entry if it cannot be serialized

            # Define folder structure and filename
            folder_name = f"historical/{ticker}/{year}/Month={month}/Day={day}/Hour={hour}/Minute={minute}"
            filename = f"{ticker}_{second}.json"
            upload_to_gcs(day_json, folder_name, filename)
    else:
        print(f"No historical data for {ticker}")

def store_data_to_gcs():
    """Fetches and stores historical data for each ticker."""
    for ticker in sp500_tickers:
        # Fetch and upload historical data
        fetch_historical_data(ticker)

        # Respect rate limits
        time.sleep(2)  # Adjust sleep time as necessary

# Run the function to store data
if __name__ == "__main__":
    store_data_to_gcs()
