import pandas as pd
import json
from google.cloud import storage
from datetime import datetime
import os
import numpy as np

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
    # Data validation and anomaly detection before preprocessing
    validate_data_quality(df)
    detect_anomalies(df)

    # Feature Engineering
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

def validate_data_quality(df):
    """Validate data quality by checking for missing values and data types."""
    print("\n--- Data Quality Validation ---")

    # Check for missing values
    missing_values = df.isnull().sum()
    print("Missing Values Per Column:")
    print(missing_values[missing_values > 0])

    # Check for invalid data types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if df[col].dtype not in [np.float64, np.int64]:
            raise ValueError(f"Invalid data type in column {col}. Expected numeric type.")

    # Check if dates are in chronological order
    if not df['Date'].is_monotonic_increasing:
        print("Warning: Dates are not in chronological order.")

def detect_anomalies(df):
    """Detect anomalies such as outliers, missing data, and unusual patterns."""
    print("\n--- Anomaly Detection ---")

    # Check for negative prices
    for col in ['Open', 'High', 'Low', 'Close']:
        if (df[col] < 0).any():
            print(f"Anomaly Detected: Negative values in column {col}.")

    # Check for unusual volume spikes
    volume_mean = df['Volume'].mean()
    volume_std = df['Volume'].std()
    high_volume_anomalies = df[df['Volume'] > (volume_mean + 3 * volume_std)]
    if not high_volume_anomalies.empty:
        print("Anomaly Detected: High volume spikes found.")
        print(high_volume_anomalies[['Date', 'Volume']])

    # Check for extreme daily return values
    df['Daily Return'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    return_mean = df['Daily Return'].mean()
    return_std = df['Daily Return'].std()
    extreme_returns = df[(df['Daily Return'] > return_mean + 3 * return_std) |
                         (df['Daily Return'] < return_mean - 3 * return_std)]
    if not extreme_returns.empty:
        print("Anomaly Detected: Extreme daily returns found.")
        print(extreme_returns[['Date', 'Daily Return']])

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

def preprocess_all_data(tickers):
    """Load, validate, preprocess, and save data for all specified tickers."""
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

# List of tickers to process
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "V", "JNJ", "WMT", "JPM", "PG", "MA", "UNH"]

if __name__ == "__main__":
    preprocess_all_data(tickers)

def inspect_data_loading(df, expected_columns, ticker):
    """Inspect if data was loaded correctly for the given ticker."""
    print(f"\n--- Data Inspection for {ticker} ---")

    # Check if DataFrame is empty
    if df.empty:
        print("Warning: The DataFrame is empty.")
        return False

    # Check for correct columns
    missing_columns = set(expected_columns) - set(df.columns)
    extra_columns = set(df.columns) - set(expected_columns)

    if missing_columns:
        print(f"Missing columns in {ticker} data: {missing_columns}")
    if extra_columns:
        print(f"Unexpected columns in {ticker} data: {extra_columns}")

    # Check data types
    print("\nColumn Data Types:")
    print(df.dtypes)

    # Check for missing values again
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("\nMissing Values Per Column:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found in the data.")

    # Display basic statistics
    print("\nBasic Statistics:")
    print(df.describe(include='all'))

    # Display the first few rows to inspect the data format
    print("\nSample Data:")
    print(df.head())

    # Final check if all expected columns are present and data is loaded
    if not missing_columns and not df.empty:
        print("Data has been correctly loaded and inspected for all expected columns.")
        return True
    else:
        print("Data loading issues detected.")
        return False

# Define expected columns for stock data
expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Perform inspection for each ticker
for ticker in tickers:
    df = load_raw_data_from_gcs(ticker)
    inspect_data_loading(df, expected_columns, ticker)

"""Step 1: Schema Validation

"""

def validate_schema(df, expected_schema, ticker):
    """Validate DataFrame schema against the expected schema."""
    print(f"\n--- Schema Validation for {ticker} ---")
    issues_found = False  # Track if any issues are found

    for column, dtype in expected_schema.items():
        if column not in df.columns:
            print(f"Missing column: {column}")
            issues_found = True
        elif not pd.api.types.is_dtype_equal(df[column].dtype, dtype):
            print(f"Type mismatch in column {column}: Expected {dtype}, got {df[column].dtype}")
            issues_found = True

    if not issues_found:
        print(f"Schema validation passed for {ticker}.")

# Run validation for each ticker
for ticker in tickers:
    df = load_raw_data_from_gcs(ticker)
    if not df.empty:
        validate_schema(df, expected_schema, ticker)

"""Step 2: Enhanced Anomaly Detection with Alerts

"""

def validate_schema(df, expected_schema, ticker):
    """Validate DataFrame schema against the expected schema and display schema information."""
    print(f"\n--- Schema Validation for {ticker} ---")

    # Display the expected schema
    print("\nExpected Schema:")
    for column, dtype in expected_schema.items():
        print(f"  {column}: {dtype}")

    # Display the actual schema in the DataFrame
    print("\nActual Schema:")
    print(df.dtypes)

    issues_found = False  # Track if any issues are found

    # Validate each column's existence and data type
    for column, dtype in expected_schema.items():
        if column not in df.columns:
            print(f"Missing column: {column}")
            issues_found = True
        elif not pd.api.types.is_dtype_equal(df[column].dtype, dtype):
            print(f"Type mismatch in column {column}: Expected {dtype}, got {df[column].dtype}")
            issues_found = True

    # Print success message if no issues are found
    if not issues_found:
        print(f"\nSchema validation passed for {ticker}.")
    else:
        print(f"\nSchema validation found issues for {ticker}.")

# Run validation for each ticker
for ticker in tickers:
    df = load_raw_data_from_gcs(ticker)
    if not df.empty:
        validate_schema(df, expected_schema, ticker)

def enhanced_anomaly_detection(df, ticker):
    """Enhanced anomaly detection with alerts for high volume and extreme returns."""
    print(f"\n--- Enhanced Anomaly Detection for {ticker} ---")

    # Volume spike alert threshold
    volume_threshold = df['Volume'].mean() + 3 * df['Volume'].std()
    high_volume_anomalies = df[df['Volume'] > volume_threshold]
    if not high_volume_anomalies.empty:
        print(f"Volume spike alert for {ticker}: {len(high_volume_anomalies)} instances found")
        print(high_volume_anomalies[['Date', 'Volume']])

    # Extreme daily returns alert threshold
    df['Daily Return'] = ((df['Close'] - df['Open']) / df['Open']) * 100
    return_threshold = 3 * df['Daily Return'].std()
    extreme_returns = df[(df['Daily Return'] > return_threshold) | (df['Daily Return'] < -return_threshold)]
    if not extreme_returns.empty:
        print(f"Extreme return alert for {ticker}: {len(extreme_returns)} instances found")
        print(extreme_returns[['Date', 'Daily Return']])

# Run enhanced anomaly detection with alerts for each ticker
for ticker in tickers:
    df = load_raw_data_from_gcs(ticker)
    if not df.empty:
        enhanced_anomaly_detection(df, ticker)

"""Step 3: Bias Detection and Mitigation

"""

def detect_bias(df, ticker):
    """Detect and report potential bias in data for a given ticker."""
    print(f"\n--- Bias Detection for {ticker} ---")

    # Bias in Volume: Compare average volume across the entire dataset
    overall_mean_volume = df['Volume'].mean()
    stock_mean_volume = df['Volume'].mean()
    if stock_mean_volume > overall_mean_volume * 1.5:
        print(f"Bias detected in volume for {ticker}: {stock_mean_volume} vs overall {overall_mean_volume}")

    # Bias in Daily Returns: Consistent higher or lower returns for this stock
    overall_return_mean = df['Daily Return'].mean()
    stock_return_mean = df['Daily Return'].mean()
    if abs(stock_return_mean - overall_return_mean) > df['Daily Return'].std():
        print(f"Bias in returns for {ticker}: {stock_return_mean} vs overall {overall_return_mean}")

# Detect bias for each ticker
for ticker in tickers:
    df = load_raw_data_from_gcs(ticker)
    if not df.empty:
        df['Daily Return'] = ((df['Close'] - df['Open']) / df['Open']) * 100
        detect_bias(df, ticker)

def detect_bias(df, ticker, overall_volume_mean, overall_return_mean):
    """Detect potential bias in data for a given ticker compared to overall metrics."""
    print(f"\n--- Bias Detection for {ticker} ---")
    stock_mean_volume = df['Volume'].mean()
    stock_return_mean = df['Daily Return'].mean()

    # Detect volume bias
    if stock_mean_volume > overall_volume_mean * 1.5:
        print(f"Volume bias detected for {ticker}: Mean Volume is {stock_mean_volume} vs. overall mean of {overall_volume_mean}")
    elif stock_mean_volume < overall_volume_mean * 0.5:
        print(f"Volume bias detected for {ticker}: Mean Volume is {stock_mean_volume} vs. overall mean of {overall_volume_mean}")

    # Detect return bias
    if abs(stock_return_mean - overall_return_mean) > df['Daily Return'].std():
        print(f"Return bias detected for {ticker}: Mean Return is {stock_return_mean} vs. overall mean of {overall_return_mean}")

# Calculate overall metrics for bias detection
all_data = pd.concat([load_raw_data_from_gcs(ticker) for ticker in tickers])
all_data['Daily Return'] = ((all_data['Close'] - all_data['Open']) / all_data['Open']) * 100
overall_volume_mean = all_data['Volume'].mean()
overall_return_mean = all_data['Daily Return'].mean()

# Detect bias for each ticker
for ticker in tickers:
    df = load_raw_data_from_gcs(ticker)
    if not df.empty:
        df['Daily Return'] = ((df['Close'] - df['Open']) / df['Open']) * 100
        detect_bias(df, ticker, overall_volume_mean, overall_return_mean)
