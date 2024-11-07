import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from stockprices_preprocessing import preprocess_stock_data  # Import your function

# Mock Google Cloud Storage Client to avoid errors during tests
@pytest.fixture(autouse=True)
def mock_gcs_client(monkeypatch):
    with patch('google.cloud.storage.Client'):
        yield

# Sample preprocessed data based on your provided example
@pytest.fixture
def sample_preprocessed_data():
    data = {
        'Adj Close': [175.75, 178.31, 180.89, 181.95, 181.48, 233.39, 230.10, 225.91, 222.91],
        'Close': [176.65, 179.23, 181.82, 182.89, 182.41, 233.39, 230.10, 225.91, 222.91],
        'High': [176.82, 179.42, 182.44, 183.44, 184.11, 234.72, 233.47, 229.83, 225.35],
        'Low': [173.35, 176.21, 178.97, 181.59, 181.81, 232.55, 229.55, 225.37, 220.27],
        'Open': [174.24, 176.38, 179.17, 182.35, 182.96, 233.32, 232.61, 229.34, 220.97],
        'Volume': [79763700, 63841300, 70530000, 49340300, 53763500, 36087100, 47070900, 64370100, 65242200],
        'Date': pd.to_datetime(['2023-11-03', '2023-11-06', '2023-11-07', '2023-11-08', 
                                '2023-11-09', '2024-10-28', '2024-10-30', '2024-10-31', '2024-11-01']),
        'Symbol': ['AAPL'] * 9,
        'Daily_Return': [np.nan, 0.0146, 0.0144, 0.0058, -0.0026, 0.0085, -0.0153, -0.0182, -0.0133],
        'SMA_20': [np.nan] * 9,
        'ATR_14': [np.nan] * 9,
        'Volume_Change': [np.nan, -0.1996, 0.1048, -0.3004, 0.0896, -0.07, 0.329, 0.367, 0.013]
    }
    return pd.DataFrame(data)

# 1. Test if the essential columns exist
def test_columns_exist(sample_preprocessed_data):
    required_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 
                        'Date', 'Symbol', 'Daily_Return', 'SMA_20', 'ATR_14', 'Volume_Change']
    assert all(col in sample_preprocessed_data.columns for col in required_columns)

# 2. Test for no missing values in essential columns
def test_no_missing_values_in_essential_columns(sample_preprocessed_data):
    essential_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Date', 'Symbol']
    for col in essential_columns:
        assert sample_preprocessed_data[col].notnull().all()

# 3. Test for correct data types
def test_column_data_types(sample_preprocessed_data):
    assert pd.api.types.is_float_dtype(sample_preprocessed_data['Adj Close'])
    assert pd.api.types.is_float_dtype(sample_preprocessed_data['Close'])
    assert pd.api.types.is_float_dtype(sample_preprocessed_data['High'])
    assert pd.api.types.is_float_dtype(sample_preprocessed_data['Low'])
    assert pd.api.types.is_float_dtype(sample_preprocessed_data['Open'])
    assert sample_preprocessed_data['Volume'].dtype == np.int64
    assert pd.api.types.is_datetime64_any_dtype(sample_preprocessed_data['Date'])

# 4. Test if `Daily_Return` values are within a reasonable range
def test_daily_return_values(sample_preprocessed_data):
    # Ignore NaN values for the check
    daily_return = sample_preprocessed_data['Daily_Return'].dropna()
    assert daily_return.between(-1, 1).all()

# 5. Test for rolling mean (SMA_20) and check that it has NaN values for the first 19 entries
def test_sma_20_calculation(sample_preprocessed_data):
    sma_20 = sample_preprocessed_data['SMA_20']
    assert sma_20[:19].isnull().all()
    if len(sma_20) > 20:
        assert sma_20[19:].notnull().any()

# 6. Test for ATR_14 and check that it has NaN values for the first 13 entries
def test_atr_14_calculation(sample_preprocessed_data):
    atr_14 = sample_preprocessed_data['ATR_14']
    assert atr_14[:13].isnull().all()
    if len(atr_14) > 14:
        assert atr_14[14:].notnull().any()

# 7. Test `Volume_Change` values for sanity
def test_volume_change_values(sample_preprocessed_data):
    # Ignore NaN values for the check
    volume_change = sample_preprocessed_data['Volume_Change'].dropna()
    assert volume_change.between(-1, 1).all()

# 8. Test for the presence of extreme values
def test_extreme_values_in_data(sample_preprocessed_data):
    assert (sample_preprocessed_data['High'] >= sample_preprocessed_data['Low']).all(), "High should be greater than or equal to Low"
    assert (sample_preprocessed_data['Volume'] > 0).all(), "Volume should be positive"
