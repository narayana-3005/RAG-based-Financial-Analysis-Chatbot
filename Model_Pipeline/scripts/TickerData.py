!pip install yfinance

#Imports
import yfinance as yf
import json

def get_ticker_company_mapping(tickers_list):
    """
    Gets a list of tickers and their corresponding company names using yfinance.

    Args:
        tickers_list: A list of ticker symbols.

    Returns:
        dict: A dictionary where keys are tickers and values are company names.
    """
    try:
        tickers = yf.Tickers(' '.join(tickers_list))
        ticker_company_map = {}
        for ticker in tickers.tickers:
            ticker_company_map[ticker] = tickers.tickers[ticker].info.get('longName', 'N/A')
        return ticker_company_map
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

tickers_list = ['AAPL', 'AMZN', 'BRK.B', 'GOOGL', 'JNJ', 'JPM', 'MA', 'META', 'MSFT', 'NVDA', 'PG', 'TSLA', 'UNH', 'V', 'WMT']
ticker_company_map = get_ticker_company_mapping(tickers_list)

if ticker_company_map:
    for ticker, company_name in ticker_company_map.items():
        print(f"{ticker}: {company_name}")
    # Convert the dictionary to a JSON string
    json_data = json.dumps(ticker_company_map, indent=4)

    # Specify the Google Cloud Storage bucket and file name
    file_name = "/content/ticker_company_map.json"
    with open(file_name, "w") as f:
        f.write(json_data)
    print(f"Ticker-company mapping saved to {file_name}")
