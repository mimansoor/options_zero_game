import yfinance as yf
import pandas as pd
import os

# --- Configuration ---
# A diverse list of symbols representing different market types and regimes
SYMBOLS_TO_CACHE = [
    # US Indices
    'SPY',  # S&P 500 ETF
    '^RUT', # Russell 2000 Index
    '^IXIC',# NASDAQ Composite
    '^DJI', # Dow Jones Industrial Average
    # Volatility & Bonds
    '^VIX', # CBOE Volatility Index
    'TLT',  # 20+ Year Treasury Bond ETF
    'BND',  # Total Bond Market ETF
    # International
    '^NSEI',# NIFTY 50 (India)
    # Commodities
    'GC=F', # Gold Futures
    # Tech Stocks
    'TSLA',
    'INTC',
    # Crypto
    'BTC-USD',
    'ETH-USD',
    'SOL-USD',
]

# Use a path relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIRECTORY = os.path.join(SCRIPT_DIR, "market_data_cache")
DATA_PERIOD = "10y"
DATA_INTERVAL = "1d"

def create_market_data_cache():
    """
    Downloads historical daily price data, cleans it by removing non-positive prices,
    and saves it to a CSV file in a cache directory.
    """
    print(f"--- Creating Market Data Cache in '{CACHE_DIRECTORY}' ---")
    os.makedirs(CACHE_DIRECTORY, exist_ok=True)

    for ticker in SYMBOLS_TO_CACHE:
        print(f"Downloading {DATA_PERIOD} of {DATA_INTERVAL} data for {ticker}...")
        try:
            data = yf.download(ticker, period=DATA_PERIOD, interval=DATA_INTERVAL, progress=False, auto_adjust=True)
            
            if data.empty:
                print(f"Warning: No data found for ticker {ticker}. Skipping.")
                continue

            # We only need the closing price for our simulation
            price_data = data[['Close']].dropna()

            # <<< --- MODIFICATION START --- >>>
            # Filter out any non-positive prices to prevent numerical errors in the environment.
            original_rows = len(price_data)
            price_data = price_data[price_data['Close'] > 0]
            cleaned_rows = len(price_data)

            if original_rows > cleaned_rows:
                print(f"-> Filtered out {original_rows - cleaned_rows} rows with non-positive prices for {ticker}.")
            # <<< --- MODIFICATION END --- >>>
            
            # If after cleaning, the data is empty, skip.
            if price_data.empty:
                print(f"Warning: No valid data remains for ticker {ticker} after cleaning. Skipping.")
                continue

            output_path = os.path.join(CACHE_DIRECTORY, f"{ticker}.csv")
            price_data.to_csv(output_path)
            print(f"Successfully saved {cleaned_rows} rows of data for {ticker} to {output_path}")

        except Exception as e:
            print(f"An error occurred while downloading data for {ticker}: {e}")

    print("\n--- Market Data Cache Creation Complete! ---")

if __name__ == "__main__":
    create_market_data_cache()
