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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIRECTORY = os.path.join(SCRIPT_DIR, "market_data_cache")
DATA_PERIOD = "10y" # Download 10 years of historical data
DATA_INTERVAL = "1d" # Use daily data

def create_market_data_cache():
    """
    Downloads historical daily price data for a list of symbols and saves
    each to a CSV file in a cache directory. This only needs to be run once,
    or whenever you want to refresh the historical data.
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
            
            output_path = os.path.join(CACHE_DIRECTORY, f"{ticker}.csv")
            price_data.to_csv(output_path)
            print(f"Successfully saved data for {ticker} to {output_path}")

        except Exception as e:
            print(f"An error occurred while downloading data for {ticker}: {e}")

    print("\n--- Market Data Cache Creation Complete! ---")

if __name__ == "__main__":
    create_market_data_cache()
