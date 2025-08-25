import yfinance as yf
import pandas as pd
import os

# --- Configuration ---
# A diverse list of symbols representing the top 500 South American stocks
SYMBOLS_TO_CACHE = [
    '^NSEI',
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
    This version uses a robust method to prevent metadata corruption in the CSV.
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

            # --- ROBUST DATA CLEANING AND RECONSTRUCTION ---

            # 1. Isolate the 'Close' column into a pandas Series.
            clean_series = data['Close'].dropna()

            # 2. Filter out any non-positive prices to prevent numerical errors.
            clean_series = clean_series[clean_series > 0]

            # If after cleaning, the data is empty, skip.
            if clean_series.empty:
                print(f"Warning: No valid data remains for ticker {ticker} after cleaning. Skipping.")
                continue

            # 3. Create a BRAND NEW, clean DataFrame from the isolated Series.
            #    This is the key step to strip any unwanted metadata or complex headers.
            output_df = pd.DataFrame(clean_series)

            # 4. Explicitly name the index column 'Date'. This will become the first column in the CSV.
            output_df.index.name = 'Date'

            # 5. Save the newly created clean DataFrame.
            #    Replace characters in ticker that are invalid for filenames, like '^'.
            safe_ticker_name = ticker.replace('^', '')
            output_path = os.path.join(CACHE_DIRECTORY, f"{safe_ticker_name}.csv")
            output_df.to_csv(output_path) # Now the simplest to_csv call will work perfectly.

            print(f"Successfully saved {len(output_df)} rows of data for {ticker} to {output_path}")

        except Exception as e:
            print(f"An error occurred while downloading data for {ticker}: {e}")

    print("\n--- Market Data Cache Creation Complete! ---")

if __name__ == "__main__":
    create_market_data_cache()
