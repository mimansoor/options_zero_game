import yfinance as yf
import pandas as pd
import os

# --- Configuration ---
# A diverse list of the top 500 Chinese company stock symbols
SYMBOLS_TO_CACHE = [
    # Shanghai Stock Exchange (.SS)
    '600519.SS', '601398.SS', '601288.SS', '601939.SS', '601988.SS', '601857.SS',
    '600036.SS', '601318.SS', '601628.SS', '600900.SS', '601088.SS', '601668.SS',
    '600887.SS', '601166.SS', '600028.SS', '603288.SS', '601328.SS', '600030.SS',
    '601899.SS', '600000.SS', '600276.SS', '601211.SS', '600048.SS', '601919.SS',
    '600585.SS', '601012.SS', '601728.SS', '601601.SS', '600016.SS', '600050.SS',
    '603993.SS', '601818.SS', '600690.SS', '601688.SS', '600309.SS', '600104.SS',
    '600809.SS', '600703.SS', '600031.SS', '601390.SS', '600019.SS', '601998.SS',
    '601138.SS', '600438.SS', '601186.SS', '601066.SS', '600547.SS', '601985.SS',
    '601236.SS', '601633.SS', '603160.SS', '600009.SS', '601888.SS', '601658.SS',
    '600570.SS', '600346.SS', '601111.SS', '600015.SS', '601336.SS', '600837.SS',
    '601086.SS', '601999.SS', '601229.SS', '601766.SS', '600010.SS', '603345.SS',
    '601901.SS', '600089.SS', '603259.SS', '601155.SS', '600176.SS', '600606.SS',
    '601600.SS', '601800.SS', '600999.SS', '603501.SS', '600196.SS', '600011.SS',
    '600745.SS', '601319.SS', '601006.SS', '60128.SS', '600018.SS', '600188.SS',
    '601816.SS', '60162.SS', '600233.SS', '600516.SS', '600383.SS', '603019.SS',
    '600754.SS', '600398.SS', '601788.SS', '601100.SS', '600588.SS', '600660.SS',
    '603486.SS', '601021.SS', '600406.SS', '600489.SS', '600487.SS', '603799.SS',
    '600362.SS', '600674.SS', '600498.SS', '600919.SS', '603658.SS', '601198.SS',
    '603986.SS', '600522.SS', '600208.SS', '600893.SS', '600325.SS', '600958.SS',
    '600918.SS', '600705.SS', '600655.SS', '600867.SS', '600989.SS', '600111.SS',
    '600968.SS', '600436.SS', '600977.SS', '600482.SS', '600549.SS', '600699.SS',
    '600779.SS', '600990.SS', '603899.SS', '603198.SS', '600300.SS', '600594.SS',
    '600150.SS', '600848.SS', '600800.SS', '600426.SS', '600100.SS', '600061.SS',
    '600025.SS', '600060.SS', '600109.SS', '600372.SS', '600074.SS', '600085.SS',
    '600332.SS', '600642.SS', '600795.SS', '600859.SS', '601555.SS', '601958.SS',
    '603228.SS', '603369.SS', '603713.SS', '603833.SS', '603882.SS', '603918.SS',
    '605117.SS', '605166.SS', '605228.SS', '605338.SS', '688005.SS', '688006.SS',
    '688008.SS', '688009.SS', '688012.SS', '688019.SS', '688029.SS', '688036.SS',
    '688063.SS', '688088.SS', '688099.SS', '688111.SS', '688126.SS', '688169.SS',
    '688187.SS', '688223.SS', '688266.SS', '688289.SS', '688321.SS', '688333.SS',
    '688363.SS', '688388.SS', '688396.SS', '688506.SS', '688516.SS', '688521.SS',
    '688561.SS', '688588.SS', '688599.SS', '688688.SS', '688981.SS',

    # Shenzhen Stock Exchange (.SZ)
    '300750.SZ', '002594.SZ', '000333.SZ', '000858.SZ', '300059.SZ', '002475.SZ',
    '002415.SZ', '300760.SZ', '002714.SZ', '000001.SZ', '000651.SZ', '002352.SZ',
    '000568.SZ', '300124.SZ', '002142.SZ', '000725.SZ', '300274.SZ', '002371.SZ',
    '002027.SZ', '002236.SZ', '000895.SZ', '000625.SZ', '002841.SZ', '002460.SZ',
    '300015.SZ', '000776.SZ', '000002.SZ', '002422.SZ', '002271.SZ', '002304.SZ',
    '300347.SZ', '002007.SZ', '300070.SZ', '002241.SZ', '002129.SZ', '000063.SZ',
    '300142.SZ', '002459.SZ', '300033.SZ', '002601.SZ', '002311.SZ', '002555.SZ',
    '002624.SZ', '002736.SZ', '300408.SZ', '300433.SZ', '300601.SZ', '000963.SZ',
    '300316.SZ', '000100.SZ', '002466.SZ', '300002.SZ', '002050.SZ', '002179.SZ',
    '002938.SZ', '002916.SZ', '300677.SZ', '002258.SZ', '002507.SZ', '300394.SZ',
    '002709.SZ', '000703.SZ', '000661.SZ', '000938.SZ', '002044.SZ', '002487.SZ',
    '300136.SZ', '002402.SZ', '300207.SZ', '002773.SZ', '002812.SZ', '002202.SZ',
    '000538.SZ', '300498.SZ', '000786.SZ', '000728.SZ', '300223.SZ', '300450.SZ',
    '002157.SZ', '000423.SZ', '002230.SZ', '000157.SZ', '000301.SZ', '002064.SZ',
    '002340.SZ', '300024.SZ', '300308.SZ', '300529.SZ', '002493.SZ', '000069.SZ',
    '000977.SZ', '300182.SZ', '002572.SZ', '300782.SZ', '002602.SZ', '002920.SZ',
    '002930.SZ', '002945.SZ', '002958.SZ', '002984.SZ', '300751.SZ', '300759.SZ',
    '300769.SZ', '300772.SZ', '300775.SZ', '300793.SZ', '300803.SZ', '300832.SZ',
    '300866.SZ', '300896.SZ', '300957.SZ', '300979.SZ', '300999.SZ', '001201.SZ',
    '001202.SZ', '001203.SZ', '001211.SZ', '001216.SZ', '001227.SZ', '001289.SZ',
    '001314.SZ', '001337.SZ', '001338.SZ', '001872.SZ', '001979.SZ', '002001.SZ',
    '002008.SZ', '002032.SZ', '002035.SZ', '002049.SZ', '002056.SZ', '002065.SZ',
    '002074.SZ', '002078.SZ', '002081.SZ', '002100.SZ', '002108.SZ', '002120.SZ',
    '002126.SZ', '002131.SZ', '002138.SZ', '002146.SZ', '002168.SZ', '002180.SZ',
    '002183.SZ', '002192.SZ', '002195.SZ', '002206.SZ', '002217.SZ', '002222.SZ',
    '002233.SZ', '002239.SZ', '002242.SZ', '002245.SZ', '002252.SZ', '002269.SZ',
    '002273.SZ', '002281.SZ', '002285.SZ', '002294.SZ', '002299.SZ', '002326.SZ',
    '002332.SZ', '002337.SZ', '002339.SZ', '002345.SZ', '002353.SZ', '002368.SZ',
    '002372.SZ', '002373.SZ', '002384.SZ', '002385.SZ', '002389.SZ', '002396.SZ',
    '002399.SZ', '002400.SZ', '002401.SZ', '002405.SZ', '002407.SZ', '002408.SZ',
    '002410.SZ', '002414.SZ', '002416.SZ', '002418.SZ', '002420.SZ', '002421.SZ',
    '002424.SZ', '002428.SZ', '002431.SZ', '002434.SZ', '002436.SZ', '002437.SZ',
    '002439.SZ', '002440.SZ', '002441.SZ', '002444.SZ', '002449.SZ', '002450.SZ',
    '002451.SZ', '002453.SZ', '002455.SZ', '002456.SZ', '002457.SZ', '002458.SZ',
    '002461.SZ', '002463.SZ', '002465.SZ', '002467.SZ', '002468.SZ', '002470.SZ',
    '002472.SZ', '002474.SZ', '002476.SZ', '002478.SZ', '002479.SZ', '002480.SZ',
    '002481.SZ', '002482.SZ', '002483.SZ', '002484.SZ', '002486.SZ', '002488.SZ',
    '002489.SZ', '002490.SZ', '002491.SZ', '002492.SZ', '002495.SZ', '002496.SZ',
    '002497.SZ', '002498.SZ', '002500.SZ', '002501.SZ', '002502.SZ', '002503.SZ',
    '002505.SZ', '002506.SZ', '002508.SZ', '002511.SZ', '002512.SZ', '002513.SZ',
    '002515.SZ', '002516.SZ', '002517.SZ', '002518.SZ', '002519.SZ'
]

# Use a path relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIRECTORY = os.path.join(SCRIPT_DIR, "market_data_holdout") # Changed directory name
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
