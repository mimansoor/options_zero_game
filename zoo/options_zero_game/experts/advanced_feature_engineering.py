# zoo/options_zero_game/experts/advanced_feature_engineering.py
# <<< DEFINITIVELY CORRECTED & COMPLETE VERSION >>>

import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.signal import hilbert
from hurst import compute_Hc
import os

# ==============================================================================
#                            CONFIGURATION
# ==============================================================================
# A central place for all feature parameters.
FEATURE_CONFIG = {
    "log_return_lag": 20,
    "ma_periods": [10, 20, 50],
    "vol_periods": [14, 30],
    "vol_autocorr_lag": 5,
    "oscillator_period": 14,
    "deviation_period": 20,
    "deviation_std_dev": 2.0,
    "channel_period": 20,
    "hurst_period": 100,
}

# ==============================================================================
#                            MAIN FEATURE FUNCTION
# ==============================================================================

def calculate_advanced_features(df: pd.DataFrame, config: dict = FEATURE_CONFIG) -> pd.DataFrame:
    df_out = df.copy()
    close = df_out['Close']
    df_out['log_return'] = np.log(close / close.shift(1))

    # <<< --- THE DEFINITIVE FIX: MINIMUM LENGTH GUARD CLAUSE --- >>>
    # The longest lookback is for the Hurst exponent (100). If we have less
    # data than that, we cannot calculate all features. In this case, we
    # return a dataframe with the correct columns but filled with zeros.
    MIN_REQUIRED_LENGTH = config["hurst_period"] + 2 # Add a small buffer
    
    if len(df_out) < MIN_REQUIRED_LENGTH:
        # Create all possible feature columns as placeholders filled with 0
        # This guarantees the DataFrame always has the same shape.
        
        # Volatility
        for n in config['vol_periods']:
            df_out[f'vol_realized_{n}'] = 0.0
            df_out[f'vol_of_vol_{n}'] = 0.0
            df_out[f'vol_autocorr_{n}'] = 0.0
        # Trend
        for n in config['ma_periods']:
            df_out[f'SMA_{n}'] = 0.0
            df_out[f'trend_ma_slope_{n}'] = 0.0
            df_out[f'trend_price_to_ma_{n}'] = 0.0
        df_out['trend_crossover'] = 0
        # Oscillator
        df_out['STOCHk_14_3_3'], df_out['STOCHd_14_3_3'], df_out['WILLR_14'], df_out['ROC_14'] = 0.0, 0.0, 0.0, 0.0
        # Deviation
        df_out[f'dev_z_score_{config["deviation_period"]}'] = 0.0
        df_out['dev_bollinger_b_pct'] = 0.0
        df_out['dev_channel_breakout'] = 0
        # Cycle
        df_out['cycle_hilbert_phase'] = 0.0
        if isinstance(df_out.index, pd.DatetimeIndex):
            df_out['cycle_day_of_week'] = 0
            df_out['cycle_month_of_year'] = 0
        # Fractal
        df_out[f'fractal_hurst_{config["hurst_period"]}'] = 0.5 # Default to neutral Hurst
        # Pattern
        for i in range(1, config['log_return_lag'] + 1):
            df_out[f'pattern_log_return_lag_{i}'] = 0.0

        # Return this safe, zero-filled DataFrame
        return df_out.fillna(0)
   
    # --- 1. Volatility Expert Features ---
    for n in config['vol_periods']:
        if len(df_out) >= n:
            df_out[f'vol_realized_{n}'] = df_out['log_return'].rolling(window=n).std()
            df_out[f'vol_of_vol_{n}'] = df_out[f'vol_realized_{n}'].rolling(window=n).std()
            df_out[f'vol_autocorr_{n}'] = df_out[f'vol_realized_{n}'].rolling(window=config['vol_autocorr_lag']).corr(df_out[f'vol_realized_{n}'].shift(config['vol_autocorr_lag']))

    # --- 2. Trend Expert Features ---
    # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
    for n in config['ma_periods']:
        # Only attempt to calculate if the DataFrame is long enough.
        if len(df_out) >= n:
            df_out.ta.sma(close=close, length=n, append=True)
            df_out[f'trend_ma_slope_{n}'] = (df_out[f'SMA_{n}'] - df_out[f'SMA_{n}'].shift(n)) / n
            df_out[f'trend_price_to_ma_{n}'] = close / df_out[f'SMA_{n}']
        else:
            # If not long enough, create placeholder NaN columns to prevent KeyErrors.
            df_out[f'SMA_{n}'] = np.nan
            df_out[f'trend_ma_slope_{n}'] = np.nan
            df_out[f'trend_price_to_ma_{n}'] = np.nan

    # This check now relies on the columns always existing.
    if 'SMA_20' in df_out.columns and 'SMA_50' in df_out.columns:
        df_out['trend_crossover'] = np.where(df_out['SMA_20'] > df_out['SMA_50'], 1, -1)
    else:
        df_out['trend_crossover'] = 0 # Neutral default

    # --- 3. Oscillator Expert Features ---
    n = config['oscillator_period']
    if len(df_out) >= n:
        df_out.ta.stoch(high=close, low=close, close=close, k=n, d=3, append=True)
        df_out.ta.willr(high=close, low=close, close=close, length=n, append=True)
        df_out.ta.roc(close=close, length=n, append=True)

    # --- 4. Deviation Expert Features ---
    n = config['deviation_period']
    k = config['deviation_std_dev']
    if len(df_out) >= n:
        df_out[f'dev_z_score_{n}'] = df_out.ta.zscore(close=close, length=n)
        bbands_df = df_out.ta.bbands(close=close, length=n, std=k)
        bbp_col_name = [col for col in bbands_df.columns if col.startswith('BBP_')]
        if bbp_col_name:
            df_out['dev_bollinger_b_pct'] = bbands_df[bbp_col_name[0]]
        
        roll_max = close.shift(1).rolling(window=config['channel_period']).max()
        roll_min = close.shift(1).rolling(window=config['channel_period']).min()
        df_out['dev_channel_breakout'] = np.where(close > roll_max, 1, np.where(close < roll_min, -1, 0))

    # --- 5. Cycle / Seasonality Expert Features (now safe) ---
    detrended = close - df_out[f'SMA_{config["ma_periods"][-1]}']
    detrended_valid = detrended.dropna()
    # Add a check to ensure detrended_valid is not empty
    if not detrended_valid.empty:
        analytic_signal = hilbert(detrended_valid)
        hilbert_phase = np.angle(analytic_signal, deg=True)
        hilbert_series = pd.Series(hilbert_phase, index=detrended_valid.index)
        df_out['cycle_hilbert_phase'] = hilbert_series
    else:
        df_out['cycle_hilbert_phase'] = np.nan # It will be filled later

    # --- 6. Fractal / Complexity Expert Features ---
    n = config["hurst_period"]
    if len(df_out) >= n:
        try:
            noise = np.random.normal(0, 1e-9, size=len(df_out['log_return']))
            hurst_input = df_out['log_return'] + noise
            df_out[f'fractal_hurst_{n}'] = hurst_input.rolling(window=n).apply(
                lambda x: compute_Hc(x)[0] if np.isfinite(x).sum() >= n else np.nan,
                raw=True
            )
        except Exception as e:
            df_out[f'fractal_hurst_{n}'] = 0.5

    # --- 7. Pattern Expert Features ---
    for i in range(1, config['log_return_lag'] + 1):
        df_out[f'pattern_log_return_lag_{i}'] = df_out['log_return'].shift(i)

    df_out.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill any placeholder columns that were created but never calculated with a neutral 0
    df_out.fillna(0, inplace=True)

    return df_out

# ==============================================================================
#                            TESTING BLOCK
# ==============================================================================
if __name__ == '__main__':
    print("--- Running Advanced Feature Engineering Test ---")

    test_file_path = "zoo/options_zero_game/data/market_data_cache/SPY.csv"

    if not os.path.exists(test_file_path):
        print(f"ERROR: Test file not found at '{test_file_path}'. Please update the path.")
    else:
        print(f"Loading data from '{test_file_path}'...")
        df = pd.read_csv(test_file_path, index_col='Date', parse_dates=True)
        df.rename(columns={df.columns[0]: 'Close'}, inplace=True)
        df = df[df['Close'] > 0].dropna()

        print("Calculating all advanced features...")
        features_df = calculate_advanced_features(df)

        print("\n--- Feature Calculation Complete ---")
        print(f"Original shape: {df.shape}")
        print(f"New shape:      {features_df.shape}")

        print("\n--- Sample of the final DataFrame (last 5 rows): ---")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        # We explicitly select a few columns to check, including the fixed one.
        print(features_df[['Close', 'dev_z_score_20', 'dev_bollinger_b_pct', 'fractal_hurst_100']].tail())

