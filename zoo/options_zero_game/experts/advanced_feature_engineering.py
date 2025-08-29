# zoo/options_zero_game/experts/advanced_feature_engineering.py
# <<< DEFINITIVE, COMPLETE, AND ARCHITECTURALLY ROBUST SCRIPT >>>

import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy.signal import hilbert
from hurst import compute_Hc
import os

# ==============================================================================
#                            CONFIGURATION
# ==============================================================================
FEATURE_CONFIG = {
    "log_return_lag": 20,
    "ma_periods": [10, 20, 50],
    "vol_periods": [14, 30],
    "vol_autocorr_lag": 5,
    "oscillator_period": 14,
    "deviation_period": 20,
    "deviation_std_dev": 2.0,
    "channel_period": 20,
    "hurst_period": 60, # Synchronized with expert_sequence_length
}

# ==============================================================================
#                            MAIN FEATURE FUNCTION
# ==============================================================================
def calculate_advanced_features(df: pd.DataFrame, config: dict = FEATURE_CONFIG) -> pd.DataFrame:
    """
    Takes a DataFrame with a 'Close' column and engineers a comprehensive suite
    of advanced features. This function is hardened to handle short data series
    without crashing by guaranteeing a consistent output shape.
    """
    df_out = df.copy()
    close = df_out['Close']

    # --- Step 1: Unconditionally create ALL possible columns with a default NaN value ---
    # This is the core of the fix and guarantees a consistent DataFrame shape.
    df_out['log_return'] = np.nan
    for n in config['vol_periods']:
        df_out[f'vol_realized_{n}'], df_out[f'vol_of_vol_{n}'], df_out[f'vol_autocorr_{n}'] = [np.nan]*3
    for n in config['ma_periods']:
        df_out[f'SMA_{n}'], df_out[f'trend_ma_slope_{n}'], df_out[f'trend_price_to_ma_{n}'] = [np.nan]*3
    df_out['trend_crossover'] = np.nan
    df_out['STOCHk_14_3_3'], df_out['STOCHd_14_3_3'], df_out['WILLR_14'], df_out['ROC_14'] = [np.nan]*4
    df_out[f'dev_z_score_{config["deviation_period"]}'] = np.nan
    df_out['dev_bollinger_b_pct'], df_out['dev_channel_breakout'] = np.nan, np.nan
    df_out['cycle_hilbert_phase'], df_out['cycle_day_of_week'], df_out['cycle_month_of_year'] = [np.nan]*3
    df_out[f'fractal_hurst_{config["hurst_period"]}'] = np.nan
    for i in range(1, config['log_return_lag'] + 1):
        df_out[f'pattern_log_return_lag_{i}'] = np.nan

    # --- Step 2: Calculate features and fill columns if data is sufficient ---
    df_out['log_return'] = np.log(close / close.shift(1))
    
    # Volatility
    for n in config['vol_periods']:
        if len(df_out) >= n:
            df_out[f'vol_realized_{n}'] = df_out['log_return'].rolling(window=n).std()
            df_out[f'vol_of_vol_{n}'] = df_out[f'vol_realized_{n}'].rolling(window=n).std()
            if len(df_out) >= config['vol_autocorr_lag']:
                 df_out[f'vol_autocorr_{n}'] = df_out[f'vol_realized_{n}'].rolling(window=config['vol_autocorr_lag']).corr(df_out[f'vol_realized_{n}'].shift(config['vol_autocorr_lag']))

    # Trend
    for n in config['ma_periods']:
        if len(df_out) >= n:
            df_out.ta.sma(close=close, length=n, append=True, col_names=(f'SMA_{n}',))
            df_out[f'trend_ma_slope_{n}'] = (df_out[f'SMA_{n}'] - df_out[f'SMA_{n}'].shift(n)) / n
            df_out[f'trend_price_to_ma_{n}'] = close / df_out[f'SMA_{n}']
            
    if len(df_out) >= 50: # Check for the longest MA period
        df_out['trend_crossover'] = np.where(df_out['SMA_20'] > df_out['SMA_50'], 1, -1)

    # Oscillator
    n = config['oscillator_period']
    if len(df_out) >= n:
        df_out.ta.stoch(high=close, low=close, close=close, k=n, d=3, append=True)
        df_out.ta.willr(high=close, low=close, close=close, length=n, append=True)
        df_out.ta.roc(close=close, length=n, append=True)

    # Deviation
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

    # Cycle / Seasonality
    if len(df_out) >= config['ma_periods'][-1]:
        detrended = close - df_out[f'SMA_{config["ma_periods"][-1]}']
        detrended_valid = detrended.dropna()
        if not detrended_valid.empty:
            analytic_signal = hilbert(detrended_valid)
            hilbert_phase = np.angle(analytic_signal, deg=True)
            hilbert_series = pd.Series(hilbert_phase, index=detrended_valid.index)
            df_out['cycle_hilbert_phase'] = hilbert_series
            
    if isinstance(df_out.index, pd.DatetimeIndex):
        df_out['cycle_day_of_week'] = df_out.index.dayofweek
        df_out['cycle_month_of_year'] = df_out.index.month
    else:
        df_out['cycle_day_of_week'] = 0
        df_out['cycle_month_of_year'] = 1

    # Fractal
    n = config["hurst_period"]
    if len(df_out) >= n:
        try:
            noise = np.random.normal(0, 1e-9, size=len(df_out['log_return']))
            hurst_input = df_out['log_return'].fillna(0) + noise
            df_out[f'fractal_hurst_{n}'] = hurst_input.rolling(window=n).apply(
                lambda x: compute_Hc(x)[0] if np.isfinite(x).sum() >= n else np.nan, 
                raw=True
            )
        except Exception:
             df_out[f'fractal_hurst_{n}'] = 0.5

    # Pattern
    for i in range(1, config['log_return_lag'] + 1):
        df_out[f'pattern_log_return_lag_{i}'] = df_out['log_return'].shift(i)
        
    # --- Final Sanitation ---
    df_out.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_out.fillna(0, inplace=True) # Fill any remaining NaNs with a neutral 0
    
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
        
        sample_cols = [
            'Close', 'dev_z_score_20', 'dev_bollinger_b_pct', 'cycle_day_of_week', 
            'fractal_hurst_60', 'pattern_log_return_lag_1'
        ]
        # Ensure all sample columns exist before trying to display them
        display_cols = [col for col in sample_cols if col in features_df.columns]
        print(features_df[display_cols].tail())
