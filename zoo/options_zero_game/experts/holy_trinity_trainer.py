# zoo/options_zero_game/experts/holy_trinity_trainer.py
# <<< CORRECTED VERSION: Uses Standard Deviation of Log Returns for Volatility >>>

import pandas as pd
import numpy as np
import pandas_ta as ta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import joblib
import os
import glob

# --- Configuration for the Holy Trinity ---
CONFIG = {
    "data_path": "zoo/options_zero_game/data/market_data_cache",
    "lookback_window": 30,
    "prediction_horizon": 5,
    "model_save_path": "zoo/options_zero_game/experts/",
    # Indicator Parameters
    "ema_short_period": 12,
    "ema_long_period": 26,
    "rsi_period": 14,
    "volatility_period": 14, # Replaces ATR period
    # RSI Target Thresholds
    "rsi_overbought": 70,
    "rsi_oversold": 30,
}

def create_holy_trinity_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers features and targets based on EMA, RSI, and Historical Volatility."""

    # --- 1. Calculate Core Indicators (All from 'Close' price) ---
    df.ta.ema(length=CONFIG['ema_short_period'], append=True)
    df.ta.ema(length=CONFIG['ema_long_period'], append=True)
    df.ta.rsi(length=CONFIG['rsi_period'], append=True)

    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    # --- THE FIX: Calculate Historical Volatility instead of ATR ---
    df['volatility'] = df['log_return'].rolling(window=CONFIG['volatility_period']).std()

    # --- 2. Feature Engineering ---
    df['ema_ratio'] = df[f'EMA_{CONFIG["ema_short_period"]}'] / df[f'EMA_{CONFIG["ema_long_period"]}']

    # Lagged features for the model
    for i in range(1, CONFIG['lookback_window'] + 1):
        df[f'log_return_lag_{i}'] = df['log_return'].shift(i)
        df[f'rsi_lag_{i}'] = df[f'RSI_{CONFIG["rsi_period"]}'].shift(i)

    # --- 3. Target Engineering ---
    # a) EMA Trend Target
    df['ema_ratio_target'] = df['ema_ratio'].shift(-CONFIG['prediction_horizon'])

    # b) RSI State Target
    future_rsi = df[f'RSI_{CONFIG["rsi_period"]}'].shift(-CONFIG['prediction_horizon'])
    conditions = [future_rsi > CONFIG['rsi_overbought'], future_rsi < CONFIG['rsi_oversold']]
    choices = [2, 0] # 2=OVERBOUGHT, 0=OVERSOLD
    df['rsi_state_target'] = np.select(conditions, choices, default=1) # 1=NEUTRAL

    # c) Volatility Target
    df['volatility_target'] = df['volatility'].shift(-CONFIG['prediction_horizon'])

    return df

def main():
    """
    The complete, correct main function to train and save the Holy Trinity experts.
    """
    print("--- Starting Holy Trinity Expert Model Training ---")
    
    all_files = glob.glob(os.path.join(CONFIG['data_path'], "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {CONFIG['data_path']}. Please run the cache_builder.py script first.")
        
    all_processed_data = []
    print(f"Found {len(all_files)} data files to process...")

    for f in all_files:
        try:
            data = pd.read_csv(f)
            if len(data.columns) > 1:
                data.rename(columns={data.columns[1]: 'Close'}, inplace=True)
            else:
                continue
            
            initial_rows = len(data)
            # This check ensures we have enough data for all lookbacks and horizons
            if initial_rows < CONFIG['lookback_window'] + CONFIG['prediction_horizon'] + 1:
                continue

            processed_df = create_holy_trinity_features_and_targets(data)
            processed_df.dropna(inplace=True)
            
            if len(processed_df) > 0:
                all_processed_data.append(processed_df)
        except Exception as e:
            print(f"Warning: Could not process file {f}. Error: {e}")

    if not all_processed_data:
        raise ValueError("No valid data could be processed from any of the historical data files.")
        
    # --- THIS IS THE CRITICAL PART ---
    # `final_dataset` is created here, before it is used.
    final_dataset = pd.concat(all_processed_data)
    print(f"\nCreated {len(final_dataset)} total samples for training after processing all files.")
    
    # Define the features to be used for all models
    feature_names = [col for col in final_dataset.columns if 'lag' in col]
    X = final_dataset[feature_names]
    
    # --- Train the 3 Holy Trinity Models ---
    
    # 1. EMA Trend Expert
    print("\n--- Training EMA Trend Expert (Regressor) ---")
    y_ema = final_dataset['ema_ratio_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y_ema, test_size=0.2, random_state=42)
    ema_model = lgb.LGBMRegressor(objective='regression_l1', random_state=42)
    ema_model.fit(X_train, y_train)
    print(f"EMA Model R-squared: {r2_score(y_test, ema_model.predict(X_test)):.3f}")
    joblib.dump(ema_model, os.path.join(CONFIG['model_save_path'], 'ema_expert.joblib'))
    print("EMA Trend Expert saved.")

    # 2. RSI State Expert
    print("\n--- Training RSI State Expert (Classifier) ---")
    y_rsi = final_dataset['rsi_state_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y_rsi, test_size=0.2, random_state=42)
    rsi_model = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
    rsi_model.fit(X_train, y_train)
    print(f"RSI Model Accuracy: {accuracy_score(y_test, rsi_model.predict(X_test)) * 100:.2f}%")
    joblib.dump(rsi_model, os.path.join(CONFIG['model_save_path'], 'rsi_expert.joblib'))
    print("RSI State Expert saved.")

    # 3. Volatility Expert
    print("\n--- Training Volatility Expert (Regressor) ---")
    y_vol = final_dataset['volatility_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y_vol, test_size=0.2, random_state=42)
    vol_model = lgb.LGBMRegressor(objective='regression_l1', random_state=42)
    vol_model.fit(X_train, y_train)
    print(f"Volatility Model R-squared: {r2_score(y_test, vol_model.predict(X_test)):.3f}")
    joblib.dump(vol_model, os.path.join(CONFIG['model_save_path'], 'volatility_expert.joblib'))
    print("Volatility Expert saved.")

    print("\n--- Holy Trinity Expert Model Training Complete! ---")

if __name__ == "__main__":
    main()
