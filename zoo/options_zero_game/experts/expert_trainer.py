# zoo/options_zero_game/experts/expert_trainer.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import joblib
import os
import glob

# --- Configuration ---
# This defines the "lookback" for features and the "horizon" for predictions.
CONFIG = {
    "data_path": "zoo/options_zero_game/data/market_data_cache",
    "lookback_window": 20,  # How many past steps to use as features
    "trend_horizon": 5,     # Predict the trend 5 steps into the future
    "vol_horizon": 1,       # Predict the volatility of the very next step
    "trend_flat_threshold": 0.005, # +/- 0.5% change is considered "FLAT"
    "model_save_path": "zoo/options_zero_game/experts/"
}

def create_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers features and target labels from a raw price series."""

    # --- Feature Engineering (Unchanged) ---
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

    for i in range(1, CONFIG['lookback_window'] + 1):
        df[f'log_return_lag_{i}'] = df['log_return'].shift(i)

    df['sma'] = df['Close'].rolling(window=CONFIG['lookback_window']).mean()
    df['momentum'] = (df['Close'] / df['sma']) - 1.0

    # --- Target Engineering (Corrected) ---

    # 1. Trend Target (Unchanged)
    future_price = df['Close'].shift(-CONFIG['trend_horizon'])
    price_change_pct = (future_price / df['Close']) - 1.0

    conditions = [
        price_change_pct > CONFIG['trend_flat_threshold'],
        price_change_pct < -CONFIG['trend_flat_threshold'],
    ]
    choices = [2, 0] # 2 = UP, 0 = DOWN
    df['trend_target'] = np.select(conditions, choices, default=1) # 1 = FLAT

    # --- THE FIX IS HERE ---
    # 2. Volatility Target (Corrected)
    # The new target is the absolute value of the next step's log return.
    # This is a much more robust proxy for next-step volatility.
    df['vol_target'] = df['log_return'].abs().shift(-1)

    return df

def main():
    print("--- Starting Expert Model Training ---")

    all_files = glob.glob(os.path.join(CONFIG['data_path'], "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {CONFIG['data_path']}. Please run the cache_builder.py script first.")

    all_processed_data = []
    print(f"Found {len(all_files)} data files to process...")

    for f in all_files:
        try:
            data = pd.read_csv(f)
            # --- THE FIX: More robust column renaming ---
            # Ensure the second column is named 'Close' regardless of its original name
            if len(data.columns) > 1:
                data.rename(columns={data.columns[1]: 'Close'}, inplace=True)
            else:
                print(f"Warning: Skipping file {f} as it does not have at least two columns.")
                continue

            initial_rows = len(data)
            if initial_rows < CONFIG['lookback_window'] + CONFIG['trend_horizon'] + 1:
                print(f"Skipping {os.path.basename(f)}: Only has {initial_rows} rows, needs at least {CONFIG['lookback_window'] + CONFIG['trend_horizon'] + 1}.")
                continue

            processed_df = create_features_and_targets(data)
            processed_df.dropna(inplace=True)

            final_rows = len(processed_df)
            if final_rows > 0:
                # Optional: Uncomment the line below for very detailed debugging
                # print(f"Processed {os.path.basename(f)}: {initial_rows} rows -> {final_rows} valid samples.")
                all_processed_data.append(processed_df)
            else:
                print(f"Warning: No valid samples generated from {os.path.basename(f)} ({initial_rows} rows). It might be too short or contain too many NaNs.")

        except Exception as e:
            print(f"Warning: Could not process file {f}. Error: {e}")

    if not all_processed_data:
        raise ValueError("No valid data could be processed. All historical data files were likely too short for the current lookback/horizon settings.")

    final_dataset = pd.concat(all_processed_data)

    print(f"\nCreated {len(final_dataset)} total samples for training after processing all files.")
   
    # 4. Define features (X) and targets (y)
    feature_names = [col for col in final_dataset.columns if 'lag' in col or 'momentum' in col]
    X = final_dataset[feature_names]
    
    # --- Train Trend Prediction Model ---
    print("\n--- Training Trend Classifier (UP/DOWN/FLAT) ---")
    y_trend = final_dataset['trend_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y_trend, test_size=0.2, random_state=42)
    
    trend_model = lgb.LGBMClassifier(objective='multiclass', num_class=3, random_state=42)
    trend_model.fit(X_train, y_train)
    
    y_pred = trend_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Trend Model Accuracy: {accuracy * 100:.2f}%")
    
    trend_model_path = os.path.join(CONFIG['model_save_path'], 'trend_expert.joblib')
    joblib.dump(trend_model, trend_model_path)
    print(f"Trend model saved to {trend_model_path}")
    
    # --- Train Volatility Prediction Model ---
    print("\n--- Training Volatility Regressor ---")
    y_vol = final_dataset['vol_target']
    X_train, X_test, y_train, y_test = train_test_split(X, y_vol, test_size=0.2, random_state=42)
    
    vol_model = lgb.LGBMRegressor(objective='regression_l1', random_state=42)
    vol_model.fit(X_train, y_train)
    
    y_pred = vol_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Volatility Model R-squared: {r2:.3f}")

    vol_model_path = os.path.join(CONFIG['model_save_path'], 'volatility_expert.joblib')
    joblib.dump(vol_model, vol_model_path)
    print(f"Volatility model saved to {vol_model_path}")

    print("\n--- Expert Model Training Complete! ---")

if __name__ == "__main__":
    main()
