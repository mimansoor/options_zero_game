# zoo/options_zero_game/experts/expert_evaluator.py
# <<< FINAL, CORRECTED VERSION >>>

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import argparse
import glob
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

from zoo.options_zero_game.experts.holy_trinity_trainer import create_holy_trinity_features_and_targets
from zoo.options_zero_game.experts.transformer_expert_trainer import (
    TransformerExpert, CONFIG as TransformerConfig, create_sequences
)

def load_holdout_data(is_transformer: bool, device: torch.device):
    """Loads and processes all data from the holdout directory."""
    holdout_path = TransformerConfig['data_path'].replace('market_data_cache', 'market_data_holdout')
    if not os.path.isdir(holdout_path):
        raise FileNotFoundError(f"Holdout directory not found: '{holdout_path}'")
    
    all_files = glob.glob(os.path.join(holdout_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files in holdout directory: '{holdout_path}'")

    if not is_transformer:
        # This part remains for evaluating the old LightGBM models
        processed_dfs = []
        for f in tqdm(all_files, desc="Loading Holdout Data (Holy Trinity)"):
            df = pd.read_csv(f)
            df.rename(columns={df.columns[1] if 'Date' in df.columns else df.columns[0]: 'Close'}, inplace=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df.dropna(subset=['Close'], inplace=True)
            if len(df) > 50:
                processed_dfs.append(df)
        return processed_dfs
    else:
        # This part is now dedicated to preparing data for the Volatility Expert
        all_X_vol, all_y_vol = [], []
        for f in tqdm(all_files, desc="Loading Holdout Data (Transformers)"):
            try:
                df = pd.read_csv(f, index_col='Date', parse_dates=True)
                df.rename(columns={df.columns[0]: 'Close'}, inplace=True)
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                df = df[df['Close'] > 0].dropna(subset=['Close'])

                if len(df) > TransformerConfig['sequence_length'] + TransformerConfig['prediction_horizon'] + 35:
                    # Use the exact same, robust function from the trainer
                    X_vol, y_vol_i = create_sequences(df.copy(), TransformerConfig, device, model_type='volatility')
                    
                    if len(X_vol) > 0:
                        all_X_vol.append(X_vol)
                        all_y_vol.append(y_vol_i)
            except Exception as e:
                 print(f"Warning: Failed to process holdout file {f}. Error: {e}")

        if not all_X_vol: raise ValueError("No valid sequences created from holdout data.")
        
        return np.concatenate(all_X_vol, axis=0), np.concatenate(all_y_vol, axis=0)

def evaluate_holy_trinity(holdout_dfs: list):
    """Evaluates all three LightGBM models from the Holy Trinity."""
    print("\n" + "="*80)
    print("         EVALUATING: HOLY TRINITY EXPERTS (LIGHTGBM)")
    print("="*80)

    try:
        ema_expert = joblib.load('zoo/options_zero_game/experts/ema_expert.joblib')
        rsi_expert = joblib.load('zoo/options_zero_game/experts/rsi_expert.joblib')
        vol_expert = joblib.load('zoo/options_zero_game/experts/volatility_expert.joblib')
    except FileNotFoundError as e:
        print(f"ERROR: Could not load Holy Trinity model. Have you trained them yet? Details: {e}")
        return

    # Create clean lists to store the validated data
    all_true = {'ema': [], 'rsi': [], 'vol': []}
    all_preds = {'ema': [], 'rsi': [], 'vol': []}

    # --- THE FIX: A new, robust data processing loop ---
    for df in tqdm(holdout_dfs, desc="Processing Holdout DataFrames"):
        # 1. Create features AND targets. This will introduce NaNs.
        processed_df = create_holy_trinity_features_and_targets(df.copy())

        # 2. CRITICAL: Drop all rows with any NaN values.
        # This is the step that was missing and caused the crash.
        processed_df.dropna(inplace=True)

        # Failsafe: if the df is empty after cleaning, skip it.
        if processed_df.empty:
            continue

        # 3. Extract features from the now-clean DataFrame
        feature_names = [col for col in processed_df.columns if 'lag' in col]
        X_test = processed_df[feature_names]

        # 4. Append the CLEANED ground truth and predictions to the lists
        all_true['ema'].extend(processed_df['ema_ratio_target'])
        all_preds['ema'].extend(ema_expert.predict(X_test))

        all_true['rsi'].extend(processed_df['rsi_state_target'])
        all_preds['rsi'].extend(rsi_expert.predict(X_test))

        all_true['vol'].extend(processed_df['volatility_target'])
        all_preds['vol'].extend(vol_expert.predict(X_test))

    # --- The rest of the function is now guaranteed to work ---

    # Failsafe for the case where no valid data was found in any file
    if not all_true['ema']:
        print("\nERROR: No valid, non-NaN data was found in the holdout set to evaluate.")
        return

    # --- EMA Trend Expert Report ---
    print("\n--- [Report for EMA Trend Expert (Regression)] ---")
    r2_ema = r2_score(all_true['ema'], all_preds['ema'])
    print(f"  - R-squared (R²): {r2_ema:.4f} (Explains {r2_ema*100:.2f}% of future EMA ratio variance)")

    # --- Volatility Expert Report ---
    print("\n--- [Report for Volatility Expert (Regression)] ---")
    r2_vol = r2_score(all_true['vol'], all_preds['vol'])
    mae_vol = mean_absolute_error(all_true['vol'], all_preds['vol'])
    print(f"  - R-squared (R²): {r2_vol:.4f} (Explains {r2_vol*100:.2f}% of future volatility variance)")
    print(f"  - MAE: {mae_vol:.4f} (Average prediction error is {mae_vol*100:.2f} IV points)")

    # --- RSI State Expert Report ---
    print("\n--- [Report for RSI State Expert (Classification)] ---")
    print(classification_report(all_true['rsi'], all_preds['rsi'], target_names=['OVERSOLD', 'NEUTRAL', 'OVERBOUGHT']))
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(all_true['rsi'], all_preds['rsi'])
    print("         Predicted ->")
    print("         SOLD   NEUTRAL   BOUGHT")
    print(f"Actual SOLD    {cm[0, 0]:>4}   {cm[0, 1]:>5}   {cm[0, 2]:>4}")
    print(f"Actual NEUTRAL {cm[1, 0]:>4}   {cm[1, 1]:>5}   {cm[1, 2]:>4}")
    print(f"Actual BOUGHT  {cm[2, 0]:>4}   {cm[2, 1]:>5}   {cm[2, 2]:>4}")
    print("="*80)

def evaluate_transformers(X_raw_vol, y_vol, device, model_path: str): # <-- Add model_path argument
    """
    Evaluates the Volatility Analyst. Processes data in batches to prevent OOM errors.
    """
    print("\n" + "="*80)
    print("         EVALUATING: TRANSFORMER EXPERTS (PYTORCH)")
    print("="*80)

    # --- Volatility Analyst Evaluation ---
    print("\n--- [Report for Volatility Analyst (Regression)] ---")
    print(f"Loading model from: {model_path}")
    vol_params = TransformerConfig['volatility_expert_params']
    vol_model = TransformerExpert(
        input_dim=vol_params['input_dim'], model_dim=TransformerConfig['embedding_dim'],
        num_heads=TransformerConfig['num_heads'], num_layers=vol_params['num_layers'],
        output_dim=vol_params['output_dim']
    ).to(device)
    vol_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    vol_model.eval()

    # 1. Create a DataLoader for the holdout set.
    eval_batch_size = 256
    X_tensor = torch.FloatTensor(X_raw_vol)
    eval_dataset = TensorDataset(X_tensor) # We only need X for prediction
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)

    # 2. Loop through the data in batches to get predictions.
    all_preds_vol = []
    with torch.no_grad():
        for (inputs,) in tqdm(eval_loader, desc="Evaluating Volatility Model"): # Note the comma: (inputs,)
            inputs = inputs.to(device)
            outputs = vol_model(inputs).cpu().numpy().flatten()
            all_preds_vol.extend(outputs)

    preds_vol = np.array(all_preds_vol)

    assert np.all(np.isfinite(preds_vol)), "FATAL: Model produced NaN/Inf predictions during evaluation."

    r2_vol_tf = r2_score(y_vol, preds_vol)
    mae_vol_tf = mean_absolute_error(y_vol, preds_vol)
    print(f"  - R-squared (R²): {r2_vol_tf:.4f} (Explains {r2_vol_tf*100:.2f}% of future volatility variance)")
    print(f"  - MAE: {mae_vol_tf:.4f} (Average prediction error is {mae_vol_tf*100:.2f} IV points)")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all expert models on holdout data.")
    parser.add_argument(
        '--expert_type', 
        type=str, 
        required=True, 
        choices=['holy_trinity', 'transformer'],
        help="The type of expert models to evaluate."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        # Default to the standard final model name
        default=os.path.join(TransformerConfig['model_save_path'], 'volatility_expert.pth'),
        help="Path to the specific model checkpoint to evaluate."
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    if args.expert_type == 'transformer':
        X_vol, y_vol = load_holdout_data(is_transformer=True, device=device)
        # Pass the model_path from the arguments to the function
        evaluate_transformers(X_vol, y_vol, device, model_path=args.model_path)
    else:
        holdout_data = load_holdout_data(is_transformer=False, device=device)
        evaluate_holy_trinity(holdout_data)
