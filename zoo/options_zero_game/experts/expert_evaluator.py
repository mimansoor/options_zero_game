# zoo/options_zero_game/experts/expert_evaluator.py
# <<< The Definitive Expert Model Evaluation Suite >>>

# --- Make the script self-aware of the project structure ---
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
import pandas as pd
import argparse
import glob
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, confusion_matrix
from tqdm import tqdm

# --- Import components from your other expert scripts ---
from zoo.options_zero_game.experts.holy_trinity_trainer import create_holy_trinity_features_and_targets
from zoo.options_zero_game.experts.transformer_expert_trainer import TransformerExpert, HybridGRUTransformerExpert, CONFIG as TransformerConfig, create_sequences

def load_holdout_data(is_transformer: bool, device: torch.device):
    """Loads and processes all data from the holdout directory."""
    holdout_path = TransformerConfig['data_path'].replace('market_data_cache', 'market_data_holdout')
    if not os.path.isdir(holdout_path):
        raise FileNotFoundError(f"Holdout directory not found: '{holdout_path}'")
    
    all_files = glob.glob(os.path.join(holdout_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files in holdout directory: '{holdout_path}'")

    # For Holy Trinity, we process and return DataFrames.
    # For Transformers, we process and return sequences.
    if not is_transformer:
        processed_dfs = []
        for f in tqdm(all_files, desc="Loading Holdout Data (Holy Trinity)"):
            df = pd.read_csv(f)
            # ... (Robust data cleaning) ...
            df.rename(columns={df.columns[1] if 'Date' in df.columns else df.columns[0]: 'Close'}, inplace=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df.dropna(subset=['Close'], inplace=True)
            if len(df) > 50: # Basic length check
                processed_dfs.append(df)
        return processed_dfs
    else:
        all_X, all_y_vol, all_y_dir = [], [], []
        for f in tqdm(all_files, desc="Loading Holdout Data (Transformers)"):
            df = pd.read_csv(f)
            # ... (Robust data cleaning) ...
            df.rename(columns={df.columns[1] if 'Date' in df.columns else df.columns[0]: 'Close'}, inplace=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df.dropna(subset=['Close'], inplace=True)
            if len(df) > TransformerConfig['sequence_length'] + TransformerConfig['prediction_horizon'] + 5:
                X, y_vol, y_dir = create_sequences(df.copy(), TransformerConfig, device, vol_expert_model=None)
                if len(X) > 0:
                    all_X.append(X)
                    all_y_vol.append(y_vol)
                    all_y_dir.append(y_dir)
        if not all_X: raise ValueError("No valid sequences created from holdout data.")
        return np.concatenate(all_X, axis=0), np.concatenate(all_y_vol, axis=0), np.concatenate(all_y_dir, axis=0)

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

    all_true = {'ema': [], 'rsi': [], 'vol': []}
    all_preds = {'ema': [], 'rsi': [], 'vol': []}

    for df in holdout_dfs:
        processed_df = create_holy_trinity_features_and_targets(df.copy())
        feature_names = [col for col in processed_df.columns if 'lag' in col]
        X_test = processed_df[feature_names]
        
        all_true['ema'].extend(processed_df['ema_ratio_target'])
        all_preds['ema'].extend(ema_expert.predict(X_test))
        
        all_true['rsi'].extend(processed_df['rsi_state_target'])
        all_preds['rsi'].extend(rsi_expert.predict(X_test))

        all_true['vol'].extend(processed_df['volatility_target'])
        all_preds['vol'].extend(vol_expert.predict(X_test))

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

def evaluate_transformers(X_raw, y_vol, y_dir, device):
    """Evaluates the Volatility Analyst and the SOTA Directional Strategist."""
    print("\n" + "="*80)
    print("         EVALUATING: TRANSFORMER EXPERTS (PYTORCH)")
    print("="*80)

    # --- Volatility Analyst Evaluation ---
    print("\n--- [Report for Volatility Analyst (Regression)] ---")
    vol_params = TransformerConfig['volatility_expert_params']
    vol_model = TransformerExpert(
        input_dim=vol_params['input_dim'], model_dim=TransformerConfig['embedding_dim'],
        num_heads=TransformerConfig['num_heads'], num_layers=vol_params['num_layers'],
        output_dim=vol_params['output_dim']
    ).to(device)
    vol_model.load_state_dict(torch.load(os.path.join(TransformerConfig['model_save_path'], 'volatility_expert.pth'), map_location=device))
    vol_model.eval()

    with torch.no_grad():
        preds_vol = vol_model(torch.FloatTensor(X_raw).to(device)).cpu().numpy().flatten()
    
    r2_vol_tf = r2_score(y_vol, preds_vol)
    mae_vol_tf = mean_absolute_error(y_vol, preds_vol)
    print(f"  - R-squared (R²): {r2_vol_tf:.4f} (Explains {r2_vol_tf*100:.2f}% of future volatility variance)")
    print(f"  - MAE: {mae_vol_tf:.4f} (Average prediction error is {mae_vol_tf*100:.2f} IV points)")

    # --- Directional Strategist Evaluation ---
    print("\n--- [Report for Directional Strategist (Classification)] ---")
    print("Performing hierarchical feature fusion...")
    with torch.no_grad():
        vol_embeddings = vol_model.encode(torch.FloatTensor(X_raw).to(device)).cpu().numpy()
    vol_embeddings_tiled = np.expand_dims(vol_embeddings, axis=1)
    vol_embeddings_tiled = np.tile(vol_embeddings_tiled, (1, TransformerConfig['sequence_length'], 1))
    X_fused_test = np.concatenate([X_raw, vol_embeddings_tiled], axis=2)
    
    dir_params = TransformerConfig['directional_expert_params']
    dir_model = HybridGRUTransformerExpert(
        input_dim=dir_params['input_dim'], model_dim=TransformerConfig['embedding_dim'],
        num_heads=TransformerConfig['num_heads'], num_layers=dir_params['num_layers'],
        gru_layers=dir_params['gru_layers'], output_dim=dir_params['output_dim']
    ).to(device)
    dir_model.load_state_dict(torch.load(os.path.join(TransformerConfig['model_save_path'], 'directional_expert.pth'), map_location=device))
    dir_model.eval()

    with torch.no_grad():
        logits = dir_model(torch.FloatTensor(X_fused_test).to(device))
        preds_dir = torch.argmax(logits, dim=1).cpu().numpy()

    print(classification_report(y_dir, preds_dir, target_names=['DOWN', 'NEUTRAL', 'UP']))
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_dir, preds_dir)
    print("         Predicted ->")
    print("         DOWN  NEUTRAL   UP")
    print(f"Actual DOWN    {cm[0, 0]:>4}   {cm[0, 1]:>5}   {cm[0, 2]:>4}")
    print(f"Actual NEUTRAL {cm[1, 0]:>4}   {cm[1, 1]:>5}   {cm[1, 2]:>4}")
    print(f"Actual UP      {cm[2, 0]:>4}   {cm[2, 1]:>5}   {cm[2, 2]:>4}")
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
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.expert_type == 'holy_trinity':
        holdout_data = load_holdout_data(is_transformer=False, device=device)
        evaluate_holy_trinity(holdout_data)
    elif args.expert_type == 'transformer':
        X_raw, y_vol, y_dir = load_holdout_data(is_transformer=True, device=device)
        evaluate_transformers(X_raw, y_vol, y_dir, device)
