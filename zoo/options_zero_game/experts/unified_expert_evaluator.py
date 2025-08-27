# zoo/options_zero_game/experts/unified_expert_evaluator.py
# <<< DEFINITIVELY CORRECTED VERSION with BATCHED EVALUATION >>>

import torch
from torch.utils.data import DataLoader, TensorDataset # <-- Import DataLoader
import numpy as np
import pandas as pd
import argparse
import glob
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from tqdm import tqdm
import sys
import warnings # <-- Import the warnings library

# <<< --- THE DEFINITIVE FIX IS HERE --- >>>
# This will specifically ignore the "X has feature names" warning from StandardScaler.
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.base')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import the necessary components from the other scripts for consistency
from zoo.options_zero_game.experts.advanced_feature_engineering import calculate_advanced_features
from zoo.options_zero_game.experts.unified_expert_trainer import MLPExpert, EXPERT_FEATURE_SETS, CONFIG

def main(expert_name: str):
    """
    Loads a trained MLP expert and its scaler, evaluates it on all holdout
    data in batches, and prints a detailed performance report.
    """
    print(f"--- Starting Unified Evaluation for: {expert_name.upper()} EXPERT ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Pre-trained Model and Scaler ---
    model_path = os.path.join(CONFIG['model_save_path'], f'{expert_name}_mlp_expert.pth')
    scaler_path = os.path.join(CONFIG['model_save_path'], f'{expert_name}_mlp_scaler.joblib')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Model or scaler not found for expert '{expert_name}'. Please train it first.")

    feature_cols = EXPERT_FEATURE_SETS[expert_name]
    input_dim = len(feature_cols)
    is_regression = expert_name == 'volatility'
    output_dim = 1 if is_regression else 3

    model = MLPExpert(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    scaler = joblib.load(scaler_path)
    print("Successfully loaded model and feature scaler.")

    # --- 2. Load and Prepare Holdout Data (unchanged) ---
    holdout_path = CONFIG['data_path'].replace('market_data_cache', 'market_data_holdout')
    all_files = glob.glob(os.path.join(holdout_path, "*.csv"))
    
    all_processed_dfs = []
    for f in tqdm(all_files, desc="Processing Holdout Data Files"):
        try:
            df = pd.read_csv(f, index_col='Date', parse_dates=True)
            close_col = next((col for col in df.columns if 'Close' in col or 'close' in col), df.columns[0])
            df.rename(columns={close_col: 'Close'}, inplace=True)
            df = df[df['Close'] > 0].dropna(subset=['Close'])
            if len(df) < 200: continue
            features_df = calculate_advanced_features(df)
            all_processed_dfs.append(features_df)
        except Exception as e:
            print(f"Warning: Skipping holdout file {f} due to error: {e}")
    
    if not all_processed_dfs:
        raise ValueError("No holdout data could be processed.")
        
    all_data_df = pd.concat(all_processed_dfs)

    # --- 3. Define Target Variable for Holdout Set ---
    if is_regression:
        all_data_df['target'] = all_data_df['vol_realized_14'].shift(-CONFIG['prediction_horizon'])
    else:
        future_return = np.log(all_data_df['Close'].shift(-CONFIG['prediction_horizon']) / all_data_df['Close'])
        conditions = [future_return > CONFIG['directional_threshold'], future_return < -CONFIG['directional_threshold']]
        choices = [2, 0]
        all_data_df['target'] = np.select(conditions, choices, default=1)

    # --- 4. Finalize Dataset (unchanged) ---
    final_df = all_data_df[feature_cols + ['target']].dropna()
    X_holdout = final_df[feature_cols].values
    y_holdout = final_df['target'].values
    X_holdout_scaled = scaler.transform(X_holdout)

    # --- 5. Get Model Predictions in Batches ---
    eval_batch_size = 512 # Can be larger than training batch size
    X_holdout_tensor = torch.FloatTensor(X_holdout_scaled)
    eval_dataset = TensorDataset(X_holdout_tensor)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for (inputs,) in tqdm(eval_loader, desc=f"Evaluating {expert_name.capitalize()} Expert"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            if is_regression:
                preds = outputs.cpu().numpy().flatten()
            else:
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
    
    final_preds = np.array(all_preds)

    # --- 6. Print Performance Report ---
    print("\n" + "="*80)
    print(f"               PERFORMANCE REPORT FOR: {expert_name.upper()} EXPERT")
    print("="*80)
    if is_regression:
        r2 = r2_score(y_holdout, final_preds)
        print(f"  - R-squared (R²): {r2:.4f} (Explains {r2*100:.2f}% of future variance)")
    else:
        # We will handle the zero division by printing a more robust confusion matrix.
        print(classification_report(y_holdout, final_preds, target_names=['DOWN', 'NEUTRAL', 'UP'], zero_division=0))

        print("--- Confusion Matrix ---")
        cm = confusion_matrix(y_holdout, final_preds, labels=[0, 1, 2]) # Ensure all labels are present
        print("         Predicted ->")
        print("         DOWN  NEUTRAL   UP")
        print(f"Actual DOWN    {cm[0, 0]:>4}   {cm[0, 1]:>5}   {cm[0, 2]:>4}")
        print(f"Actual NEUTRAL {cm[1, 0]:>4}   {cm[1, 1]:>5}   {cm[1, 2]:>4}")
        print(f"Actual UP      {cm[2, 0]:>4}   {cm[2, 1]:>5}   {cm[2, 2]:>4}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a member of the council of MLP experts.")
    parser.add_argument(
        '--expert',
        type=str,
        required=True,
        choices=list(EXPERT_FEATURE_SETS.keys()),
        help="The name of the expert to evaluate."
    )
    args = parser.parse_args()
    main(args.expert)
