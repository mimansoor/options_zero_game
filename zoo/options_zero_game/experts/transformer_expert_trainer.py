# zoo/options_zero_game/experts/transformer_expert_trainer.py
# <<< DEFINITIVE SOTA VERSION with Training and Evaluation Modes >>>

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import glob
import os
import math
import random
import joblib
import shutil
import warnings # <-- Import the warnings library

# <<< --- THE DEFINITIVE FIX IS HERE --- >>>
# This will specifically ignore the "X has feature names" warning from StandardScaler.
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn.base')

from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Import all necessary components from our other scripts
from zoo.options_zero_game.experts.advanced_feature_engineering import calculate_advanced_features
from zoo.options_zero_game.experts.unified_expert_trainer import MLPExpert, EXPERT_FEATURE_SETS as MLP_FEATURE_SETS

# ==============================================================================
#                            CONFIGURATION
# ==============================================================================
CONFIG = {
    "data_path": "zoo/options_zero_game/data/market_data_cache",
    "model_save_path": "zoo/options_zero_game/experts/",
    "embedding_dim": 128,
    "num_heads": 4,
    "dropout": 0.1,

    "volatility_expert_params": {
        "num_layers": 3,
        "input_dim": 4,  # [log_return, volatility, vol_of_vol, vol_autocorr]
        "output_dim": 1,
    },
    "directional_expert_params": {
        "num_layers": 4,
        "input_dim": 128 + (64 * 5), # Volatility (128) + 5 MLP Experts (64*5=320) = 448
        "output_dim": 3,
    },

    "sequence_length": 60,
    "prediction_horizon": 5,
    "batch_size": 64,
    "epochs": 10,
    "lr": 1e-4,

    # Feature engineering parameters are now self-contained
    "volatility_period": 14,
    "vol_of_vol_period": 14,
    "vol_autocorr_lag": 5,
    "directional_threshold": 0.005,
}

# ==============================================================================
#                          MODEL ARCHITECTURES
# ==============================================================================
class TransformerExpert(nn.Module):
    """ Standard Transformer Encoder. Uses the final token's output for robust predictions. """
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.input_embedding = nn.Linear(input_dim, model_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_head = nn.Linear(model_dim, output_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        embedding_sequence = self.encode(src)
        final_embedding = embedding_sequence[:, -1, :]
        return self.output_head(final_embedding)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        x = self.input_embedding(src) * math.sqrt(self.model_dim)
        return self.transformer_encoder(x)

# ==============================================================================
#                         DATA PREPARATION PIPELINE
# ==============================================================================
def create_sequences(df: pd.DataFrame, config: dict, device: torch.device, model_type: str, experts: dict = None):
    """
    Self-contained function to generate features and create sequences for a given DataFrame.
    """
    # --- Step 1: Base Feature Engineering & Sanitation ---
    df['Close'].replace(0, np.nan, inplace=True)
    df.dropna(subset=['Close'], inplace=True)
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # --- Step 2: Calculate All Features Needed for ALL Experts ---
    # This ensures consistency. We calculate everything, then select what we need.
    df['volatility'] = df['log_return'].rolling(window=config['volatility_period']).std()
    df['vol_of_vol'] = df['volatility'].rolling(window=config['vol_of_vol_period']).std()
    df['vol_autocorr'] = df['volatility'].rolling(window=config['vol_autocorr_lag']).corr(df['volatility'].shift(config['vol_autocorr_lag']))
    
    # Define target variables for both models
    df['volatility_target'] = df['volatility'].shift(-config['prediction_horizon'])
    future_return = np.log(df['Close'].shift(-config['prediction_horizon']) / df['Close'])
    df['target'] = np.select([future_return > config['directional_threshold'], future_return < -config['directional_threshold']], [2, 0], default=1)

    # --- Step 3: Dispatch to the Correct Sequence Creation Logic ---
    if model_type == 'volatility':
        feature_cols = ['log_return', 'volatility', 'vol_of_vol', 'vol_autocorr']
        target_col = 'volatility_target'
        final_df = df[feature_cols + [target_col]].dropna()
        
        feature_data = final_df[feature_cols].values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        targets = final_df[target_col].values

    elif model_type == 'directional':
        if not experts: raise ValueError("Experts dictionary must be provided for directional model.")
        
        # We need to calculate ALL features for the MLP experts to use
        # This is now safe because this function is self-contained.
        from zoo.options_zero_game.experts.advanced_feature_engineering import calculate_advanced_features
        features_df = calculate_advanced_features(df)
        features_df['target'] = df['target'] # Copy target over
        
        all_embeddings = {}
        with torch.no_grad():
            # Generate Volatility Embeddings
            vol_data = features_df[ ['log_return', 'volatility', 'vol_of_vol', 'vol_autocorr'] ].dropna()
            if len(vol_data) >= config['sequence_length']:
                sequences = np.array([vol_data.iloc[i:i+config['sequence_length']].values for i in range(len(vol_data) - config['sequence_length'] + 1)])
                sequences = np.nan_to_num(sequences)
                vol_embeddings = experts['volatility'].encode(torch.FloatTensor(sequences).to(device)).cpu().numpy().mean(axis=1)
                all_embeddings['volatility'] = pd.DataFrame(vol_embeddings, index=vol_data.index[config['sequence_length']-1:])

            # Generate MLP Embeddings
            for name in ['trend', 'oscillator', 'deviation', 'cycle', 'pattern']:
                feature_cols = MLP_FEATURE_SETS[name]
                expert_data = features_df[feature_cols].dropna()
                if not expert_data.empty:
                    scaled_data = experts[f'{name}_scaler'].transform(expert_data)
                    embeddings = experts[name].encode(torch.FloatTensor(scaled_data).to(device)).cpu().numpy()
                    all_embeddings[name] = pd.DataFrame(embeddings, index=expert_data.index)
        
        if not all_embeddings: return [], []
        
        master_df = features_df[['target']]
        for name, embed_df in all_embeddings.items():
            embed_df.columns = [f'{name}_embed_{i}' for i in range(embed_df.shape[1])]
            master_df = master_df.join(embed_df)
        
        master_df.dropna(inplace=True)
        feature_data = master_df.drop('target', axis=1).values
        targets = master_df['target'].values
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # --- Final Sequence Creation ---
    if len(feature_data) < config['sequence_length']: return [], []
    X, y = [], []
    for i in range(len(feature_data) - config['sequence_length'] + 1):
        X.append(feature_data[i : i + config['sequence_length']])
        y.append(targets[i + config['sequence_length'] - 1])
    return np.array(X), np.array(y)

# ==============================================================================
#                              MAIN TRAINING SCRIPT
# ==============================================================================
def main(model_type: str, epochs: int, mode: str):
    print(f"--- Starting {mode.upper()} for: {model_type.upper()} EXPERT ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Define Model Architecture ---
    if model_type == 'volatility':
        params = CONFIG['volatility_expert_params']
    elif model_type == 'directional':
        params = CONFIG['directional_expert_params']
    else:
        raise ValueError("Invalid model_type specified.")

    model = TransformerExpert(
        input_dim=params['input_dim'], model_dim=CONFIG['embedding_dim'],
        num_heads=CONFIG['num_heads'], num_layers=params['num_layers'],
        output_dim=params['output_dim'], dropout=CONFIG['dropout']
    ).to(device)

    # --- Step 2: Load Prerequisite Models (Needed for both Train and Eval of Directional Expert) ---
    experts_for_fusion = {}
    if model_type == 'directional':
        try:
            print("Loading prerequisite expert models...")
            # Load all 6 experts and 5 scalers
            vol_params = CONFIG['volatility_expert_params']
            vol_expert = TransformerExpert(input_dim=vol_params['input_dim'], model_dim=CONFIG['embedding_dim'], num_heads=CONFIG['num_heads'], num_layers=vol_params['num_layers'], output_dim=1).to(device)
            vol_expert.load_state_dict(torch.load(os.path.join(CONFIG['model_save_path'], 'volatility_expert.pth'), weights_only=True))
            experts_for_fusion['volatility'] = vol_expert

            for name in ['trend', 'oscillator', 'deviation', 'cycle', 'pattern']:
                mlp_params = MLP_FEATURE_SETS[name]
                mlp_model = MLPExpert(input_dim=len(mlp_params), output_dim=3).to(device)
                mlp_model.load_state_dict(torch.load(os.path.join(CONFIG['model_save_path'], f'{name}_mlp_expert.pth'), weights_only=True))
                scaler = joblib.load(os.path.join(CONFIG['model_save_path'], f'{name}_mlp_scaler.joblib'))
                experts_for_fusion[name] = mlp_model
                experts_for_fusion[f'{name}_scaler'] = scaler
            print("All prerequisite expert models loaded successfully.")
        except Exception as e:
            print(f"FATAL ERROR: Could not load a prerequisite model. Details: {e}")
            return

    # --- Step 3: Execute the correct mode (Train or Eval) ---
    if mode == 'train':
        run_training(model, model_type, epochs, device, experts_for_fusion)
    elif mode == 'eval':
        # Load the final, trained model for evaluation
        model_path = os.path.join(CONFIG['model_save_path'], f'{model_type}_expert.pth')
        if not os.path.exists(model_path):
            print(f"FATAL ERROR: Model not found at {model_path}. Please train it first.")
            return
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        run_evaluation(model, model_type, device, experts_for_fusion)

def run_training(model, model_type, epochs, device, experts):
    # This function contains the original training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss() if model_type == 'volatility' else nn.CrossEntropyLoss()

    print(f"\nStarting training for {epochs} epochs...")
    all_files = glob.glob(os.path.join(CONFIG['data_path'], "*.csv"))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        random.shuffle(all_files)

        file_iterator = tqdm(all_files, desc=f"Epoch {epoch+1}/{epochs}")
        for f in file_iterator:
            try:
                df = pd.read_csv(f, index_col='Date', parse_dates=True)
                df.rename(columns={df.columns[0]: 'Close'}, inplace=True)
                df = df[df['Close'] > 0]
                if len(df) <= 200: continue
                
                # The create_sequences function now handles all feature generation internally
                X, y = create_sequences(df.copy(), CONFIG, device, model_type, experts=experts_for_fusion)
                if len(X) == 0: continue

                y_tensor = torch.LongTensor(y) if model_type == 'directional' else torch.FloatTensor(y).view(-1, 1)
                X_tensor = torch.FloatTensor(X)
                dataset = TensorDataset(X_tensor, y_tensor)
                data_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
                
                for inputs, targets in data_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1
            except Exception as e:
                print(f"Warning: Failed to process file {f}. Error: {e}")
        
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs} complete. Average Loss: {avg_epoch_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} complete. No valid data processed.")
            
        epoch_save_path = os.path.join(CONFIG['model_save_path'], f'{model_type}_expert_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), epoch_save_path)
        print(f"Saved epoch checkpoint to '{epoch_save_path}'")

    final_save_path = os.path.join(CONFIG['model_save_path'], f'{model_type}_expert.pth')
    if os.path.exists(epoch_save_path):
        shutil.copy(epoch_save_path, final_save_path)
        print(f"\nTraining complete. Final model saved to '{final_save_path}'")

def run_evaluation(model, model_type, device, experts):
    """
    Runs the evaluation on the holdout dataset.
    MODIFIED: Now processes data in a streaming, file-by-file fashion to
    prevent system RAM (OOM) errors.
    """
    print("\n--- Preparing Holdout Data for Evaluation ---")
    holdout_path = CONFIG['data_path'].replace('market_data_cache', 'market_data_holdout')
    all_files = glob.glob(os.path.join(holdout_path, "*.csv"))
    
    # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
    # We will store only the final, small prediction and target arrays.
    all_final_preds = []
    all_y_holdout = []
    
    model.eval()
    
    file_iterator = tqdm(all_files, desc="Processing Holdout Files")
    for f in file_iterator:
        try:
            df = pd.read_csv(f, index_col='Date', parse_dates=True)
            df.rename(columns={df.columns[0]: 'Close'}, inplace=True)
            df = df[df['Close'] > 0]
            if len(df) <= 200: continue

            # 1. Generate features and sequences for ONE file.
            X, y = create_sequences(df.copy(), CONFIG, device, model_type, experts=experts)
            if len(X) == 0: continue

            # 2. Get predictions for this file's data (in batches to save VRAM).
            eval_batch_size = 256
            X_tensor = torch.FloatTensor(X)
            eval_dataset = TensorDataset(X_tensor)
            eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False)
            
            file_preds = []
            with torch.no_grad():
                for (inputs,) in eval_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    if model_type == 'volatility':
                        preds = outputs.cpu().numpy().flatten()
                    else:
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    file_preds.extend(preds)
            
            # 3. Append the small result arrays to our master lists.
            all_final_preds.extend(file_preds)
            all_y_holdout.extend(y)
            
            # 4. The large X and y arrays are now discarded as the loop continues.
            
        except Exception as e:
            print(f"Warning: Failed to process holdout file {f}. Error: {e}")

    if not all_y_holdout:
        print("FATAL: No valid evaluation data could be generated from the holdout set.")
        return

    final_preds = np.array(all_final_preds)
    y_holdout = np.array(all_y_holdout)

    # --- Print Performance Report ---
    print("\n" + "="*80)
    print(f"        PERFORMANCE REPORT FOR: {model_type.upper()} EXPERT (on Holdout Data)")
    print("="*80)
    if model_type == 'volatility':
        r2 = r2_score(y_holdout, final_preds)
        print(f"  - R-squared (R²): {r2:.4f} (Explains {r2*100:.2f}% of future variance)")
    else:
        print(classification_report(y_holdout, final_preds, target_names=['DOWN', 'NEUTRAL', 'UP'], zero_division=0))
        cm = confusion_matrix(y_holdout, final_preds, labels=[0, 1, 2])
        print("--- Confusion Matrix ---")
        print("         Predicted ->")
        print("         DOWN  NEUTRAL   UP")
        print(f"Actual DOWN    {cm[0, 0]:>4}   {cm[0, 1]:>5}   {cm[0, 2]:>4}")
        print(f"Actual NEUTRAL {cm[1, 0]:>4}   {cm[1, 1]:>5}   {cm[1, 2]:>4}")
        print(f"Actual UP      {cm[2, 0]:>4}   {cm[2, 1]:>5}   {cm[2, 2]:>4}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Evaluate Transformer Expert Models.")
    parser.add_argument('--model_type', type=str, required=True, choices=['volatility', 'directional'], help="The type of expert model to process.")
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'], help="The number of epochs to train for (ignored in eval mode).")
    # <<< --- NEW: The mode argument --- >>>
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'eval'],
        help="Set the script to training or evaluation mode."
    )
    args = parser.parse_args()
    main(args.model_type, args.epochs, args.mode)

