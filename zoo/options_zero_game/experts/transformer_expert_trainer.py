# zoo/options_zero_game/experts/transformer_expert_trainer.py
# <<< DEFINITIVE SOTA VERSION with "Council of Experts" Architecture >>>

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import glob
import os
import math
import random
import joblib # <-- Import joblib to load LightGBM models
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
# Import the feature engineering function from the other trainer
from zoo.options_zero_game.experts.holy_trinity_trainer import create_holy_trinity_features_and_targets, CONFIG as HolyTrinityConfig

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
        "input_dim": 2, # [log_return, volatility]
        "output_dim": 1,
    },
    # <<< --- UPDATED: The new directional expert configuration --- >>>
    "directional_expert_params": {
        "num_layers": 4, # A slightly deeper model for the complex features
        "input_dim": 128 + 1 + 3, # 128 (vol_embed) + 1 (ema_pred) + 3 (rsi_probs) = 132
        "output_dim": 3, # DOWN, NEUTRAL, UP
    },

    "sequence_length": 60,
    "prediction_horizon": 5,
    "batch_size": 64,
    "epochs": 15,
    "lr": 1e-4,

    "volatility_period": 14,
    "directional_threshold": 0.005,
}

# ==============================================================================
#                          MODEL ARCHITECTURES
# ==============================================================================

class TransformerExpert(nn.Module):
    # This class is unchanged, but will now be used for BOTH models.
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.input_embedding = nn.Linear(input_dim, model_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_head = nn.Linear(model_dim, output_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(src)
        # For classification, we use the last token's output for prediction
        return self.output_head(embedding[:, -1, :])

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        x = self.input_embedding(src) * math.sqrt(self.model_dim)
        return self.transformer_encoder(x)

# The HybridGRUTransformerExpert is no longer needed for this new architecture.

# ==============================================================================
#                         DATA PREPARATION PIPELINE
# ==============================================================================
def create_sequences(df: pd.DataFrame, config: dict, device: torch.device, model_type: str, vol_expert_model=None, ema_expert=None, rsi_expert=None):
    """
    Creates sequences and targets. This function now has two distinct modes.
    """
    # --- Universal Target Engineering ---
    future_return = np.log(df['Close'].shift(-config['prediction_horizon']) / df['Close'])
    conditions = [future_return > config['directional_threshold'], future_return < -config['directional_threshold']]
    choices = [2, 0] # 2=UP, 0=DOWN
    df['direction_target'] = np.select(conditions, choices, default=1) # 1=NEUTRAL

    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=config['volatility_period']).std()
    df['volatility_target'] = df['volatility'].shift(-config['prediction_horizon'])

    # --- MODE 1: Training the Volatility Expert (uses raw price data) ---
    if model_type == 'volatility':
        df.dropna(subset=['log_return', 'volatility', 'volatility_target'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        if len(df) < config['sequence_length']: return [], [], []

        feature_data = df[['log_return', 'volatility']].values
        targets = df['volatility_target'].values
        X, y = [], []
        for i in range(len(df) - config['sequence_length']):
            X.append(feature_data[i : i + config['sequence_length']])
            y.append(targets[i + config['sequence_length'] - 1])
        return np.array(X), np.array(y)

    # --- MODE 2: Training the Directional Expert (uses other experts' outputs) ---
    elif model_type == 'directional':
        if not all([vol_expert_model, ema_expert, rsi_expert]):
            raise ValueError("All three expert models must be provided for directional training.")

        # a) Create the features needed for the Holy Trinity models
        ht_df = create_holy_trinity_features_and_targets(df.copy())

        # b) Combine all necessary data and drop NaNs
        df = df.join(ht_df[[col for col in ht_df.columns if 'lag' in col]])
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        if len(df) < config['sequence_length']: return [], [], []

        # c) Pre-calculate all expert predictions for the entire DataFrame
        vol_expert_model.eval()
        with torch.no_grad():
            # Create sequences of raw features for the vol expert
            raw_vol_features = df[['log_return', 'volatility']].values
            vol_sequences = np.array([raw_vol_features[i:i+config['sequence_length']] for i in range(len(df) - config['sequence_length'])])

            # Get embeddings for all sequences in one batch
            vol_embeddings = vol_expert_model.encode(torch.FloatTensor(vol_sequences).to(device)).cpu().numpy()

        ht_feature_names = [f'log_return_lag_{i}' for i in range(1, 31)] + [f'rsi_lag_{i}' for i in range(1, 31)]
        ht_features = df[ht_feature_names].values

        ema_preds = ema_expert.predict(ht_features).reshape(-1, 1)
        rsi_probs = rsi_expert.predict_proba(ht_features)

        # d) Assemble the final sequences from the pre-calculated expert outputs
        X, y_dir = [], []
        num_samples = len(df) - config['sequence_length']

        for i in range(num_samples):
            # The input for each timestep is the concatenated expert opinions at that time
            ema_sequence = ema_preds[i : i + config['sequence_length']]
            rsi_sequence = rsi_probs[i : i + config['sequence_length']]

            # The volatility embedding is for the whole sequence, so we use the pre-calculated one
            vol_embedding_for_sequence = vol_embeddings[i]

            # We need to tile the embedding to match the sequence length
            vol_embedding_tiled = np.tile(vol_embedding_for_sequence.mean(axis=0), (config['sequence_length'], 1))

            # Concatenate to form the final feature vector for the sequence
            combined_sequence = np.concatenate([vol_embedding_tiled, ema_sequence, rsi_sequence], axis=1)
            X.append(combined_sequence)
            y_dir.append(df['direction_target'].iloc[i + config['sequence_length'] - 1])

        return np.array(X), np.array(y_dir)

# ==============================================================================
#                              MAIN TRAINING SCRIPT
# ==============================================================================
def main(model_type: str):
    print(f"--- Starting Training for: {model_type.upper()} EXPERT ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load prerequisite models ---
    vol_expert_for_fusion = None
    ema_expert = None
    rsi_expert = None

    if model_type == 'directional':
        try:
            print("Loading prerequisite expert models for directional training...")
            # Load Volatility Transformer
            vol_params = CONFIG['volatility_expert_params']
            vol_expert_for_fusion = TransformerExpert(
                input_dim=vol_params['input_dim'], model_dim=CONFIG['embedding_dim'],
                num_heads=CONFIG['num_heads'], num_layers=vol_params['num_layers'],
                output_dim=vol_params['output_dim']
            ).to(device)
            vol_expert_for_fusion.load_state_dict(torch.load(os.path.join(CONFIG['model_save_path'], 'volatility_expert.pth'), weights_only=True))

            # Load Holy Trinity Models
            ema_expert = joblib.load(os.path.join(CONFIG['model_save_path'], 'ema_expert.joblib'))
            rsi_expert = joblib.load(os.path.join(CONFIG['model_save_path'], 'rsi_expert.joblib'))
            print("All prerequisite models loaded successfully.")
        except FileNotFoundError as e:
            print(f"FATAL ERROR: Could not load a prerequisite model. Please ensure volatility, ema, and rsi experts are trained first. Details: {e}")
            return

    # --- Define Model and Optimizer ---
    if model_type == 'volatility':
        params = CONFIG['volatility_expert_params']
        model = TransformerExpert(
            input_dim=params['input_dim'], model_dim=CONFIG['embedding_dim'],
            num_heads=CONFIG['num_heads'], num_layers=params['num_layers'],
            output_dim=params['output_dim'], dropout=CONFIG['dropout']
        ).to(device)
        criterion = nn.MSELoss()
    elif model_type == 'directional':
        params = CONFIG['directional_expert_params']
        # The directional expert now also uses the standard Transformer architecture
        model = TransformerExpert(
            input_dim=params['input_dim'], model_dim=CONFIG['embedding_dim'],
            num_heads=CONFIG['num_heads'], num_layers=params['num_layers'],
            output_dim=params['output_dim'], dropout=CONFIG['dropout']
        ).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid model_type specified.")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    model.train()

    # --- Training Loop ---
    print(f"\nStarting training for {CONFIG['epochs']} epochs...")
    all_files = glob.glob(os.path.join(CONFIG['data_path'], "*.csv"))

    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0.0
        num_batches = 0
        random.shuffle(all_files)

        file_iterator = tqdm(all_files, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for f in file_iterator:
            df = pd.read_csv(f, index_col='Date', parse_dates=True)
            df.rename(columns={df.columns[0]: 'Close'}, inplace=True)
            df = df[df['Close'] > 0]
            if len(df) <= CONFIG['sequence_length'] + HolyTrinityConfig['lookback_window'] + 5:
                continue

            # Process ONE file at a time
            X, y = create_sequences(df.copy(), CONFIG, device, model_type, vol_expert_for_fusion, ema_expert, rsi_expert)
            if len(X) == 0: continue

            # Select the correct target type
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

        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} complete. Average Loss: {avg_epoch_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} complete. No valid data processed.")

    save_path = os.path.join(CONFIG['model_save_path'], f'{model_type}_expert.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Final model saved to '{save_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer Expert Models for Options-Zero-Game.")
    parser.add_argument(
        '--model_type', 
        type=str, 
        required=True, 
        choices=['volatility', 'directional'],
        help="The type of expert model to train."
    )
    args = parser.parse_args()
    main(args.model_type)
