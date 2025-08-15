# zoo/options_zero_game/experts/transformer_expert_trainer.py
# <<< DEFINITIVE SOTA VERSION >>>

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import glob
import os
import math
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

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
        "input_dim": 2,
        "output_dim": 1,
    },
    "directional_expert_params": {
        "num_layers": 2, # Note: Fewer Transformer layers are often better in a hybrid
        "gru_layers": 2,
        "input_dim": 2 + 128, # base_features + vol_embedding
        "output_dim": 3,
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
    """Standard Transformer Encoder for the Volatility Analyst."""
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.input_embedding = nn.Linear(input_dim, model_dim)

        # <<< THE FIX: The self.pos_encoder line has been completely removed >>>

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_head = nn.Linear(model_dim, output_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(src)
        return self.output_head(embedding)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        # The logic is simpler and cleaner without the unused encoder.
        x = self.input_embedding(src) * math.sqrt(self.model_dim)
        x = self.transformer_encoder(x)
        return x.mean(dim=1)

class HybridGRUTransformerExpert(nn.Module):
    """SOTA GRU-Transformer Hybrid for the Directional Strategist."""
    def __init__(self, input_dim, model_dim, num_heads, num_layers, gru_layers, output_dim, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=model_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_head = nn.Linear(model_dim, output_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(src)
        transformer_out = self.transformer_encoder(gru_out)
        final_embedding = transformer_out[:, -1, :] # Use last hidden state
        return self.output_head(final_embedding)

# ==============================================================================
#                         DATA PREPARATION PIPELINE
# ==============================================================================
def create_sequences(df: pd.DataFrame, config: dict, device: torch.device, vol_expert_model: TransformerExpert = None):
    """
    Creates sequences and targets for training. This is a robust, multi-pass version.
    """
    # --- Pass 1: Calculate all base features and targets on the full DataFrame ---
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=config['volatility_period']).std()

    df['volatility_target'] = df['volatility'].shift(-config['prediction_horizon'])

    future_return = np.log(df['Close'].shift(-config['prediction_horizon']) / df['Close'])
    conditions = [future_return > config['directional_threshold'], future_return < -config['directional_threshold']]
    choices = [2, 0] # 2=UP, 0=DOWN
    df['direction_target'] = np.select(conditions, choices, default=1) # 1=NEUTRAL

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < config['sequence_length']:
        return [], [], []

    # --- Pass 2: Create the raw numpy arrays for sequences and targets ---
    base_features = ['log_return', 'volatility']
    feature_data = df[base_features].values
    vol_targets = df['volatility_target'].values
    dir_targets = df['direction_target'].values

    X_raw, y_vol, y_dir = [], [], []
    for i in range(len(df) - config['sequence_length']):
        X_raw.append(feature_data[i : i + config['sequence_length']])
        y_vol.append(vol_targets[i + config['sequence_length'] - 1])
        y_dir.append(dir_targets[i + config['sequence_length'] - 1])

    X_raw = np.array(X_raw)

    # --- Pass 3: If in directional mode, perform hierarchical fusion ---
    if vol_expert_model:
        # print("Performing hierarchical feature fusion...") # You can uncomment for debugging
        vol_expert_model.eval()
        
        with torch.no_grad():
            # <<< THE FIX: Create the tensor and immediately move it to the correct device >>>
            X_raw_tensor = torch.FloatTensor(X_raw).to(device)
            
            vol_embeddings = vol_expert_model.encode(X_raw_tensor).cpu().numpy() # Move back to CPU for numpy
        
        vol_embeddings_tiled = np.expand_dims(vol_embeddings, axis=1)
        vol_embeddings_tiled = np.tile(vol_embeddings_tiled, (1, config['sequence_length'], 1))

        X_fused = np.concatenate([X_raw, vol_embeddings_tiled], axis=2)
        return X_fused, np.array(y_vol), np.array(y_dir)
    else:
        return X_raw, np.array(y_vol), np.array(y_dir)

# ==============================================================================
#                              MAIN TRAINING SCRIPT
# ==============================================================================
def main(model_type: str):
    """
    Main function to orchestrate the training process. This version uses the
    appropriate SOTA model for each expert.
    """
    # --- 1. Setup ---
    print(f"--- Starting Training for: {model_type.upper()} EXPERT ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Define Model and Optimizer ---
    if model_type == 'volatility':
        params = CONFIG['volatility_expert_params']
        # Use the standard Transformer for the proven volatility task
        model = TransformerExpert(
            input_dim=params['input_dim'], model_dim=CONFIG['embedding_dim'],
            num_heads=CONFIG['num_heads'], num_layers=params['num_layers'],
            output_dim=params['output_dim'], dropout=CONFIG['dropout']
        ).to(device)
        criterion = nn.MSELoss()
    elif model_type == 'directional':
        params = CONFIG['directional_expert_params']
        # Use the new, more powerful SOTA Hybrid model for the difficult directional task
        model = HybridGRUTransformerExpert(
            input_dim=params['input_dim'], model_dim=CONFIG['embedding_dim'],
            num_heads=CONFIG['num_heads'], num_layers=params['num_layers'],
            gru_layers=params['gru_layers'], output_dim=params['output_dim'],
            dropout=CONFIG['dropout']
        ).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid model_type specified.")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    model.train()

    # --- 3. Training Loop with Streaming Data Processing ---
    print(f"\nStarting training for {CONFIG['epochs']} epochs...")
    all_files = glob.glob(os.path.join(CONFIG['data_path'], "*.csv"))
    
    vol_expert_for_fusion = None # Will be loaded if needed

    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0.0
        num_batches = 0
        random.shuffle(all_files)
        
        # Load the vol expert once per epoch if in directional mode
        if model_type == 'directional' and vol_expert_for_fusion is None:
            vol_params = CONFIG['volatility_expert_params']
            try:
                # We need to load the *standard* TransformerExpert here for fusion
                vol_expert_for_fusion = TransformerExpert(
                    input_dim=vol_params['input_dim'], model_dim=CONFIG['embedding_dim'],
                    num_heads=CONFIG['num_heads'], num_layers=vol_params['num_layers'],
                    output_dim=vol_params['output_dim']
                ).to(device)
                vol_expert_for_fusion.load_state_dict(torch.load(os.path.join(CONFIG['model_save_path'], 'volatility_expert.pth')))
                print("Successfully loaded pre-trained Volatility Expert for feature generation.")
            except FileNotFoundError:
                print("ERROR: Could not find 'volatility_expert.pth'. Please train the volatility model first.")
                return

        file_iterator = tqdm(all_files, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for f in file_iterator:
            df = pd.read_csv(f)
            # ... (Robust data cleaning) ...
            close_col_name = df.columns[1] if 'Date' in df.columns else df.columns[0]
            df.rename(columns={close_col_name: 'Close'}, inplace=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df.dropna(subset=['Close'], inplace=True)
            df = df[df['Close'] > 0]
            df.reset_index(drop=True, inplace=True)
            
            if len(df) <= CONFIG['sequence_length'] + CONFIG['prediction_horizon']:
                continue

            # Process ONE file at a time
            X, y_vol, y_dir = create_sequences(df.copy(), CONFIG, device, vol_expert_for_fusion)
            
            if len(X) == 0:
                continue

            # Select the correct target for this training run
            y = y_dir if model_type == 'directional' else y_vol
            y_tensor = torch.LongTensor(y) if model_type == 'directional' else torch.FloatTensor(y).view(-1, 1)

            # Create a DataLoader for just this file's data
            X_tensor = torch.FloatTensor(X)
            dataset = TensorDataset(X_tensor, y_tensor)
            data_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
            
            # Train on the batches from this single file
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
        
        # --- 4. Log Epoch Results ---
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} complete. Average Loss: {avg_epoch_loss:.6f}")
        else:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} complete. No valid data processed.")

    # --- 5. Save Final Model ---
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
