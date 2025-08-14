# zoo/options_zero_game/experts/transformer_expert_trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import pandas_ta as ta
import os
import glob
import argparse
import math
from tqdm import tqdm

# ==============================================================================
#                            CONFIGURATION
# ==============================================================================
CONFIG = {
    "data_path": "zoo/options_zero_game/data/market_data_cache",
    "model_save_path": "zoo/options_zero_game/experts/",
    
    # --- Shared Model Hyperparameters ---
    "embedding_dim": 128,   # The core size of the model's internal representation
    "num_heads": 4,         # Number of attention heads
    "dropout": 0.1,
    
    # <<< NEW: Expert-Specific Architectural Parameters >>>
    "volatility_expert_params": {
        "num_layers": 3,       # As you requested
        "input_dim": 2,        # log_return, volatility
        "output_dim": 1,       # A single predicted volatility value (regression)
    },
    "directional_expert_params": {
        "num_layers": 4,       # As you requested
        "input_dim": 2 + 128,  # base_features + volatility_embedding_dim
        "output_dim": 3,       # UP, NEUTRAL, DOWN probabilities (classification)
    },
    
    # --- Shared Training Hyperparameters ---
    "sequence_length": 60,
    "prediction_horizon": 5,
    "batch_size": 64,
    "epochs": 15,
    "lr": 1e-4,
    
    # --- Shared Target Engineering ---
    "volatility_period": 14,
    "directional_threshold": 0.005,
}

# ==============================================================================
#                          TRANSFORMER MODEL DEFINITION
# ==============================================================================

class PositionalEncoding(nn.Module):
    """ Standard sinusoidal positional encoding """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerExpert(nn.Module):
    """
    A flexible Transformer model for both regression and classification tasks.
    It includes an `encode` method to extract embeddings for inference.
    """
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_head = nn.Linear(model_dim, output_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # Generate the rich embedding first
        embedding = self.encode(src)
        # Pass the embedding through the final output layer
        output = self.output_head(embedding)
        return output

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        Runs the model up to the point of producing the embedding.
        This is used for feature generation.
        """
        # src shape: (batch_size, seq_len, input_dim)
        src = self.input_embedding(src) * math.sqrt(self.model_dim)
        # Note: Positional encoding expects (seq_len, batch_size, model_dim) if batch_first=False
        # Since we use batch_first=True, we need to adapt. Let's assume a simpler addition.
        # A more robust implementation might require permuting dimensions.
        
        # Simple positional encoding for batch_first=True
        # x = src.permute(1, 0, 2) # (seq_len, batch_size, features)
        # x = self.pos_encoder(x)
        # x = x.permute(1, 0, 2) # (batch_size, seq_len, features)
        x = self.transformer_encoder(src)
        
        # Pool the sequence of embeddings into a single embedding
        # Taking the mean of the sequence is a standard and effective method
        pooled_embedding = x.mean(dim=1)
        return pooled_embedding

# ==============================================================================
#                         DATA PREPARATION PIPELINE
# ==============================================================================
def create_sequences(df: pd.DataFrame, config: dict, vol_expert_model: TransformerExpert = None):
    """
    Creates sequences and targets for training.
    If `vol_expert_model` is provided, it performs hierarchical feature fusion.
    """
    sequences = []
    
    # 1. Calculate base features
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_return'].rolling(window=config['volatility_period']).std()
    df.dropna(inplace=True)
    
    base_features = ['log_return', 'volatility']
    feature_data = df[base_features].values
    
    # 2. HIERARCHICAL FEATURE FUSION (for Directional model)
    if vol_expert_model:
        print("Performing hierarchical feature fusion with Volatility Expert...")
        vol_expert_model.eval()
        
        # Temporarily create sequences of just the base features to get embeddings
        temp_sequences = []
        for i in range(len(feature_data) - config['sequence_length']):
            temp_sequences.append(feature_data[i:i + config['sequence_length']])
            
        with torch.no_grad():
            temp_tensor = torch.FloatTensor(np.array(temp_sequences))
            vol_embeddings = vol_expert_model.encode(temp_tensor).numpy()

        # Now, create the final fused features
        fused_feature_data = []
        # The first `sequence_length` points don't have embeddings, so we align.
        for i in range(len(vol_embeddings)):
            # For each step in the original sequence, append the *single* embedding for that *entire* sequence
            fused_sequence = np.hstack([
                feature_data[i: i + config['sequence_length']],
                np.tile(vol_embeddings[i], (config['sequence_length'], 1))
            ])
            fused_feature_data.append(fused_sequence)
        
        # Update feature_data to the new fused data
        feature_data = np.array(fused_feature_data)
        # We now have one less sequence because of the fusion alignment
        df = df.iloc[config['sequence_length']-1:].reset_index(drop=True)
    
    # 3. Create targets
    # a) Volatility Target (for Volatility Analyst)
    df['volatility_target'] = df['volatility'].shift(-config['prediction_horizon'])
    
    # b) Directional Target (for Directional Strategist)
    future_return = np.log(df['Close'].shift(-config['prediction_horizon']) / df['Close'])
    conditions = [
        future_return > config['directional_threshold'],
        future_return < -config['directional_threshold']
    ]
    choices = [2, 0] # 2=UP, 0=DOWN
    df['direction_target'] = np.select(conditions, choices, default=1) # 1=NEUTRAL
    
    df.dropna(inplace=True)
    
    # 4. Final sequence creation
    for i in range(len(df) - config['sequence_length'] - config['prediction_horizon']):
        seq = feature_data[i : i + config['sequence_length']]
        
        vol_target = df['volatility_target'].iloc[i + config['sequence_length'] -1]
        dir_target = df['direction_target'].iloc[i + config['sequence_length'] -1]
        
        sequences.append((seq, vol_target, dir_target))
        
    return sequences


# ==============================================================================
#                              MAIN TRAINING SCRIPT
# ==============================================================================
def main(model_type: str):
    """Main function to orchestrate the training process."""
    
    # --- 1. Setup ---
    print(f"--- Starting Training for: {model_type.upper()} EXPERT ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 2. Data Loading and Processing ---
    all_files = glob.glob(os.path.join(CONFIG['data_path'], "*.csv"))
    all_sequences = []
    
    vol_expert_for_fusion = None
    if model_type == 'directional':
        # <<< MODIFICATION: Use the specific params to load the vol expert >>>
        vol_params = CONFIG['volatility_expert_params']
        try:
            vol_expert_for_fusion = TransformerExpert(
                input_dim=vol_params['input_dim'],
                model_dim=CONFIG['embedding_dim'],
                num_heads=CONFIG['num_heads'],
                num_layers=vol_params['num_layers'], # Use the vol expert's layer count
                output_dim=vol_params['output_dim']
            )
            vol_expert_for_fusion.load_state_dict(torch.load(os.path.join(CONFIG['model_save_path'], 'volatility_expert.pth')))
            print("Successfully loaded pre-trained Volatility Expert for feature generation.")
        except FileNotFoundError:
            print("ERROR: Could not find 'volatility_expert.pth'. Please train the volatility model first.")
            return

    # ... (the data processing loop is unchanged) ...

    # --- 3. Model and Training Specifics ---
    if model_type == 'volatility':
        X = np.array([s[0] for s in all_sequences])
        y = np.array([s[1] for s in all_sequences])
        
        # <<< MODIFICATION: Get params from the specific config block >>>
        params = CONFIG['volatility_expert_params']
        model = TransformerExpert(
            input_dim=params['input_dim'],
            model_dim=CONFIG['embedding_dim'],
            num_heads=CONFIG['num_heads'],
            num_layers=params['num_layers'], # Use 3 layers
            output_dim=params['output_dim'],
            dropout=CONFIG['dropout']
        ).to(device)
        criterion = nn.MSELoss()
        y_tensor = torch.FloatTensor(y).view(-1, 1)
        
    elif model_type == 'directional':
        X = np.array([s[0] for s in all_sequences])
        y = np.array([s[2] for s in all_sequences])
        
        # <<< MODIFICATION: Get params from the specific config block >>>
        params = CONFIG['directional_expert_params']
        model = TransformerExpert(
            input_dim=params['input_dim'],
            model_dim=CONFIG['embedding_dim'],
            num_heads=CONFIG['num_heads'],
            num_layers=params['num_layers'], # Use 4 layers
            output_dim=params['output_dim'],
            dropout=CONFIG['dropout']
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        y_tensor = torch.LongTensor(y)
        
    else:
        raise ValueError("Invalid model_type specified.")

    # --- 4. Create DataLoader & Training Loop (Unchanged) ---
    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # --- 5. Training Loop ---
    print(f"\nStarting training on {len(dataset)} sequences...")
    model.train()
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        for inputs, targets in tqdm(data_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # For CrossEntropyLoss, target should not be one-hot and shape (batch_size)
            if isinstance(criterion, nn.CrossEntropyLoss):
                targets = targets.squeeze()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Average Loss: {avg_loss:.6f}")

    # --- 6. Save Model ---
    save_path = os.path.join(CONFIG['model_save_path'], f'{model_type}_expert.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved to '{save_path}'")

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
