# zoo/options_zero_game/experts/unified_expert_trainer.py
#
# A unified script to train a "council" of specialized MLP experts, each focused
# on a specific category of advanced features.

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import glob
import os
import random
import joblib
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import our new feature engineering pipeline
from zoo.options_zero_game.experts.advanced_feature_engineering import calculate_advanced_features

# ==============================================================================
#                            CONFIGURATION
# ==============================================================================
CONFIG = {
    "data_path": "zoo/options_zero_game/data/market_data_cache",
    "model_save_path": "zoo/options_zero_game/experts/",
    "prediction_horizon": 5,
    "directional_threshold": 0.005,
    "batch_size": 256,
    "epochs": 20,
    "lr": 1e-4,
    "test_size": 0.2,
}

# --- Define the feature sets for each expert ---
EXPERT_FEATURE_SETS = {
    "volatility": ['vol_realized_14', 'vol_of_vol_14', 'vol_autocorr_14', 'vol_realized_30', 'vol_of_vol_30', 'vol_autocorr_30'],
    "trend": ['trend_ma_slope_10', 'trend_price_to_ma_10', 'trend_ma_slope_20', 'trend_price_to_ma_20', 'trend_ma_slope_50', 'trend_price_to_ma_50', 'trend_crossover'],
    "oscillator": ['STOCHk_14_3_3', 'STOCHd_14_3_3', 'WILLR_14', 'ROC_14'],
    "deviation": ['dev_z_score_20', 'dev_bollinger_b_pct', 'dev_channel_breakout'], # <-- CORRECTED
    "cycle": ['cycle_hilbert_phase', 'cycle_day_of_week', 'cycle_month_of_year'],
    "fractal": ['fractal_hurst_100'],
    "pattern": [f'pattern_log_return_lag_{i}' for i in range(1, 21)],
}

# ==============================================================================
#                          MODEL ARCHITECTURE
# ==============================================================================
class MLPExpert(nn.Module):
    """A simple but effective MLP for learning from structured feature sets."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer (the embedding)
        self.network = nn.Sequential(*layers)
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.network(x)
        return self.output_head(embedding)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the final embedding before the output head."""
        return self.network(x)

# ==============================================================================
#                              MAIN TRAINING SCRIPT
# ==============================================================================
def main(expert_name: str, epochs: int):
    print(f"--- Starting Unified Training for: {expert_name.upper()} EXPERT ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load and Prepare Data ---
    all_files = glob.glob(os.path.join(CONFIG['data_path'], "*.csv"))
    all_processed_dfs = []
    for f in tqdm(all_files, desc="Processing Data Files"):
        try:
            df = pd.read_csv(f, index_col='Date', parse_dates=True)
            close_col = next((col for col in df.columns if 'Close' in col or 'close' in col), df.columns[0])
            df.rename(columns={close_col: 'Close'}, inplace=True)
            df = df[df['Close'] > 0].dropna(subset=['Close'])
            if len(df) < 200: continue
            features_df = calculate_advanced_features(df)
            all_processed_dfs.append(features_df)
        except Exception as e:
            print(f"Warning: Skipping file {f} due to processing error: {e}")

    if not all_processed_dfs:
        raise ValueError("No data could be processed. Check data files and feature engineering logic.")
        
    all_data_df = pd.concat(all_processed_dfs)

    # <<< --- THE DEFINITIVE FIX IS HERE: Correct Order of Operations --- >>>

    # --- 2. Define Target Variable FIRST ---
    is_regression = expert_name == 'volatility'
    if is_regression:
        all_data_df['target'] = all_data_df['vol_realized_14'].shift(-CONFIG['prediction_horizon'])
        output_dim = 1
        criterion = nn.MSELoss()
    else: # Classification for all other experts
        future_return = np.log(all_data_df['Close'].shift(-CONFIG['prediction_horizon']) / all_data_df['Close'])
        conditions = [future_return > CONFIG['directional_threshold'], future_return < -CONFIG['directional_threshold']]
        choices = [2, 0] # UP, DOWN
        all_data_df['target'] = np.select(conditions, choices, default=1) # NEUTRAL
        output_dim = 3
        criterion = nn.CrossEntropyLoss()

    # --- 3. Finalize Dataset AFTER the target column exists ---
    feature_cols = EXPERT_FEATURE_SETS[expert_name]
    # Now this line will work correctly because 'target' has been created.
    final_df = all_data_df[feature_cols + ['target']].dropna()
    
    X = final_df[feature_cols].values
    y = final_df['target'].values   

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG['test_size'], random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- 4. Create DataLoaders ---
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train) if not is_regression else torch.FloatTensor(y_train).view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # --- 5. Initialize Model and Optimizer ---
    input_dim = len(feature_cols)
    model = MLPExpert(input_dim=input_dim, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # --- 6. Training Loop ---
    print(f"\nTraining on {len(X_train)} samples...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_loader):.6f}")

    # --- 7. Save Model and Scaler ---
    model_save_path = os.path.join(CONFIG['model_save_path'], f'{expert_name}_mlp_expert.pth')
    scaler_save_path = os.path.join(CONFIG['model_save_path'], f'{expert_name}_mlp_scaler.joblib')
    torch.save(model.state_dict(), model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"\nTraining complete. Final model saved to '{model_save_path}'")
    print(f"Feature scaler saved to '{scaler_save_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a unified council of MLP experts.")
    parser.add_argument(
        '--expert',
        type=str,
        required=True,
        choices=list(EXPERT_FEATURE_SETS.keys()),
        help="The name of the expert to train."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=CONFIG['epochs'],
        help="The number of epochs to train for."
    )
    args = parser.parse_args()
    main(args.expert, args.epochs)
