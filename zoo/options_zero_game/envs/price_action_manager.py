# zoo/options_zero_game/envs/price_action_manager.py
# <<< DEFINITIVELY CORRECTED VERSION with Price Generation Logic Restored >>>

import os
import random
import math
import joblib
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

# Import all necessary components from the expert trainers
from zoo.options_zero_game.experts.transformer_expert_trainer import TransformerExpert, CONFIG, CONFIG as TransformerConfig
from zoo.options_zero_game.experts.unified_expert_trainer import MLPExpert, EXPERT_FEATURE_SETS as MLP_FEATURE_SETS
from zoo.options_zero_game.experts.advanced_feature_engineering import calculate_advanced_features


class PriceActionManager:
    """
    Handles all logic for market data and runs the full "Council of Experts"
    inference pipeline at every step to provide rich features for the agent.
    """
    def __init__(self, cfg: Dict, np_random: any):
        self._cfg = cfg
        self.price_source = cfg['price_source']
        self.historical_data_path = cfg['historical_data_path']
        self.start_price_config = cfg['start_price']
        self.np_random = np_random
        self.expert_sequence_length = cfg.get('expert_sequence_length', 60)
        self._available_tickers = self._load_tickers()

        # --- State Variables ---
        self.price_path: np.ndarray = np.array([])
        self.historical_context_path: np.ndarray = np.array([])
        self.features_df: pd.DataFrame = pd.DataFrame()
        self.current_price: float = 0.0
        self.start_price: float = 0.0
        self.episode_iv_anchor: float = 0.2 # Initialize with a safe default
        
        # --- Expert Model State ---
        self.experts = self._load_all_experts()
        self.directional_prediction_probs = np.array([0.33, 0.34, 0.33])
        self.volatility_embedding = None
        self.trend_embedding = None
        self.oscillator_embedding = None
        self.deviation_embedding = None
        self.cycle_embedding = None
        self.pattern_embedding = None

    def _load_tickers(self) -> list:
        if self.price_source in ['historical', 'mixed']:
            if not os.path.isdir(self.historical_data_path):
                raise FileNotFoundError(f"Historical data path not found: {self.historical_data_path}")
            tickers = [f.replace('.csv', '') for f in os.listdir(self.historical_data_path) if f.endswith('.csv')]
            if not tickers: raise FileNotFoundError(f"No CSV files found in {self.historical_data_path}")
            return tickers
        return []

    def _load_all_experts(self) -> dict:
        """Loads the complete council of 7 NNs and 5 scalers."""
        experts = {}
        device = 'cpu'
        model_path = TransformerConfig['model_save_path']
        print("--- PriceActionManager: Loading Council of Experts ---")
        try:
            # 1. Load Volatility Transformer
            vol_params = TransformerConfig['volatility_expert_params']
            vol_expert = TransformerExpert(input_dim=vol_params['input_dim'], model_dim=TransformerConfig['embedding_dim'], num_heads=TransformerConfig['num_heads'], num_layers=vol_params['num_layers'], output_dim=1).to(device)
            vol_expert.load_state_dict(torch.load(os.path.join(model_path, 'volatility_expert.pth'), map_location=device, weights_only=True))
            experts['volatility'] = vol_expert

            # 2. Load the 5 MLP Experts and their scalers
            for name in ['trend', 'oscillator', 'deviation', 'cycle', 'pattern']:
                mlp_params = MLP_FEATURE_SETS[name]
                model = MLPExpert(input_dim=len(mlp_params), output_dim=3).to(device)
                model.load_state_dict(torch.load(os.path.join(model_path, f'{name}_mlp_expert.pth'), map_location=device, weights_only=True))
                scaler = joblib.load(os.path.join(model_path, f'{name}_mlp_scaler.joblib'))
                experts[name] = model
                experts[f'{name}_scaler'] = scaler

            # 3. Load the Master Directional Transformer
            dir_params = TransformerConfig['directional_expert_params']
            dir_expert = TransformerExpert(input_dim=dir_params['input_dim'], model_dim=TransformerConfig['embedding_dim'], num_heads=CONFIG['num_heads'], num_layers=dir_params['num_layers'], output_dim=dir_params['output_dim']).to(device)
            dir_expert.load_state_dict(torch.load(os.path.join(model_path, 'directional_expert.pth'), map_location=device, weights_only=True))
            experts['master_directional'] = dir_expert
            
            print("Successfully loaded all 7 neural networks and 5 scalers.")
            return experts
        except Exception as e:
            print(f"WARNING: Failed to load one or more expert models. The agent's observation will be degraded. Error: {e}")
            return {}

    def reset(self, total_steps: int, chosen_regime: dict):
        self.total_steps = total_steps
        self.start_price = self.start_price_config
        source_to_use = self._cfg.price_source

        try:
            if source_to_use == 'garch':
                self._generate_garch_price_path(chosen_regime)
            elif source_to_use == 'historical':
                self._generate_historical_price_path() # This call was failing
            else: # Mixed mode
                if self.np_random.choice([True, False]):
                    self._generate_historical_price_path()
                else:
                    self._generate_garch_price_path(chosen_regime)

            if not np.all(np.isfinite(self.price_path)) or np.any(self.price_path <= 0):
                raise ValueError("Generated path contains invalid values.")

        except Exception as e:
            print(f"--- PRICE GENERATION FAILED: {type(e).__name__}: {e}. USING FAILSAFE (FLAT) PATH. ---")
            self.price_path = np.full(self.total_steps + 1, self.start_price, dtype=np.float32)

        self.features_df = calculate_advanced_features(pd.DataFrame({'Close': self.price_path}, index=pd.to_datetime(pd.RangeIndex(start=0, stop=len(self.price_path), step=1), unit='D')))
        self.current_price = self.price_path[0]

    def _generate_historical_price_path(self):
        """Generates a price path from a random slice of historical data."""
        forced_symbol = self._cfg.get('forced_historical_symbol')
        forced_days_back = self._cfg.get('forced_days_back')

        for _ in range(5): # Failsafe loop
            try:
                selected_ticker = forced_symbol or self.np_random.choice(self._available_tickers)
                file_path = os.path.join(self.historical_data_path, f"{selected_ticker}.csv")
                data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                data.rename(columns={data.columns[0]: 'Close'}, inplace=True)

                required_length = self.total_steps + 1
                if len(data) < required_length: continue

                max_start_index = len(data) - required_length
                if forced_days_back is not None:
                    start_index = max(0, max_start_index - (forced_days_back - 1))
                else:
                    start_index = self.np_random.integers(0, max_start_index + 1)
                
                context_start_index = max(0, start_index - 30)
                context_segment = data['Close'].iloc[context_start_index:start_index]
                price_segment = data['Close'].iloc[start_index : start_index + required_length]
                
                raw_start_price = price_segment.iloc[0]
                if raw_start_price > 1e-4:
                    normalization_factor = self.start_price_config / raw_start_price
                    self.price_path = (price_segment * normalization_factor).to_numpy(dtype=np.float32)
                    self.historical_context_path = (context_segment * normalization_factor).to_numpy(dtype=np.float32)
                    self.start_price = self.price_path[0]
                    # Calculate the historical volatility of the generated price path
                    # to set a realistic baseline IV for the episode.
                    log_returns = np.log(self.price_path[1:] / self.price_path[:-1])
                    annualized_vol = np.std(log_returns) * np.sqrt(252) # Assuming daily steps
                    
                    # Set the attribute, with a failsafe for the case of zero volatility
                    self.episode_iv_anchor = annualized_vol if np.isfinite(annualized_vol) and annualized_vol > 0 else 0.2
                    return # Success
            except Exception as e:
                print(f"(WARNING) Historical path generation failed for {selected_ticker}: {e}. Retrying.")
                continue
        raise ValueError("Failed to generate a valid historical price path after 5 attempts.")

    def _generate_garch_price_path(self, chosen_regime: dict):
        """Generates a price path using an expert-driven GARCH model."""
        self.current_regime_name = f"GARCH: {chosen_regime['name']}"
        prices = np.zeros(self.total_steps + 1, dtype=np.float32)
        prices[0] = self.start_price_config

        bootstrap_end_step = min(self.expert_sequence_length + 1, self.total_steps + 1)
        initial_vol = chosen_regime['atm_iv'] / 100.0 / np.sqrt(252)
        for i in range(1, bootstrap_end_step):
            step_return = self.np_random.normal(0, initial_vol)
            prices[i] = prices[i-1] * np.exp(np.clip(step_return, -0.5, 0.5))

        # This part is simplified as the full expert-driven simulation is very complex
        # for a standard GARCH path. We use the regime's base vol.
        for t in range(bootstrap_end_step, self.total_steps + 1):
            step_vol = chosen_regime['atm_iv'] / 100.0 / np.sqrt(252)
            step_return = self.np_random.normal(0, step_vol)
            prices[t] = prices[t-1] * np.exp(np.clip(step_return, -0.5, 0.5))

        self.price_path = prices
        self.start_price = self.price_path[0]
        # Set the anchor based on the regime's base IV
        self.episode_iv_anchor = chosen_regime.get('atm_iv', 20.0) / 100.0

    def step(self, current_step: int):
        """Updates the manager's state and runs the full inference pipeline."""
        self.current_price = self.price_path[current_step]
        
        if not self.experts or current_step < self.expert_sequence_length:
            return

        with torch.no_grad():
            history_df = self.features_df.iloc[max(0, current_step + 1 - self.expert_sequence_length) : current_step + 1].copy()
            if len(history_df) < self.expert_sequence_length: return

            embeddings_for_master = []
            
            # 1. Volatility Transformer
            vol_cols = ['log_return', 'volatility', 'vol_of_vol', 'vol_autocorr']
            vol_data = history_df[vol_cols].fillna(0).values
            vol_sequence = torch.FloatTensor(vol_data).unsqueeze(0).to(next(self.experts['volatility'].parameters()).device)
            vol_embedding = self.experts['volatility'].encode(vol_sequence).cpu().numpy().mean(axis=1)
            embeddings_for_master.append(vol_embedding)
            self.volatility_embedding = vol_embedding.flatten()

            # 2. MLP Experts (using only the most recent data point)
            latest_features = history_df.iloc[-1]
            for name in ['trend', 'oscillator', 'deviation', 'cycle', 'pattern']:
                feature_cols = MLP_FEATURE_SETS[name]
                expert_data = latest_features[feature_cols].values.reshape(1, -1)
                
                embedding = np.zeros((1, 64)) # Default to zeros if data is invalid
                if not np.isnan(expert_data).any():
                    scaled_data = self.experts[f'{name}_scaler'].transform(expert_data)
                    embedding = self.experts[name].encode(torch.FloatTensor(scaled_data).to(next(self.experts[name].parameters()).device)).cpu().numpy()
                
                embeddings_for_master.append(embedding)
                setattr(self, f"{name}_embedding", embedding.flatten())

            # 3. Create the "super-vector" for the Master Directional Expert
            final_feature_vector = np.concatenate(embeddings_for_master, axis=1)
            final_sequence = np.tile(final_feature_vector, (self.expert_sequence_length, 1))
            final_sequence_tensor = torch.FloatTensor(final_sequence).unsqueeze(0).to(next(self.experts['master_directional'].parameters()).device)

            # 4. Get the final prediction
            dir_prediction_logits = self.experts['master_directional'](final_sequence_tensor)
            self.directional_prediction_probs = torch.nn.functional.softmax(dir_prediction_logits, dim=-1).cpu().numpy().flatten()
