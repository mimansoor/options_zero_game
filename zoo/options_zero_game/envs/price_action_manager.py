# zoo/options_zero_game/envs/price_action_manager.py
# <<< DEFINITIVE SOTA VERSION >>>

import os
import random
import math
import joblib
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

# --- Import the necessary components from the expert trainer scripts ---
from zoo.options_zero_game.experts.transformer_expert_trainer import TransformerExpert, HybridGRUTransformerExpert, CONFIG as ExpertConfig

class PriceActionManager:
    """
    Handles all logic for generating and stepping through market price data.
    Can use historical data or an expert-driven GARCH model.
    It also runs all expert models (Holy Trinity and Transformers) to provide
    rich features for the agent's observation.
    """
    def __init__(self, cfg: Dict, np_random: Any):
        self._cfg = cfg
        self.price_source = cfg['price_source']
        self.historical_data_path = cfg['historical_data_path']
        self.market_regimes = cfg['market_regimes']
        self.start_price_config = cfg['start_price']
        self.steps_per_day = cfg['steps_per_day']
        self.momentum_window_steps = cfg['momentum_window_steps']
        self.np_random = np_random
        self.expert_sequence_length = cfg.get('expert_sequence_length', 60)

        self._available_tickers = self._load_tickers()

        # --- State Variables ---
        self.price_path: np.ndarray = np.array([])
        self.historical_context_path: np.ndarray = np.array([])
        self.sma_path: np.ndarray = np.array([])
        self.current_price: float = 0.0
        self.momentum_signal: float = 0.0
        self.start_price: float = 0.0
        self.current_regime_name: str = ""
        self.episode_iv_anchor: float = 0.2

        # --- Holy Trinity State ---
        self.ema_expert, self.rsi_expert = None, None
        self.expert_ema_pred: float = 0.0
        self.expert_rsi_pred: np.ndarray = np.array([0.33, 0.34, 0.33])
        self._load_holy_trinity_experts()

        # --- Transformer Experts State ---
        self.volatility_expert = None
        self.directional_expert = None
        self.volatility_embedding = None
        self.directional_prediction = 0.0
        self._load_transformer_experts()

    def _load_tickers(self) -> List[str]:
        if self.price_source in ['historical', 'mixed']:
            if not os.path.isdir(self.historical_data_path):
                raise FileNotFoundError(f"Historical data path not found: {self.historical_data_path}")
            tickers = [f.replace('.csv', '') for f in os.listdir(self.historical_data_path) if f.endswith('.csv')]
            if not tickers: raise FileNotFoundError(f"No CSV files found in {self.historical_data_path}")
            return tickers
        return []

    def _load_holy_trinity_experts(self):
        """Loads the three LightGBM expert models."""
        try:
            self.ema_expert = joblib.load('zoo/options_zero_game/experts/ema_expert.joblib')
            self.rsi_expert = joblib.load('zoo/options_zero_game/experts/rsi_expert.joblib')
        except FileNotFoundError:
            print("Warning: Holy Trinity models not found. Run holy_trinity_trainer.py.")

    def _load_transformer_experts(self):
        """Rebuilds the Transformer architectures and loads their trained weights."""
        device = 'cpu' # Use CPU for inference in the environment for stability
        try:
            vol_params = ExpertConfig['volatility_expert_params']
            self.volatility_expert = TransformerExpert(
                input_dim=vol_params['input_dim'], model_dim=ExpertConfig['embedding_dim'],
                num_heads=ExpertConfig['num_heads'], num_layers=vol_params['num_layers'],
                output_dim=vol_params['output_dim']
            ).to(device)
            state_dict = torch.load('zoo/options_zero_game/experts/volatility_expert.pth', map_location=device)
            self.volatility_expert.load_state_dict(state_dict)
            self.volatility_expert.eval()
            print("Successfully loaded Volatility Transformer expert.")
        except FileNotFoundError:
            print("Warning: Volatility Transformer expert not found. Run trainer first.")

        try:
            dir_params = ExpertConfig['directional_expert_params']
            self.directional_expert = HybridGRUTransformerExpert(
                input_dim=dir_params['input_dim'], model_dim=ExpertConfig['embedding_dim'],
                num_heads=ExpertConfig['num_heads'], num_layers=dir_params['num_layers'],
                gru_layers=dir_params['gru_layers'], output_dim=dir_params['output_dim']
            ).to(device)
            state_dict = torch.load('zoo/options_zero_game/experts/directional_expert.pth', map_location=device)
            self.directional_expert.load_state_dict(state_dict)
            self.directional_expert.eval()
            print("Successfully loaded Directional Transformer expert.")
        except FileNotFoundError:
            print("Warning: Directional Transformer expert not found. Run trainer first.")

    def reset(self, total_steps: int):
        self.total_steps = total_steps
        self.start_price = self.start_price_config
        source_to_use = random.choice(['garch', 'historical']) if self.price_source == 'mixed' else self.price_source
        self.historical_context_path = np.array([])

        try:
            # ... (the try/except block for calling _generate_*_price_path) ...
            if source_to_use == 'garch' and self.market_regimes:
                self._generate_garch_price_path()
            elif source_to_use == 'historical' and self._available_tickers:
                self._generate_historical_price_path()
            else:
                raise ValueError("No valid price source could be selected.")
                
            if np.any(self.price_path <= 0): raise ValueError("Generated path contains non-positive prices")

        except Exception as e:
            # ... (the except block) ...
            print(f"--- PRICE GENERATION FAILED: {e}. USING FAILSAFE (FLAT) PATH. ---")
            self.current_regime_name = "Failsafe (Flat)"
            self.price_path = np.full(self.total_steps + 1, self.start_price, dtype=np.float32)
            self.episode_iv_anchor = 0.2

        # Use pandas for a fast, clean rolling average calculation
        price_series = pd.Series(self.price_path)
        self.sma_path = price_series.rolling(
            window=self.momentum_window_steps,
            min_periods=1 # Calculate even if window isn't full at the start
        ).mean().to_numpy(dtype=np.float32)
        
        # --- Defensive Assertions ---
        # Ensure the generated price path is valid before starting the episode.
        assert len(self.price_path) == self.total_steps + 1, f"Price path length is {len(self.price_path)}, expected {self.total_steps + 1}"
        assert np.all(np.isfinite(self.sma_path)), "SMA path contains invalid numbers."
        assert np.all(np.isfinite(self.price_path)), "Price path contains NaN or Inf values."
        assert np.all(self.price_path > 0), "Price path contains non-positive values."

        self.current_price = self.price_path[0]

    def step(self, current_step: int):
        """Updates the manager's state for the current step and runs all expert predictions."""
        self.current_price = self.price_path[current_step]
        
        # --- Standard Predictions ---
        self.momentum_signal = math.tanh((self.current_price / self.sma_path[current_step]) - 1.0) if self.sma_path[current_step] > 1e-6 else 0.0

        # --- Holy Trinity Predictions ---
        # --- THE FIX: Get predictions ONLY from the Holy Trinity experts ---
        if self.ema_expert and self.rsi_expert
            features = self._get_features_for_experts(current_step)
            if features is not None:
                self.expert_ema_pred = self.ema_expert.predict(features)[0]
                self.expert_rsi_pred = self.rsi_expert.predict_proba(features)[0]

        # --- Hierarchical Transformer Expert Pipeline ---
        if self.volatility_expert and self.directional_expert and current_step >= self.expert_sequence_length:
            raw_history_tensor = self._get_raw_history_for_transformer(current_step)
            
            with torch.no_grad():
                # 1. Run Volatility Expert to get the context embedding
                vol_embedding = self.volatility_expert.encode(raw_history_tensor.unsqueeze(0))
                self.volatility_embedding = vol_embedding.cpu().numpy().flatten()

                # 2. Fuse features for the Directional Expert
                vol_embedding_tiled = vol_embedding.repeat(self.expert_sequence_length, 1)
                directional_input = torch.cat([raw_history_tensor, vol_embedding_tiled], dim=1)

                # 3. Run Directional Expert on the fused input
                dir_prediction_logits = self.directional_expert(directional_input.unsqueeze(0))
                dir_prediction_probs = torch.nn.functional.softmax(dir_prediction_logits, dim=-1)
                self.directional_prediction = dir_prediction_probs[0, 0].item()

    def _get_raw_history_for_transformer(self, current_step: int) -> torch.Tensor:
        """A helper to prepare the input tensor for the Transformer models."""
        start_index = current_step - self.expert_sequence_length
        end_index = current_step
        
        price_sequence = self.price_path[start_index:end_index]
        
        # Calculate features (must match the trainer)
        log_returns = np.log(price_sequence[1:] / price_sequence[:-1])
        log_returns = np.insert(log_returns, 0, 0) # Pad to maintain length
        
        volatility = pd.Series(log_returns).rolling(window=14).std().fillna(0).values
        
        # Stack features and convert to tensor
        feature_array = np.stack([log_returns, volatility], axis=1)
        return torch.FloatTensor(feature_array)

    def _generate_historical_price_path(self):
        forced_symbol = self._cfg.get('forced_historical_symbol')
        if forced_symbol:
            # Check if the requested symbol is valid
            if forced_symbol in self._available_tickers:
                selected_ticker = forced_symbol
                print(f"(INFO) Using forced historical symbol: {selected_ticker}")
            else:
                # Fallback if the requested symbol doesn't exist in the cache
                print(f"(WARNING) Forced symbol '{forced_symbol}' not found. Choosing a random ticker instead.")
                selected_ticker = random.choice(self._available_tickers)
        else:
            selected_ticker = random.choice(self._available_tickers)

        file_path = os.path.join(self.historical_data_path, f"{selected_ticker}.csv")
        data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        data.rename(columns={data.columns[0]: 'Close'}, inplace=True)

        if len(data) < self.total_steps + 1:
            raise ValueError(f"Historical data for {selected_ticker} is too short.")

        start_index = random.randint(0, len(data) - self.total_steps - 1)

        # 1. Capture the price data *before* the episode starts, safely handling edge cases.
        context_start_index = max(0, start_index - 30)
        context_segment = data['Close'].iloc[context_start_index:start_index]
        
        # 2. Slice the main episode data.
        price_segment = data['Close'].iloc[start_index:start_index + self.total_steps + 1]
        
        raw_start_price = price_segment.iloc[0]
        if raw_start_price > 1e-4:
            normalization_factor = self.start_price_config / raw_start_price
            self.price_path = (price_segment * normalization_factor).to_numpy(dtype=np.float32)
            # 3. Normalize the historical context with the SAME factor for a continuous chart.
            self.historical_context_path = (context_segment * normalization_factor).to_numpy(dtype=np.float32)
        else:
            raise ValueError(f"Corrupt data for {selected_ticker} (start price <= 0).")

        self.start_price = self.price_path[0]
        self.current_regime_name = f"Historical: {selected_ticker}"
        log_returns = np.log(self.price_path[1:] / self.price_path[:-1])
        annualized_vol = np.std(log_returns) * np.sqrt(252 * self.steps_per_day)
        self.episode_iv_anchor = annualized_vol if np.isfinite(annualized_vol) else 0.2
        self.trend = np.mean(log_returns) * (252 * self.steps_per_day)

    def _generate_garch_price_path(self):
        """
        Generates a price path using a simulation that is dynamically
        CALIBRATED by the pre-trained Volatility Transformer expert. This ensures
        the synthetic data is self-consistent with the expert's learned patterns.
        """
        # --- 1. Setup ---
        chosen_regime = random.choice(self.market_regimes)
        self.current_regime_name = f"GARCH: {chosen_regime['name']}"

        prices = np.zeros(self.total_steps + 1, dtype=np.float32)
        prices[0] = self.start_price_config

        # We need to bootstrap the simulation with some initial noise.
        # Create a small buffer of initial prices to generate the first sequence.
        if self.total_steps > self.expert_sequence_length:
            initial_vol = chosen_regime['atm_iv'] / 100.0 / np.sqrt(252 * self.steps_per_day)
            for i in range(1, self.expert_sequence_length + 1):
                step_return = self.np_random.normal(0, initial_vol)
                prices[i] = prices[i-1] * np.exp(step_return)

        # --- 2. The Expert-Driven Simulation Loop ---
        # We start the main loop after our initial bootstrap period.
        for t in range(self.expert_sequence_length + 1, self.total_steps + 1):

            # --- a. Get the Expert's Prediction ---
            # Default to the regime's base IV if the expert isn't available
            predicted_vol = chosen_regime['atm_iv'] / 100.0

            if self.volatility_expert:
                # Prepare the feature sequence for the expert from the data we've just generated
                # The helper function gets the last `expert_sequence_length` steps
                history_tensor = self._get_raw_history_for_transformer(t)

                with torch.no_grad():
                    # Use .forward() to get the final, single-number prediction
                    # .item() extracts the Python float value from the tensor.
                    predicted_vol = self.volatility_expert.forward(history_tensor.unsqueeze(0)).item()

            # --- b. Use the Prediction as the Volatility for the Next Step ---
            # Clamp the prediction to a sensible range to prevent explosive behavior
            annual_vol = np.clip(predicted_vol, 0.10, 2.50) # Capped at 250% IV

            # De-annualize to get the volatility for a single time step
            step_vol = annual_vol / np.sqrt(252 * self.steps_per_day)

            # --- c. Generate the Next Price Step ---
            # The expert provides the magnitude, the random shock provides the direction
            shock = self.np_random.normal(0, 1)
            step_return = shock * step_vol

            # (Optional but recommended) Add overnight/weekend gap logic here
            # ...

            prices[t] = prices[t-1] * np.exp(step_return)

            # Failsafe to prevent the price from going to zero or negative
            if prices[t] <= 0.01:
                prices[t] = prices[t-1]

        # --- 3. Finalize the Price Path ---
        self.price_path = prices
        self.start_price = self.price_path[0]
        self.episode_iv_anchor = chosen_regime['atm_iv'] / 100.0

