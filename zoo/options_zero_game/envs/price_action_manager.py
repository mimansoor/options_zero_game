# zoo/options_zero_game/envs/price_action_manager.py
# <<< DEFINITIVE SOTA VERSION >>>

import os
import random
import math
import joblib
import torch
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, List

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
        self.market_regimes = cfg.get('unified_regimes', [])
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
        self.expert_lookback = 30 # Must match the lookback used in training
        self.expert_rsi_pred: np.ndarray = np.array([0.33, 0.34, 0.33])
        self._load_holy_trinity_experts()

        # --- Transformer Experts State ---
        self.volatility_expert = None
        self.directional_expert = None
        self.volatility_embedding = None
        self.directional_prediction = 0.0
        self._load_transformer_experts()
        self.volatility_transformer_prediction = 0.20 # Default value

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
                
            # --- THE FIX: Add an explicit check for NaN and Inf ---
            if not np.all(np.isfinite(self.price_path)):
                raise ValueError("Generated path contains NaN or Inf values.")
            if np.any(self.price_path <= 0):
                raise ValueError("Generated path contains non-positive prices.")

        except Exception as e:
            # ... (the except block) ...
            print(f"--- PRICE GENERATION FAILED: {type(e).__name__}: {e}. USING FAILSAFE (FLAT) PATH. ---")
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
        if self.ema_expert and self.rsi_expert:
            features = self._get_features_for_experts(current_step)
            if features is not None:
                self.expert_ema_pred = self.ema_expert.predict(features)[0]
                self.expert_rsi_pred = self.rsi_expert.predict_proba(features)[0]

        # --- Hierarchical Transformer Expert Pipeline ---
        if self.volatility_expert and self.directional_expert and current_step >= self.expert_sequence_length:
            raw_history_tensor = self._get_raw_history_for_transformer(current_step, self.price_path)
            
            with torch.no_grad():
                # 1. Run Volatility Expert to get the context embedding
                vol_embedding = self.volatility_expert.encode(raw_history_tensor.unsqueeze(0))
                self.volatility_embedding = vol_embedding.cpu().numpy().flatten()
                self.volatility_transformer_prediction = self.volatility_expert.forward(raw_history_tensor.unsqueeze(0)).item()

                # 2. Fuse features for the Directional Expert
                vol_embedding_tiled = vol_embedding.repeat(self.expert_sequence_length, 1)
                directional_input = torch.cat([raw_history_tensor, vol_embedding_tiled], dim=1)

                # 3. Run Directional Expert on the fused input
                dir_prediction_logits = self.directional_expert(directional_input.unsqueeze(0))
                dir_prediction_probs = torch.nn.functional.softmax(dir_prediction_logits, dim=-1)
                self.directional_prediction = dir_prediction_probs[0, 0].item()

    def _get_features_for_experts(self, current_step: int) -> np.ndarray:
        """Prepares the feature vector for the Holy Trinity experts."""
        if current_step < self.expert_lookback: 
            return None

        # A more robust start index calculation
        start_index = max(0, current_step - self.expert_lookback - 30) 
        end_index = current_step + 1
        history_df = pd.DataFrame({'Close': self.price_path[start_index:end_index]})
        
        # Calculate indicators needed for features
        history_df.ta.rsi(length=14, append=True)
        history_df['log_return'] = np.log(history_df['Close'] / history_df['Close'].shift(1))
        
        if len(history_df.dropna()) < self.expert_lookback:
            return None

        valid_history = history_df.dropna().tail(self.expert_lookback)
        
        log_return_features = valid_history['log_return'].values
        rsi_features = valid_history['RSI_14'].values
        
        feature_vector = np.concatenate([log_return_features, rsi_features])

    def _get_raw_history_for_transformer(self, current_step: int, price_history: np.ndarray) -> torch.Tensor:
        """A helper to prepare the input tensor. This version is NaN-proof."""
        start_index = current_step - self.expert_sequence_length
        price_sequence = price_history[start_index:current_step]
        
        # Failsafe for non-positive prices in the sequence
        if np.any(price_sequence <= 0):
            price_sequence = np.maximum(price_sequence, 1e-6)

        log_returns = np.log(price_sequence[1:] / price_sequence[:-1])
        log_returns = np.insert(log_returns, 0, 0)
        
        # Failsafe for NaN in log returns
        log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        volatility = pd.Series(log_returns).rolling(window=14).std().fillna(0).values
        
        feature_array = np.stack([log_returns, volatility], axis=1)
        return torch.FloatTensor(feature_array)

    def _generate_historical_price_path(self):
        forced_symbol = self._cfg.get('forced_historical_symbol')
        
        # <<< --- NEW: Add a failsafe loop --- >>>
        # Try up to 5 times to find a valid historical data segment.
        for _ in range(5):
            try:
                if forced_symbol:
                    if forced_symbol in self._available_tickers:
                        selected_ticker = forced_symbol
                    else:
                        print(f"(WARNING) Forced symbol '{forced_symbol}' not found. Choosing random ticker.")
                        selected_ticker = random.choice(self._available_tickers)
                else:
                    selected_ticker = random.choice(self._available_tickers)

                file_path = os.path.join(self.historical_data_path, f"{selected_ticker}.csv")
                data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                data.rename(columns={data.columns[0]: 'Close'}, inplace=True)

                # <<< --- NEW: More Robust Data Length Check --- >>>
                required_length = self.total_steps + 1
                if len(data) < required_length:
                    # Don't raise an error, just log a warning and try another ticker
                    print(f"(WARNING) Historical data for {selected_ticker} is too short ({len(data)} rows), needs {required_length}. Trying another ticker.")
                    continue # Go to the next iteration of the for loop

                # <<< --- NEW: Safer Index Calculation --- >>>
                max_start_index = len(data) - required_length
                # Ensure the upper bound of randint is not negative
                if max_start_index < 0:
                    print(f"(WARNING) Logic error for {selected_ticker}: max_start_index is negative. Trying another ticker.")
                    continue
                
                start_index = random.randint(0, max_start_index)
                
                context_start_index = max(0, start_index - 30)
                context_segment = data['Close'].iloc[context_start_index:start_index]
                
                price_segment = data['Close'].iloc[start_index : start_index + required_length]
                
                raw_start_price = price_segment.iloc[0]
                if raw_start_price > 1e-4:
                    normalization_factor = self.start_price_config / raw_start_price
                    self.price_path = (price_segment * normalization_factor).to_numpy(dtype=np.float32)
                    self.historical_context_path = (context_segment * normalization_factor).to_numpy(dtype=np.float32)
                else:
                    # This could be another source of silent failure
                    print(f"(WARNING) Corrupt data for {selected_ticker} (start price <= 0). Trying another ticker.")
                    continue

                self.start_price = self.price_path[0]
                self.current_regime_name = f"Historical: {selected_ticker}"
                log_returns = np.log(self.price_path[1:] / self.price_path[:-1])
                annualized_vol = np.std(log_returns) * np.sqrt(252 * self.steps_per_day)
                self.episode_iv_anchor = annualized_vol if np.isfinite(annualized_vol) else 0.2
                self.trend = np.mean(log_returns) * (252 * self.steps_per_day)

                # If we successfully reach this point, break out of the retry loop
                return

            except Exception as e:
                # Catch any other unexpected errors during file processing
                print(f"(WARNING) Unexpected error processing {selected_ticker}: {e}. Trying another ticker.")
                continue
        
        # If the loop finishes without returning, it means we failed 5 times.
        # This is the final failsafe from the original code.
        raise ValueError("Failed to generate a valid historical price path after 5 attempts.")

    def _generate_garch_price_path(self):
        """
        The definitive GARCH generator. It is calibrated by the Volatility Expert,
        numerically stable, and correctly handles short episode lengths.
        """
        # --- 1. Setup ---
        chosen_regime = random.choice(self.market_regimes)
        self.current_regime_name = f"GARCH: {chosen_regime['name']}"
        prices = np.zeros(self.total_steps + 1, dtype=np.float32)
        prices[0] = self.start_price_config

        # --- 2. Bootstrap Phase (The Fix is Here) ---
        # Determine the correct number of steps to bootstrap.
        # It's either the full sequence length or the entire episode if it's too short.
        bootstrap_end_step = min(self.expert_sequence_length + 1, self.total_steps + 1)

        initial_vol = chosen_regime['atm_iv'] / 100.0 / np.sqrt(252 * self.steps_per_day)
        for i in range(1, bootstrap_end_step):
            step_return = self.np_random.normal(0, initial_vol)
            prices[i] = prices[i-1] * np.exp(np.clip(step_return, -0.5, 0.5))
            if prices[i] <= 0.01: prices[i] = prices[i-1] # Safety

        # --- 3. The Expert-Driven Simulation Loop ---
        # This loop will now only run if the episode is long enough.
        for t in range(self.expert_sequence_length + 1, self.total_steps + 1):
            predicted_vol = chosen_regime['atm_iv'] / 100.0
            if self.volatility_expert:
                history_tensor = self._get_raw_history_for_transformer(t, prices)
                with torch.no_grad():
                    prediction = self.volatility_expert.forward(history_tensor.unsqueeze(0)).item()
                    if np.isfinite(prediction):
                        predicted_vol = prediction

            annual_vol = np.clip(predicted_vol, 0.10, 2.50)
            step_vol = annual_vol / np.sqrt(252 * self.steps_per_day)
            shock = self.np_random.normal(0, 1)
            step_return = shock * step_vol
            stable_step_return = np.clip(step_return, -0.5, 0.5)
            prices[t] = prices[t-1] * np.exp(stable_step_return)

            if not np.isfinite(prices[t]) or prices[t] <= 0.01:
                # This debug print is still valuable as a final safety net
                prices[t] = prices[t-1]

        # --- 4. Finalize the Price Path ---
        self.price_path = prices
        self.start_price = self.price_path[0]
        self.garch_implied_vol = chosen_regime['atm_iv'] / 100.0

