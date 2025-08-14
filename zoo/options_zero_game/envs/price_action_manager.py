# zoo/options_zero_game/envs/price_action_manager.py
import torch # Make sure torch is imported
import os, random, math, joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Any
from zoo.options_zero_game.experts.transformer_expert_trainer import TransformerExpert, CONFIG as ExpertConfig

class PriceActionManager:
    """
    Handles all logic for generating and stepping through market price data.
    Can use either a GARCH(1,1) model or historical data.
    """
    def __init__(self, cfg: Dict, np_random: Any):
        self._cfg = cfg
        self.price_source = cfg['price_source']
        self.historical_data_path = cfg['historical_data_path']
        self.market_regimes = cfg['market_regimes']
        self.start_price_config = cfg['start_price']
        self.total_steps = cfg['time_to_expiry_days'] * cfg['steps_per_day']
        self.steps_per_day = cfg['steps_per_day']
        self.momentum_window_steps = cfg['momentum_window_steps']
        self.expert_sequence_length = cfg.get('expert_sequence_length', 60)

        # --- NEW: Load the expert models ---
        self.trend_expert = None
        self.volatility_expert = None
        self.expert_lookback = 30 # Must match the lookback used in training
        self.volatility_period = 14 # Match the trainer

        # --- THE FIX: Load ONLY the Holy Trinity expert models ---
        self.ema_expert, self.rsi_expert, self.volatility_expert = None, None, None
        try:
            self.ema_expert = joblib.load('zoo/options_zero_game/experts/ema_expert.joblib')
            self.rsi_expert = joblib.load('zoo/options_zero_game/experts/rsi_expert.joblib')
            self.volatility_expert = joblib.load('zoo/options_zero_game/experts/volatility_expert.joblib')
            print("Successfully loaded Holy Trinity expert models.")
        except FileNotFoundError:
            print("Warning: Holy Trinity models not found. Running without experts. Run holy_trinity_trainer.py.")

        # --- Load the Transformer Experts (The Correct Way) ---
        self.volatility_expert = None
        self.directional_expert = None
        self.volatility_embedding = None 
        self.directional_prediction = None
        device = 'cpu' # Always use CPU for inference in the environment

        try:
            # 1. Re-create the model architecture
            vol_params = ExpertConfig['volatility_expert_params']
            self.volatility_expert = TransformerExpert(
                input_dim=vol_params['input_dim'], model_dim=ExpertConfig['embedding_dim'],
                num_heads=ExpertConfig['num_heads'], num_layers=vol_params['num_layers'],
                output_dim=vol_params['output_dim'], dropout=ExpertConfig['dropout']
            ).to(device)
            # 2. Load the saved weights (state_dict) into the model
            state_dict = torch.load('zoo/options_zero_game/experts/volatility_expert.pth', map_location=device)
            self.volatility_expert.load_state_dict(state_dict)
            self.volatility_expert.eval() # 3. Set to evaluation mode
            print("Successfully loaded Volatility Transformer expert.")
        except FileNotFoundError:
            print("Warning: Volatility Transformer expert not found. Run trainer first.")

        try:
            # 1. Re-create the model architecture
            dir_params = ExpertConfig['directional_expert_params']
            self.directional_expert = TransformerExpert(
                input_dim=dir_params['input_dim'], model_dim=ExpertConfig['embedding_dim'],
                num_heads=ExpertConfig['num_heads'], num_layers=dir_params['num_layers'],
                output_dim=dir_params['output_dim'], dropout=ExpertConfig['dropout']
            ).to(device)
            # 2. Load the saved weights (state_dict) into the model
            state_dict = torch.load('zoo/options_zero_game/experts/directional_expert.pth', map_location=device)
            self.directional_expert.load_state_dict(state_dict)
            self.directional_expert.eval() # 3. Set to evaluation mode
            print("Successfully loaded Directional Transformer expert.")
        except FileNotFoundError:
            print("Warning: Directional Transformer expert not found. Run trainer first.")

        self.np_random = np_random
        self._available_tickers = self._load_tickers()
        
        # Time constants for GARCH gaps
        self.mins_per_step = cfg['trading_day_in_mins'] / self.steps_per_day
        self.TRADING_DAY_IN_MINS = cfg['trading_day_in_mins']
        self.MINS_IN_DAY = 24 * 60
        self.TRADING_DAYS_IN_WEEK = 5

        # State variables
        self.price_path: np.ndarray = np.array([])
        self.historical_context_path: np.ndarray = np.array([])
        self.sma_path: np.ndarray = np.array([])
        self.current_price: float = 0.0
        self.momentum_signal: float = 0.0
        self.start_price: float = 0.0
        self.current_regime_name: str = ""
        self.garch_implied_vol: float = 0.2
        self.trend: float = 0.0
        # State variables for Holy Trinity expert predictions
        self.expert_ema_pred: float = 0.0
        self.expert_rsi_pred: np.ndarray = np.array([0.33, 0.34, 0.33]) # [P(Sell), P(Neu), P(Buy)]
        self.expert_vol_pred: float = 0.0
        self.total_steps: int = 0

    def _load_tickers(self) -> List[str]:
        if self.price_source in ['historical', 'mixed']:
            if not os.path.isdir(self.historical_data_path):
                raise FileNotFoundError(f"Historical data path not found: {self.historical_data_path}")
            tickers = [f.replace('.csv', '') for f in os.listdir(self.historical_data_path) if f.endswith('.csv')]
            if not tickers: raise FileNotFoundError(f"No CSV files found in {self.historical_data_path}")
            return tickers
        return []

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
            self.garch_implied_vol = 0.2

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
        self.current_price = self.price_path[current_step]
        current_sma = self.sma_path[current_step]
        
        # Calculate momentum as a normalized ratio (-1 to 1 is a good range)
        # We use tanh for stable normalization.
        if current_sma > 1e-6:
            self.momentum_signal = math.tanh((self.current_price / current_sma) - 1.0)
        else:
            self.momentum_signal = 0.0

        # --- THE FIX: Get predictions ONLY from the Holy Trinity experts ---
        if self.ema_expert and self.rsi_expert and self.volatility_expert:
            features = self._get_features_for_experts(current_step)
            if features is not None:
                self.expert_ema_pred = self.ema_expert.predict(features)[0]
                self.expert_rsi_pred = self.rsi_expert.predict_proba(features)[0]
                self.expert_vol_pred = self.volatility_expert.predict(features)[0]    

        # Hierarchical Transformer Expert Pipeline
        # Ensure we have enough historical data to form a sequence
        if self.volatility_expert and self.directional_expert and current_step >= self.expert_sequence_length:
            
            # 1. Get raw history as a tensor
            raw_history = self._get_raw_history_for_transformer(current_step)

            with torch.no_grad():
                # 2. Run Volatility Expert to get the context embedding
                vol_embedding = self.volatility_expert.encode(raw_history.unsqueeze(0)) # Add batch dim
                self.volatility_embedding = vol_embedding.cpu().numpy().flatten()

                # 3. Fuse features for the Directional Expert
                # Tile the embedding to match the sequence length and concatenate
                vol_embedding_tiled = vol_embedding.repeat(raw_history.shape[0], 1)
                directional_input = torch.cat([raw_history, vol_embedding_tiled], dim=1)

                # 4. Run Directional Expert on the fused input
                dir_prediction_logits = self.directional_expert(directional_input.unsqueeze(0)) # Add batch dim
                dir_prediction_probs = torch.nn.functional.softmax(dir_prediction_logits, dim=-1)
                self.directional_prediction = dir_prediction_probs.cpu().numpy().flatten()

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
            
        return feature_vector.reshape(1, -1)

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
        self.garch_implied_vol = annualized_vol if np.isfinite(annualized_vol) else 0.2
        self.trend = np.mean(log_returns) * (252 * self.steps_per_day)

    def _generate_garch_price_path(self):
        chosen_regime = random.choice(self.market_regimes)
        self.current_regime_name = chosen_regime['name']
        mu, omega, alpha, beta = chosen_regime['mu'], chosen_regime['omega'], chosen_regime['alpha'], chosen_regime['beta']
        overnight_vol_multiplier = chosen_regime.get('overnight_vol_multiplier', 1.5)
        
        if alpha + beta >= 1.0:
            total = alpha + beta
            alpha, beta = alpha / (total + 0.01), beta / (total + 0.01)

        mu_step = mu / (252 * self.steps_per_day)
        omega_step = omega / (252 * self.steps_per_day)

        returns = np.zeros(self.total_steps + 1, dtype=np.float32)
        variances = np.zeros(self.total_steps + 1, dtype=np.float32)

        initial_variance_denom = 1 - alpha - beta
        initial_variance = omega / initial_variance_denom if initial_variance_denom > 1e-9 else omega
        variances[0] = initial_variance / (252 * self.steps_per_day)
        variance_cap = variances[0] * 200.0

        trading_day = 0
        for t in range(1, self.total_steps + 1):
            variances[t] = omega_step + alpha * (returns[t-1]**2) + beta * variances[t-1]
            variances[t] = min(max(1e-9, variances[t]), variance_cap)

            shock = self.np_random.normal(0, 1)
            step_return = mu_step + math.sqrt(variances[t]) * shock

            if t % self.steps_per_day == 0:
                is_friday = (trading_day % self.TRADING_DAYS_IN_WEEK) == 4
                overnight_steps = (self.MINS_IN_DAY - self.TRADING_DAY_IN_MINS) / self.mins_per_step
                if is_friday: overnight_steps += (self.MINS_IN_DAY * 2) / self.mins_per_step
                
                overnight_variance = variances[t] * overnight_steps * (overnight_vol_multiplier ** 2)
                overnight_return = (mu_step * overnight_steps) + (math.sqrt(overnight_variance) * self.np_random.normal(0, 1))
                step_return += overnight_return
                trading_day += 1

            returns[t] = step_return

        self.price_path = self.start_price_config * np.exp(np.cumsum(np.clip(returns, -0.5, 0.5)))
        self.start_price = self.price_path[0]
        self.garch_implied_vol = math.sqrt(initial_variance)

