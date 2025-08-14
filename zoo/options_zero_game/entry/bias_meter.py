# zoo/options_zero_game/entry/bias_meter.py

import numpy as np
import math

class BiasMeter:
    """
    A class that takes a full observation vector from the OptionsZeroGameEnv
    and synthesizes it into human-readable directional and volatility biases.
    """
    def __init__(self, observation_vector: np.ndarray, obs_idx_map: dict):
        self.obs = observation_vector
        self.idx = obs_idx_map

        # --- Calculate the final scores upon initialization ---
        self._directional_score = self._calculate_directional_score()
        self._volatility_score = self._calculate_volatility_score()

    def _calculate_directional_score(self) -> float:
        """
        Calculates a final directional score by combining multiple trend and momentum indicators.
        A positive score is bullish, a negative score is bearish.
        """
        # --- Define Weights for each signal ---
        # The expert's trend prediction is the most important signal.
        WEIGHT_EXPERT_TREND = 0.5
        # The current established momentum is the next most important.
        WEIGHT_CURRENT_MOMENTUM = 0.3
        # The expert's RSI prediction is a good contrary indicator.
        WEIGHT_EXPERT_RSI = 0.2
        
        # --- Gather the Signals from the Observation Vector ---
        
        # 1. Expert Trend Signal (from EMA Ratio prediction)
        # This is already a tanh'd value from -1 (bearish) to +1 (bullish).
        expert_trend_signal = self.obs[self.idx['EXPERT_EMA_RATIO']]
        
        # 2. Current Momentum Signal
        # This is the price vs. its SMA, also tanh'd from -1 to +1.
        current_momentum_signal = self.obs[self.idx['MOMENTUM_NORM']]
        
        # 3. Expert RSI Signal (as a contrarian indicator)
        # If the expert predicts a high chance of being overbought, that's a bearish signal.
        # If it predicts a high chance of being oversold, that's a bullish signal.
        prob_oversold = self.obs[self.idx['EXPERT_RSI_OVERSOLD']]
        prob_overbought = self.obs[self.idx['EXPERT_RSI_OVERBOUGHT']]
        expert_rsi_signal = prob_oversold - prob_overbought # Ranges from +1 (bullish) to -1 (bearish)

        # --- Calculate the Final Weighted Score ---
        final_score = (
            (expert_trend_signal * WEIGHT_EXPERT_TREND) +
            (current_momentum_signal * WEIGHT_CURRENT_MOMENTUM) +
            (expert_rsi_signal * WEIGHT_EXPERT_RSI)
        )
        
        return final_score

    def _calculate_volatility_score(self) -> float:
        """
        Calculates a score representing the likelihood of high volatility.
        A positive score means high vol is expected.
        """
        # --- Define Weights ---
        WEIGHT_EXPERT_VOL = 0.6 # The expert's forecast is most important.
        WEIGHT_VOL_MISMATCH = 0.4 # The current market state is also important.

        # --- Gather Signals ---
        # 1. Expert Volatility Prediction (tanh'd value)
        expert_vol_signal = self.obs[self.idx['EXPERT_VOL_NORM']]
        
        # 2. Current Volatility Surprise (tanh'd value)
        vol_mismatch_signal = self.obs[self.idx['VOL_MISMATCH_NORM']]
        
        # --- Calculate Final Score ---
        final_score = (
            (expert_vol_signal * WEIGHT_EXPERT_VOL) +
            (vol_mismatch_signal * WEIGHT_VOL_MISMATCH)
        )
        return final_score

    @property
    def directional_bias(self) -> str:
        """Returns the human-readable directional bias."""
        score = self._directional_score
        if score > 0.6: return "Hugely Bullish"
        if score > 0.2: return "Mildly Bullish"
        if score < -0.6: return "Hugely Bearish"
        if score < -0.2: return "Mildly Bearish"
        return "Neutral"
    
    @property
    def volatility_bias(self) -> str:
        """Returns the human-readable volatility bias."""
        score = self._volatility_score
        # If the combined score is positive, it indicates an expectation of volatility.
        if score > 0.3: return "High Volatility Expected"
        return "Neutral / Low Volatility Expected"

    def summary(self):
        """Prints a full summary of the bias analysis."""
        print("\n--- Bias Meter Analysis ---")
        print(f"Directional Bias: {self.directional_bias} (Score: {self._directional_score:.3f})")
        print(f"Volatility Bias:  {self.volatility_bias} (Score: {self._volatility_score:.3f})")
        print("---------------------------")
