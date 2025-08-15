# zoo/options_zero_game/entry/bias_meter.py
# <<< DEFINITIVE SOTA VERSION >>>

import numpy as np
import math

class BiasMeter:
    """
    A definitive, SOTA Bias Meter that synthesizes the agent's observation
    vector, including the powerful Transformer expert outputs, into a
    human-readable market bias.
    """
    def __init__(self, observation_vector: np.ndarray, obs_idx_map: dict):
        self.obs = observation_vector
        self.idx = obs_idx_map

        # --- Calculate the final scores upon initialization ---
        self._directional_score = self._calculate_directional_score()
        self._volatility_score = self._calculate_volatility_score()

    def _calculate_directional_score(self) -> float:
        """
        Calculates a directional score. For the SOTA version, this relies
        almost entirely on the powerful Tail Risk Expert.
        
        Score Interpretation:
        - Highly Negative: Strong bearish signal (high tail risk).
        - Around Zero: Neutral.
        - Positive: Mildly bullish signal (low tail risk).
        """
        # --- Define Weights ---
        # The new Tail Risk expert is our most powerful directional signal.
        # Its prediction should dominate the calculation.
        WEIGHT_TAIL_RISK = 0.8
        
        # The simple momentum signal is a good secondary confirmation.
        WEIGHT_MOMENTUM = 0.2
        
        # --- Gather Signals ---
        # 1. Tail Risk Signal (from the Directional Transformer)
        # We get the probability of a "DOWN" move. This is a direct bearish signal.
        # The observation is scaled to [-1, 1], so we need to un-scale it first.
        tail_risk_prob_scaled = self.obs[self.idx['EXPERT_TAIL_RISK_PROB']]
        tail_risk_prob = (tail_risk_prob_scaled + 1.0) / 2.0
        
        # We want a score from +1 (bullish) to -1 (bearish).
        # So, we'll use (1 - prob_of_down) and scale it.
        # A 50% tail risk prob -> 0 score. 100% -> -1 score. 0% -> +1 score.
        tail_risk_signal = 1.0 - (2 * tail_risk_prob)

        # 2. Current Momentum Signal (unchanged)
        momentum_signal = self.obs[self.idx['MOMENTUM_NORM']]
        
        # --- Calculate Final Score ---
        final_score = (
            (tail_risk_signal * WEIGHT_TAIL_RISK) +
            (momentum_signal * WEIGHT_MOMENTUM)
        )
        return final_score

    def _calculate_volatility_score(self) -> float:
        """
        Calculates a volatility score. The SOTA version uses the powerful
        128-dim embedding from the Volatility Transformer as its primary input.
        """
        # --- Feature Extraction from the Embedding ---
        # The 128-dim vector is a rich, high-dimensional representation.
        # A simple, robust way to collapse this into a single "volatility score"
        # is to calculate its L2 norm (magnitude).
        # A large magnitude means the model's internal "neurons" are firing
        # intensely, which usually corresponds to a confident or extreme prediction.
        
        start_idx = self.idx['VOL_EMBEDDING_START']
        end_idx = start_idx + 128 # Assuming embedding size is 128
        
        vol_embedding = self.obs[start_idx:end_idx]
        
        # Calculate the magnitude (L2 norm) of the embedding vector.
        # We use tanh to squash the result into a predictable [-1, 1] range.
        embedding_magnitude_score = math.tanh(np.linalg.norm(vol_embedding))

        # --- Secondary Signal: The Volatility Mismatch ---
        # This tells us if the current market is behaving as expected.
        # A large mismatch is also a sign of potential volatility.
        vol_mismatch_signal = self.obs[self.idx['VOL_MISMATCH_NORM']]
        
        # --- Final Score (Weighted Average) ---
        # We'll weight the powerful embedding score more heavily.
        final_score = (embedding_magnitude_score * 0.7) + (vol_mismatch_signal * 0.3)
        return final_score

    @property
    def directional_bias(self) -> str:
        """Returns the human-readable directional bias."""
        score = self._directional_score
        if score > 0.6: return "Strongly Bullish"
        if score > 0.2: return "Mildly Bullish"
        if score < -0.6: return "Strongly Bearish"
        if score < -0.2: return "Mildly Bearish"
        return "Neutral"
    
    @property
    def volatility_bias(self) -> str:
        """Returns the human-readable volatility bias."""
        score = self._volatility_score
        if score > 0.5: return "High Volatility Expected"
        if score < -0.2: return "Low Volatility Expected"
        return "Neutral / Low Volatility Expected"

    def summary(self):
        """Prints a full summary of the bias analysis."""
        print("\n--- Bias Meter Analysis ---")
        print(f"Directional Bias: {self.directional_bias} (Score: {self._directional_score:.3f})")
        print(f"Volatility Bias:  {self.volatility_bias} (Score: {self._volatility_score:.3f})")
        print("---------------------------")
