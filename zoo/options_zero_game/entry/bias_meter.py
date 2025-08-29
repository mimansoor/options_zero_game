# zoo/options_zero_game/entry/bias_meter.py
# <<< DEFINITIVE SOTA VERSION - SYNCHRONIZED WITH "COUNCIL OF EXPERTS" ARCHITECTURE >>>

import numpy as np
import math

class BiasMeter:
    """
    A definitive, SOTA Bias Meter that synthesizes the agent's observation
    vector, including the powerful outputs from the "Council of Experts", into a
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
        Calculates a directional score based on the final probabilities from
        the master Directional Transformer.
        
        Score Interpretation:
        - Highly Negative: Strong bearish signal (high P(DOWN)).
        - Around Zero: Neutral.
        - Highly Positive: Strong bullish signal (high P(UP)).
        """
        # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
        # 1. Get the final probabilities directly from the observation vector.
        # These are now the primary source of truth for direction.
        prob_down = self.obs[self.idx['DIR_EXPERT_PROB_DOWN']]
        prob_up = self.obs[self.idx['DIR_EXPERT_PROB_UP']]

        # 2. The score is simply the net difference between the bullish and bearish probabilities.
        # This naturally creates a score in the range [-1.0, 1.0].
        # The old momentum signal is no longer needed, as it was an input to the experts.
        final_score = prob_up - prob_down
        
        return float(final_score)

    def _calculate_volatility_score(self) -> float:
        """
        Calculates a volatility score. This version still uses the powerful
        128-dim embedding from the Volatility Transformer as its primary input.
        This logic remains robust and does not need to change.
        """
        # --- Feature Extraction from the Embedding ---
        start_idx = self.idx['VOLATILITY_EMBEDDING_START']
        # The embedding size is now a class attribute of the env
        embedding_size = 128 # Assuming vol_embedding_size from env
        end_idx = start_idx + embedding_size
        
        vol_embedding = self.obs[start_idx:end_idx]
        
        embedding_magnitude_score = math.tanh(np.linalg.norm(vol_embedding))

        return embedding_magnitude_score

    @property
    def directional_bias(self) -> str:
        """Returns the human-readable directional bias."""
        score = self._directional_score
        if score > 0.4: return "Strongly Bullish"
        if score > 0.1: return "Mildly Bullish"
        if score < -0.4: return "Strongly Bearish"
        if score < -0.1: return "Mildly Bearish"
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
