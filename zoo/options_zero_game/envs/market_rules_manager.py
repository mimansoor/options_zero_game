# zoo/options_zero_game/envs/market_rules_manager.py
import numpy as np
from typing import Dict

class MarketRulesManager:
    """
    Encapsulates the specific, arbitrary rules of the options market itself,
    such as strike distances and the volatility skew.
    """
    def __init__(self, cfg: Dict):
        self.strike_distance = cfg['strike_distance']
        self.max_strike_offset = cfg['max_strike_offset']
        self.iv_bins = self._discretize_iv_skew(cfg['iv_skew_table'])

    def _discretize_iv_skew(self, skew_table: Dict, num_bins: int = 5) -> Dict:
        """Creates the binned IV table from the raw config."""
        binned_ivs = {'call': {}, 'put': {}}
        for option_type, table in skew_table.items():
            for offset_str, iv_range in table.items():
                min_iv, max_iv = iv_range
                binned_ivs[option_type][offset_str] = np.linspace(min_iv, max_iv, num_bins, dtype=np.float32) / 100.0
        return binned_ivs

    def get_atm_price(self, current_price: float) -> float:
        """Calculates and returns the current at-the-money strike price."""
        if self.strike_distance <= 0: return current_price
        return int(current_price / self.strike_distance + 0.5) * self.strike_distance

    def get_implied_volatility(self, offset: int, option_type: str, iv_bin_index: int) -> float:
        """Gets the correct implied volatility from the skew table for a given strike offset."""
        clamped_offset = str(max(-self.max_strike_offset, min(self.max_strike_offset, offset)))
        return self.iv_bins[option_type][clamped_offset][iv_bin_index]
