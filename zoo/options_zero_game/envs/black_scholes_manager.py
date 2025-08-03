# zoo/options_zero_game/envs/black_scholes_manager.py
import math
import numpy as np
from typing import Tuple, Dict
from numba import jit

# --- Numba JIT-compiled functions for performance ---

@jit(nopython=True)
def _numba_erf(x):
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1 = 0.254829592; a2 = -0.284496736; a3 = 1.421413741
    a4 = -1.453152027; a5 = 1.061405429; p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y

@jit(nopython=True)
def _numba_cdf(x):
    return 0.5 * (1 + _numba_erf(x / math.sqrt(2.0)))

@jit(nopython=True)
def _numba_pdf(x):
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

@jit(nopython=True)
def _numba_black_scholes(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 1e-6 or K <= 1e-6 or S <= 1e-6 or sigma <= 1e-6:
        return max(0.0, S - K) if is_call else max(0.0, K - S)

    d1_denominator = sigma * math.sqrt(T)
    if d1_denominator < 1e-8: return max(0.0, S - K) if is_call else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / d1_denominator
    d2 = d1 - sigma * math.sqrt(T)

    if is_call:
        price = S * _numba_cdf(d1) - K * math.exp(-r * T) * _numba_cdf(d2)
    else:
        price = K * math.exp(-r * T) * _numba_cdf(-d2) - S * _numba_cdf(-d1)
    return price if math.isfinite(price) else 0.0

@jit(nopython=True)
def _numba_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 1e-6 or sigma <= 1e-6 or K <= 1e-6 or S <= 1e-6:
        return 1.0 if S > K else 0.0 if is_call else -1.0 if S < K else 0.0

    d1_denominator = sigma * math.sqrt(T)
    if d1_denominator < 1e-8: return 1.0 if S > K else 0.0 if is_call else -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / d1_denominator
    delta = _numba_cdf(d1) if is_call else _numba_cdf(d1) - 1.0
    return delta if math.isfinite(delta) else 0.0

@jit(nopython=True)
def _numba_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 1e-6 or sigma <= 1e-6 or K <= 1e-6 or S <= 1e-6: return 0.0

    d1_denominator = sigma * math.sqrt(T)
    if d1_denominator < 1e-8: return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / d1_denominator
    
    pdf_d1 = _numba_pdf(d1)
    gamma = pdf_d1 / (S * d1_denominator)
    return gamma if math.isfinite(gamma) else 0.0

@jit(nopython=True)
def _numba_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 1e-6 or sigma <= 1e-6 or K <= 1e-6 or S <= 1e-6: return 0.0

    d1_denominator = sigma * math.sqrt(T)
    if d1_denominator < 1e-8: return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / d1_denominator

    pdf_d1 = _numba_pdf(d1)
    vega = (S * pdf_d1 * math.sqrt(T)) / 100.0
    return vega if math.isfinite(vega) else 0.0

@jit(nopython=True)
def _numba_theta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 1e-6 or sigma <= 1e-6 or K <= 1e-6 or S <= 1e-6: return 0.0

    d1_denominator = sigma * math.sqrt(T)
    if d1_denominator < 1e-8: return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / d1_denominator
    d2 = d1 - sigma * math.sqrt(T)
    
    pdf_d1 = _numba_pdf(d1)
    
    if is_call:
        term1 = -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
        term2 = r * K * math.exp(-r * T) * _numba_cdf(d2)
        theta = term1 - term2
    else:
        term1 = -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))
        term2 = r * K * math.exp(-r * T) * _numba_cdf(-d2)
        theta = term1 + term2

    return (theta / 365.25) if math.isfinite(theta) else 0.0


class BlackScholesManager:
    """
    Handles all financial mathematics for option pricing and the Greeks.
    This class is mostly stateless and focused on pure calculation.
    """
    def __init__(self, cfg: Dict):
        self.risk_free_rate = cfg['risk_free_rate']
        self.iv_bins = self._discretize_iv_skew(cfg['iv_skew_table'])
        self.max_strike_offset = cfg['max_strike_offset']

    def _discretize_iv_skew(self, skew_table: Dict, num_bins: int = 5) -> Dict:
        binned_ivs = {'call': {}, 'put': {}}
        for option_type, table in skew_table.items():
            for offset_str, iv_range in table.items():
                min_iv, max_iv = iv_range
                binned_ivs[option_type][offset_str] = np.linspace(min_iv, max_iv, num_bins, dtype=np.float32) / 100.0
        return binned_ivs

    def get_implied_volatility(self, offset: int, option_type: str, iv_bin_index: int) -> float:
        clamped_offset = str(max(-self.max_strike_offset, min(self.max_strike_offset, offset)))
        return self.iv_bins[option_type][clamped_offset][iv_bin_index]

    def get_all_greeks_and_price(self, S: float, K: float, T_days: float, sigma: float, is_call: bool) -> Dict:
        T_annual = T_days / 365.25

        if T_annual <= 1e-6 or sigma <= 1e-6 or K <= 1e-6 or S <= 1e-6:
            # ... (terminal condition logic)
            price = max(0.0, S - K) if is_call else max(0.0, K - S)
            delta = 1.0 if S > K else 0.0 if is_call else -1.0 if S < K else 0.0
            return {'price': price, 'delta': delta, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'd2': 0.0}

        # Calculate all values
        price = _numba_black_scholes(S, K, T_annual, self.risk_free_rate, sigma, is_call)
        delta = _numba_delta(S, K, T_annual, self.risk_free_rate, sigma, is_call)
        gamma = _numba_gamma(S, K, T_annual, self.risk_free_rate, sigma)
        vega = _numba_vega(S, K, T_annual, self.risk_free_rate, sigma)
        theta = _numba_theta(S, K, T_annual, self.risk_free_rate, sigma, is_call)
        
        d1_denominator = sigma * math.sqrt(T_annual)
        d1 = (math.log(S / K) + (self.risk_free_rate + 0.5 * sigma ** 2) * T_annual) / d1_denominator
        d2 = d1 - sigma * math.sqrt(T_annual)
        
        # --- Defensive Assertions ---
        # Ensure that no calculation resulted in a non-finite number.
        assert math.isfinite(price), f"Price calculation failed: {price} with inputs S={S}, K={K}, T={T_annual}, sigma={sigma}"
        assert math.isfinite(delta), f"Delta calculation failed: {delta}"
        assert math.isfinite(gamma), f"Gamma calculation failed: {gamma}"
        assert math.isfinite(vega), f"Vega calculation failed: {vega}"
        assert math.isfinite(theta), f"Theta calculation failed: {theta}"
        
        return {'price': price, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'd2': d2}

    def get_price_with_spread(self, mid_price: float, is_buy: bool, bid_ask_spread_pct: float) -> float:
        return mid_price * (1 + bid_ask_spread_pct) if is_buy else mid_price * (1 - bid_ask_spread_pct)
