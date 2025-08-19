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

    # <<< --- THE FIX: Return the raw, annualized Vega --- >>>
    # The standard formula gives the change per 100% move in IV.
    # The conversion to "per 1%" will happen in the manager class.
    vega = S * pdf_d1 * math.sqrt(T)
    return vega if math.isfinite(vega) else 0.0

@jit(nopython=True)
def _numba_theta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 1e-6 or sigma <= 1e-6 or K <= 1e-6 or S <= 1e-6: return 0.0

    d1_denominator = sigma * math.sqrt(T)
    if d1_denominator < 1e-8: return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / d1_denominator
    d2 = d1 - sigma * math.sqrt(T)

    pdf_d1 = _numba_pdf(d1)

    term1 = -(S * pdf_d1 * sigma) / (2 * math.sqrt(T))

    if is_call:
        term2 = r * K * math.exp(-r * T) * _numba_cdf(d2)
    else: # Put
        term2 = r * K * math.exp(-r * T) * _numba_cdf(-d2)

    annual_theta = term1 + term2

    return annual_theta if math.isfinite(annual_theta) else 0.0

class BlackScholesManager:
    """
    Handles all financial mathematics for option pricing and the Greeks.
    This class is mostly stateless and focused on pure calculation.
    """
    def __init__(self, cfg: Dict):
        self.risk_free_rate = cfg['risk_free_rate']

    def get_all_greeks_and_price(self, S: float, K: float, T_days: float, sigma: float, is_call: bool) -> Dict:
        T_annual = T_days / 365.25

        if T_annual <= 1e-6 or sigma <= 1e-6 or K <= 1e-6 or S <= 1e-6:
            # ... (terminal condition logic)
            price = max(0.0, S - K) if is_call else max(0.0, K - S)
            delta = 1.0 if S > K else 0.0 if is_call else -1.0 if S < K else 0.0
            return {'price': price, 'delta': delta, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'd2': 0.0}

        # 1. Get the raw, annualized values from the pure Numba functions
        price = _numba_black_scholes(S, K, T_annual, self.risk_free_rate, sigma, is_call)
        delta = _numba_delta(S, K, T_annual, self.risk_free_rate, sigma, is_call)
        gamma = _numba_gamma(S, K, T_annual, self.risk_free_rate, sigma)
        raw_annual_vega = _numba_vega(S, K, T_annual, self.risk_free_rate, sigma)
        raw_annual_theta = _numba_theta(S, K, T_annual, self.risk_free_rate, sigma, is_call)

        d1_denominator = sigma * math.sqrt(T_annual)
        d1 = (math.log(S / K) + (self.risk_free_rate + 0.5 * sigma ** 2) * T_annual) / d1_denominator
        d2 = d1 - sigma * math.sqrt(T_annual)
        
        # Convert Vega from "per 100% IV change" to "per 1% IV change"
        final_vega = raw_annual_vega / 100.0
        
        # Convert Theta from "per year" to "per calendar day"
        final_theta = raw_annual_theta / 365.25

        # --- Defensive Assertions ---
        assert math.isfinite(price), "Price calculation failed"
        assert math.isfinite(delta), "Delta calculation failed"
        assert math.isfinite(gamma), "Gamma calculation failed"
        assert math.isfinite(final_vega), "Vega calculation failed"
        assert math.isfinite(final_theta), "Theta calculation failed"
        
        # Return the final, practical values
        return {'price': price, 'delta': delta, 'gamma': gamma, 'vega': final_vega, 'theta': final_theta, 'd2': d2}

    def get_price_with_spread(self, mid_price: float, is_buy: bool, bid_ask_spread_pct: float) -> float:
        return mid_price * (1 + bid_ask_spread_pct) if is_buy else mid_price * (1 - bid_ask_spread_pct)
