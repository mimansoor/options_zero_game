# zoo/options_zero_game/envs/utils.py
import numpy as np
from typing import Dict

def generate_dynamic_iv_skew_table(max_offset: int, atm_iv: float, far_otm_put_iv: float, far_otm_call_iv: float) -> Dict:
    """
    Generates a realistic, dynamic IV skew table using a quadratic curve.
    This creates a "volatility smirk" where OTM puts have the highest IV.
    """
    A = np.array([
        [max_offset**2, -max_offset, 1],
        [0, 0, 1],
        [max_offset**2, max_offset, 1]
    ])
    B = np.array([far_otm_put_iv, atm_iv, far_otm_call_iv])
    a, b, c = np.linalg.solve(A, B)
    skew_table = {'call': {}, 'put': {}}
    for offset in range(-max_offset, max_offset + 1):
        iv = a * offset**2 + b * offset + c
        iv_range = [max(5.0, iv - 1.0), iv + 1.0]
        skew_table['call'][str(offset)] = iv_range
        skew_table['put'][str(offset)] = iv_range
    return skew_table
