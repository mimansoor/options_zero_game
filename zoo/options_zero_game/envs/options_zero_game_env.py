import os
import copy
import math
import random
from typing import Tuple, Dict, Any, List

import gymnasium as gym
import numpy as np
import pandas as pd
from easydict import EasyDict
from gymnasium import spaces
from gymnasium.utils import seeding

from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY

# Numba JIT compilation for performance-critical math
try:
    from numba import jit
except ImportError:
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator

@jit(nopython=True)
def _numba_erf(x):
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1 =  0.254829592; a2 = -0.284496736; a3 =  1.421413741
    a4 = -1.453152027; a5 =  1.061405429; p  =  0.3275911
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    return sign*y

@jit(nopython=True)
def _numba_cdf(x):
    return 0.5 * (1 + _numba_erf(x / math.sqrt(2.0)))

@jit(nopython=True)
def _numba_black_scholes(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 1e-6 or K <= 1e-6 or S <= 1e-6:
        if is_call: return max(0.0, S - K)
        else: return max(0.0, K - S)
    if sigma <= 1e-6:
        if is_call: return max(0.0, S - K)
        else: return max(0.0, K - S)

    d1_denominator = sigma * math.sqrt(T)
    if d1_denominator < 1e-8: return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / d1_denominator
    d2 = d1 - sigma * math.sqrt(T)
    
    if is_call:
        price = S * _numba_cdf(d1) - K * math.exp(-r * T) * _numba_cdf(d2)
    else:
        price = K * math.exp(-r * T) * _numba_cdf(-d2) - S * _numba_cdf(-d1)
    return price

@jit(nopython=True)
def _numba_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 1e-6 or sigma <= 1e-6 or K <= 1e-6 or S <= 1e-6:
        if is_call: return 1.0 if S > K else 0.0
        else: return -1.0 if S < K else 0.0
    
    d1_denominator = sigma * math.sqrt(T)
    if d1_denominator < 1e-8: return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / d1_denominator
    if is_call:
        return _numba_cdf(d1)
    else:
        return _numba_cdf(d1) - 1.0

@ENV_REGISTRY.register('options_zero_game')
class OptionsZeroGameEnv(gym.Env):
    VERSION = "1.3"
    metadata = {'render.modes': ['human']}

    config = dict(
        start_price=20000.0,
        initial_cash=100000.0,
        market_regimes = [
            {'name': 'Developed_Market', 'mu': 0.00005, 'omega': 0.000005, 'alpha': 0.09, 'beta': 0.90},
        ],
        time_to_expiry_days=20,
        steps_per_day=1,
        trading_day_in_mins=375,
        rolling_vol_window=5,
        iv_skew_table={
            'call': {'-5': (13.5, 16.0), '-4': (13.3, 15.8), '-3': (13.2, 15.7), '-2': (13.1, 15.5), '-1': (13.0, 15.4), '0': (13.0, 15.3), '1': (13.0, 15.4), '2': (13.1, 15.6), '3': (13.2, 15.7), '4': (13.3, 15.8), '5': (13.5, 16.0)},
            'put':  {'-5': (14.0, 16.5), '-4': (13.8, 16.3), '-3': (13.8, 16.1), '-2': (13.6, 16.0), '-1': (13.5, 15.8), '0': (13.5, 15.8), '1': (13.5, 15.8), '2': (13.6, 16.0), '3': (13.8, 16.1), '4': (13.8, 16.3), '5': (14.0, 16.5)},
        },
        strike_distance=50.0,
        lot_size=75,
        max_positions=4,
        bid_ask_spread_pct=0.002,
        risk_free_rate=0.10,
        pnl_scaling_factor=1000,
        drawdown_penalty_weight=0.1,
        illegal_action_penalty=-1.0,
        ignore_legal_actions=True,
        otm_delta_threshold=0.15,
        itm_delta_threshold=0.85,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None) -> None:
        self._cfg = self.default_config()
        if cfg is not None:
            self._cfg.update(cfg)

        # <<< NEW: Handle price source configuration >>>
        self.price_source = self._cfg.price_source
        self.historical_data_path = self._cfg.historical_data_path
        self._available_tickers = []
        if self.price_source in ['historical', 'mixed']:
            if not os.path.isdir(self.historical_data_path):
                raise FileNotFoundError(
                    f"Historical data path not found: {self.historical_data_path}. "
                    "Please run the cache_builder.py script first."
                )
            self._available_tickers = [f.replace('.csv', '') for f in os.listdir(self.historical_data_path) if f.endswith('.csv')]
            if not self._available_tickers:
                 raise FileNotFoundError(
                    f"No CSV files found in {self.historical_data_path}. "
                    "Please run the cache_builder.py script first."
                )

        # Time Constants
        self.TRADING_DAY_IN_MINS = self._cfg.trading_day_in_mins
        self.MINS_IN_DAY = 24 * 60
        self.TRADING_DAYS_IN_WEEK = 5
        self.TOTAL_DAYS_IN_WEEK = 7
        self.UNDEFINED_RISK_CAP_MULTIPLIER = 10
        self.steps_per_day: int = self._cfg.steps_per_day
        assert self.steps_per_day <= self.TRADING_DAY_IN_MINS, "steps_per_day cannot exceed total trading minutes"
        self.mins_per_step: float = self.TRADING_DAY_IN_MINS / self.steps_per_day
        self.decay_per_step_trading: float = self.mins_per_step / self.MINS_IN_DAY
        self.decay_overnight: float = (self.MINS_IN_DAY - self.TRADING_DAY_IN_MINS) / self.MINS_IN_DAY
        self.decay_weekend: float = 2.0

        # Core Parameters
        self.start_price: float = self._cfg.start_price
        self.initial_cash: float = self._cfg.initial_cash
        self.market_regimes: List[Dict[str, Any]] = self._cfg.market_regimes
        self.rolling_vol_window: int = self._cfg.rolling_vol_window
        self.time_to_expiry_days: int = self._cfg.time_to_expiry_days
        self.total_steps: int = self.time_to_expiry_days * self.steps_per_day
        self.risk_free_rate: float = self._cfg.risk_free_rate
        self.lot_size: int = self._cfg.lot_size
        self.strike_distance: float = self._cfg.strike_distance
        self.max_positions: int = self._cfg.max_positions
        self.bid_ask_spread_pct: float = self._cfg.bid_ask_spread_pct
        self.pnl_scaling_factor: float = self._cfg.pnl_scaling_factor
        self.drawdown_penalty_weight: float = self._cfg.drawdown_penalty_weight
        self.illegal_action_penalty: float = self._cfg.illegal_action_penalty
        self.ignore_legal_actions: bool = self._cfg.ignore_legal_actions
        self.otm_threshold: float = self._cfg.otm_delta_threshold
        self.itm_threshold: float = self._cfg.itm_delta_threshold
        self.undefined_risk_cap: float = self.initial_cash * self.UNDEFINED_RISK_CAP_MULTIPLIER
        
        self.iv_bins: Dict[str, Dict[str, np.ndarray]] = self._discretize_iv_skew(self._cfg.iv_skew_table)
        
        self.strategy_name_to_id = {
            # --- Base Strategies ---
            'LONG_CALL': 0, 'SHORT_CALL': 1, 'LONG_PUT': 2, 'SHORT_PUT': 3,
            'LONG_STRADDLE': 4, 'SHORT_STRADDLE': 5,
            
            'LONG_STRANGLE_1': 6, 'SHORT_STRANGLE_1': 7,
            'LONG_STRANGLE_2': 8, 'SHORT_STRANGLE_2': 9,
            
            'LONG_IRON_FLY': 10, 'SHORT_IRON_FLY': 11,
            'LONG_IRON_CONDOR': 12, 'SHORT_IRON_CONDOR': 13,

            'LONG_VERTICAL_CALL_1': 14, 'LONG_VERTICAL_CALL_2': 15,
            'SHORT_VERTICAL_CALL_1': 16, 'SHORT_VERTICAL_CALL_2': 17,
            'LONG_VERTICAL_PUT_1': 18, 'LONG_VERTICAL_PUT_2': 19,
            'SHORT_VERTICAL_PUT_1': 20, 'SHORT_VERTICAL_PUT_2': 21,

            'LONG_BUTTERFLY_1': 22, 'SHORT_BUTTERFLY_1': 23,
            'LONG_BUTTERFLY_2': 24, 'SHORT_BUTTERFLY_2': 25,
        }
       
        self.actions_to_indices: Dict[str, int] = self._build_action_space()
        self.indices_to_actions: Dict[int, str] = {v: k for k, v in self.actions_to_indices.items()}
        self.action_space_size: int = len(self.actions_to_indices)

        self._action_space: spaces.Discrete = spaces.Discrete(self.action_space_size)
        self.market_and_portfolio_state_size = 5 + self.max_positions * 9
        self.obs_vector_size = self.market_and_portfolio_state_size + self.action_space_size
        self._observation_space: spaces.Box = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_vector_size,), dtype=np.float32)
        
        self._reward_range: Tuple[float, float] = (-1.0, 1.0)
        self.np_random: Any = None
        
        self.portfolio_columns: List[str] = ['type', 'direction', 'entry_step', 'strike_price', 'entry_premium', 'days_to_expiry', 'creation_id', 'strategy_id', 'strategy_max_profit', 'strategy_max_loss']
        self.portfolio_dtypes = {'type': 'object', 'direction': 'object', 'entry_step': 'int64', 'strike_price': 'float64', 'entry_premium': 'float64', 'days_to_expiry': 'float64', 'creation_id': 'int64', 'strategy_id': 'int64', 'strategy_max_profit': 'float64', 'strategy_max_loss': 'float64'}
        self.portfolio: pd.DataFrame = pd.DataFrame(columns=self.portfolio_columns).astype(self.portfolio_dtypes)
        #Add max and min strikes.
        self.max_offset = max(int(k) for k in self._cfg.iv_skew_table['call'].keys())
        self.min_offset = min(int(k) for k in self._cfg.iv_skew_table['call'].keys())

    def _discretize_iv_skew(self, skew_table: Dict[str, Dict[str, Tuple[float, float]]], num_bins: int = 5) -> Dict[str, Dict[str, np.ndarray]]:
        binned_ivs = {'call': {}, 'put': {}}
        for option_type, table in skew_table.items():
            for offset_str, iv_range in table.items():
                min_iv, max_iv = iv_range
                binned_ivs[option_type][offset_str] = np.linspace(min_iv, max_iv, num_bins, dtype=np.float32) / 100.0
        return binned_ivs

    def _build_action_space(self) -> Dict[str, int]:
        actions = {'HOLD': 0}; i = 1

        # --- Single Legs ---
        for offset in range(-5, 6):
            sign = '+' if offset >= 0 else ''
            actions[f'OPEN_LONG_CALL_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_SHORT_CALL_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_LONG_PUT_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_SHORT_PUT_ATM{sign}{offset}'] = i; i+=1

        # --- Multi-Leg Spreads ---
        actions['OPEN_LONG_STRADDLE_ATM'] = i; i+=1
        actions['OPEN_SHORT_STRADDLE_ATM'] = i; i+=1

        for width in [1, 2]:
            actions[f'OPEN_LONG_STRANGLE_ATM_{width}'] = i; i+=1
            actions[f'OPEN_SHORT_STRANGLE_ATM_{width}'] = i; i+=1

        # VERTICAL SPREADS ---
        for width in [1, 2]: # For spreads 1 or 2 strikes wide
            # Call Spreads (Debit/Credit)
            actions[f'OPEN_LONG_VERTICAL_CALL_{width}'] = i; i+=1 # Bull Call / Debit Spread
            actions[f'OPEN_SHORT_VERTICAL_CALL_{width}'] = i; i+=1 # Bear Call / Credit Spread
            # Put Spreads (Debit/Credit)
            actions[f'OPEN_LONG_VERTICAL_PUT_{width}'] = i; i+=1 # Bear Put / Debit Spread
            actions[f'OPEN_SHORT_VERTICAL_PUT_{width}'] = i; i+=1 # Bull Put / Credit Spread

        actions['OPEN_LONG_IRON_FLY'] = i; i+=1
        actions['OPEN_SHORT_IRON_FLY'] = i; i+=1
        actions['OPEN_LONG_IRON_CONDOR'] = i; i+=1
        actions['OPEN_SHORT_IRON_CONDOR'] = i; i+=1

        # --- Butterfly Spreads (Widths: 1 and 2 Strikes) ---
        for width in [1, 2]:
            actions[f'OPEN_LONG_CALL_FLY_{width}'] = i; i+=1
            actions[f'OPEN_SHORT_CALL_FLY_{width}'] = i; i+=1
            actions[f'OPEN_LONG_PUT_FLY_{width}'] = i; i+=1
            actions[f'OPEN_SHORT_PUT_FLY_{width}'] = i; i+=1

        # --- Closing Actions ---
        for j in range(self.max_positions):
            actions[f'CLOSE_POSITION_{j}'] = i; i+=1

        # --- Shifting/Rolling Actions ---
        for j in range(self.max_positions):
            actions[f'SHIFT_UP_POS_{j}'] = i; i+=1
            actions[f'SHIFT_DOWN_POS_{j}'] = i; i+=1

        actions['CLOSE_ALL'] = i
        return actions

    def seed(self, seed: int, dynamic_seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _generate_historical_price_path(self) -> None:
        """
        Loads a random ticker from the cache, samples a random segment,
        and normalizes it to begin at the environment's configured start_price.
        This version is designed for the clean 'Date,TICKER' CSV format.
        """
        if not self._available_tickers:
            raise RuntimeError("Historical mode selected but no tickers are available.")

        # 1. Select a random ticker and load its data
        selected_ticker = random.choice(self._available_tickers)
        file_path = os.path.join(self.historical_data_path, f"{selected_ticker}.csv")
        
        # --- CLEAN READER LOGIC ---
        # This is now much simpler because we know the format is consistent.
        # We tell pandas that the first column ('Date') is our index and to parse it as dates.
        data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        
        # The price data is in the first (and only) column.
        # We will rename it to 'Close' for clarity and consistency.
        # .columns[0] will get the ticker name (e.g., 'SPY', 'INTC')
        data.rename(columns={data.columns[0]: 'Close'}, inplace=True)
        # --- END OF CLEAN READER ---

        # 2. Ensure data is long enough for an episode
        if len(data) < self.total_steps + 1:
            print(f"Warning: {selected_ticker} data is too short. Falling back to GARCH.")
            # ... (rest of the fallback logic remains the same)
            if not hasattr(self, 'trend'):
                chosen_regime = random.choice(self.market_regimes)
                self.current_regime_name, self.trend, self.omega, self.alpha, self.beta = chosen_regime['name'], chosen_regime['mu'], chosen_regime['omega'], chosen_regime['alpha'], chosen_regime['beta']
                self.overnight_vol_multiplier = chosen_regime.get('overnight_vol_multiplier', 1.5)
            self._generate_garch_price_path()
            return

        # 3. Sample and normalize the data
        start_index = random.randint(0, len(data) - self.total_steps - 1)
        end_index = start_index + self.total_steps + 1
        price_segment = data['Close'].iloc[start_index:end_index]
        
        raw_start_price = price_segment.iloc[0]
        target_start_price = self._cfg.start_price
        
        if raw_start_price > 1e-4:
            normalization_factor = target_start_price / raw_start_price
            normalized_price_segment = price_segment * normalization_factor
        else:
            print(f"Warning: Corrupt data for {selected_ticker} (start price <= 0). Using flat path.")
            normalized_price_segment = pd.Series([target_start_price] * len(price_segment), index=price_segment.index)

        # 4. Set the price path and continue
        self.price_path = normalized_price_segment.to_numpy(dtype=np.float32)
        self.start_price = target_start_price
        self.current_regime_name = f"Historical: {selected_ticker}"
        log_returns = np.log(self.price_path[1:] / self.price_path[:-1])
        annualized_vol = np.std(log_returns) * np.sqrt(252 * self.steps_per_day)
        self.garch_implied_vol = annualized_vol if np.isfinite(annualized_vol) else 0.2
        self.trend = np.mean(log_returns) * (252 * self.steps_per_day)

    def _generate_garch_price_path(self) -> None:
        # Scale daily GARCH params to per-step
        daily_ann_factor = math.sqrt(252)
        step_ann_factor = math.sqrt(252 * self.steps_per_day)

        mu_step = self.trend / (252 * self.steps_per_day)
        omega_step = self.omega / (252 * self.steps_per_day)
        alpha_step = self.alpha / self.steps_per_day
        beta_step = self.beta ** (1/self.steps_per_day)

        returns = np.zeros(self.total_steps + 1, dtype=np.float32)
        variances = np.zeros(self.total_steps + 1, dtype=np.float32)

        # <<< FIX: Calculate unconditional variance and a cap to prevent explosion >>>
        initial_variance_denominator = 1 - alpha_step - beta_step
        # Ensure denominator is positive to avoid issues with unstable parameters
        if initial_variance_denominator <= 1e-9:
            initial_variance = omega_step / 1e-9
        else:
            initial_variance = omega_step / initial_variance_denominator

        variances[0] = initial_variance
        # Define a generous but finite cap for the variance
        variance_cap = initial_variance * 200.0

        trading_day = 0
        for t in range(1, self.total_steps + 1):
            is_end_of_day = t % self.steps_per_day == 0

            # GARCH variance equation
            new_variance = omega_step + alpha_step * (returns[t-1]**2) + beta_step * variances[t-1]

            # <<< FIX: Apply the cap to the variance calculation >>>
            variances[t] = min(max(1e-9, new_variance), variance_cap)

            # Simulate the normal intra-step return
            shock = self.np_random.normal(0, 1)
            step_return = mu_step + math.sqrt(variances[t]) * shock

            # If it's the end of the day, add the overnight/weekend gap
            if is_end_of_day:
                day_of_week = trading_day % self.TRADING_DAYS_IN_WEEK
                is_friday = day_of_week == 4

                # 1. Calculate the number of "steps" in the non-trading period
                overnight_steps = (self.MINS_IN_DAY - self.TRADING_DAY_IN_MINS) / self.mins_per_step
                if is_friday:
                    overnight_steps += (self.MINS_IN_DAY * 2) / self.mins_per_step

                # 2. Calculate the variance of the entire overnight period
                overnight_variance = variances[t] * overnight_steps * (self.overnight_vol_multiplier ** 2)

                # 3. Simulate the return for the entire overnight period
                overnight_shock = self.np_random.normal(0, 1)
                overnight_return = (mu_step * overnight_steps) + (math.sqrt(overnight_variance) * overnight_shock)

                # 4. Add the gap return to the current step's return
                step_return += overnight_return

                trading_day += 1

            returns[t] = step_return

        # Convert returns to price path (vectorized for speed)
        cumulative_returns = np.cumsum(returns)
        # Clip cumulative returns to prevent np.exp from overflowing
        cumulative_returns = np.clip(cumulative_returns, a_min=None, a_max=700)
        self.price_path = self.start_price * np.exp(cumulative_returns)

    def _simulate_price_step(self) -> None:
        self.current_price = self.price_path[self.current_step]
    
    def _get_implied_volatility(self, offset: int, option_type: str) -> float:
        clamped_offset = max(-5, min(5, offset))
        return self.iv_bins[option_type][str(clamped_offset)][self.iv_bin_index]

    def _get_option_details(self, underlying_price: float, strike_price: float, days_to_expiry: float, option_type: str) -> Tuple[float, float, float]:
        if underlying_price <= 0 or strike_price <= 0 or days_to_expiry < 0:
            return 0.0, 0.0, 0.0
        t = days_to_expiry / 365.25
        is_call = option_type == 'call'
        if t < 1e-6:
            intrinsic = max(0.0, underlying_price - strike_price) if is_call else max(0.0, strike_price - underlying_price)
            return intrinsic, 1.0 if intrinsic > 0 else 0.0, 0.0
        try:
            atm_price = 0
            if np.isfinite(self.current_price) and np.isfinite(self.strike_distance) and self.strike_distance > 0:
                atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
            offset = round((strike_price - atm_price) / self.strike_distance)
            vol = self._get_implied_volatility(offset, option_type)
            if vol <= 1e-6:
                intrinsic = max(0.0, underlying_price - strike_price) if is_call else max(0.0, strike_price - underlying_price)
                return intrinsic, 1.0 if intrinsic > 0 else 0.0, 0.0
            try:
                price = _numba_black_scholes(underlying_price, strike_price, t, self.risk_free_rate, vol, is_call)
                signed_delta = _numba_delta(underlying_price, strike_price, t, self.risk_free_rate, vol, is_call)
            except (ValueError, ZeroDivisionError):
                intrinsic = max(0.0, underlying_price - strike_price) if is_call else max(0.0, strike_price - underlying_price)
                return intrinsic, 1.0 if intrinsic > 0 else 0.0, 0.0
            if not (math.isfinite(price) and math.isfinite(signed_delta)):
                intrinsic = max(0.0, underlying_price - strike_price) if is_call else max(0.0, strike_price - underlying_price)
                return intrinsic, 1.0 if intrinsic > 0 else 0.0, 0.0
            d2 = 0.0
            if vol > 1e-6:
                d1_denominator = vol * math.sqrt(t)
                if d1_denominator >= 1e-8:
                    d1 = (math.log(underlying_price / strike_price) + (self.risk_free_rate + 0.5 * vol ** 2) * t) / d1_denominator
                    d2 = d1 - vol * math.sqrt(t)
            return price, signed_delta, d2
        except Exception:
            return 0.0, 0.0, 0.0

    def _get_option_price(self, mid_price: float, is_buy: bool) -> float:
        if is_buy: return mid_price * (1 + self.bid_ask_spread_pct)
        else: return mid_price * (1 - self.bid_ask_spread_pct)

    def _get_total_pnl(self) -> float:
        if self.portfolio.empty:
            return self.realized_pnl
        def calculate_pnl(row):
            mid_price, _, _ = self._get_option_details(self.current_price, row['strike_price'], row['days_to_expiry'], row['type'])
            current_price = self._get_option_price(mid_price, row['direction'] == 'short')
            price_diff = current_price - row['entry_premium']
            return price_diff * self.lot_size * (1 if row['direction'] == 'long' else -1)
        unrealized_pnl = self.portfolio.apply(calculate_pnl, axis=1).sum()
        return self.realized_pnl + unrealized_pnl

    def _get_current_equity(self) -> float:
        return self.initial_cash + self._get_total_pnl()

    # In the OptionsZeroGameEnv class...

    def reset(self, seed: int = None, **kwargs):
        if seed is not None: self.seed(seed)
        elif self.np_random is None: self.seed(0)

        # 1. Always start by setting a default start_price from config.
        #    This will be overwritten by the historical loader if needed.
        self.start_price = self._cfg.start_price

        source_to_use = self.price_source
        if self.price_source == 'mixed':
            source_to_use = random.choice(['garch', 'historical'])

        # 3. Generate the price path with a failsafe.
        try:
            if source_to_use == 'garch':
                # Setup GARCH parameters
                chosen_regime = random.choice(self.market_regimes)
                self.current_regime_name = chosen_regime['name']
                self.trend: float = chosen_regime['mu']
                self.omega: float = chosen_regime['omega']
                self.alpha: float = chosen_regime['alpha']
                self.beta: float = chosen_regime['beta']
                self.overnight_vol_multiplier: float = chosen_regime.get('overnight_vol_multiplier', 1.5)

                if self.alpha + self.beta >= 1.0:
                    total = self.alpha + self.beta
                    self.alpha = self.alpha / (total + 0.01)
                    self.beta = self.beta / (total + 0.01)

                # Generate the path
                self._generate_garch_price_path()

            else:  # source_to_use == 'historical'
                self._generate_historical_price_path()

            # Final check to prevent zero prices in the generated path
            if np.any(self.price_path <= 0):
                print("Warning: Generated path contains non-positive prices. Triggering failsafe.")
                raise ValueError("Invalid prices in path")

        except Exception as e:
            print(f"--- PRICE GENERATION FAILED: {e}. USING FAILSAFE PATH. ---")
            self.current_regime_name = "Failsafe (Flat)"
            # Create a simple, flat, non-zero price path to prevent crashing.
            self.price_path = np.full(self.total_steps + 1, self.start_price, dtype=np.float32)

        self.iv_bin_index: int = random.randint(0, 4)
        self.realized_vol_series = np.zeros(self.total_steps + 1, dtype=np.float32)

        self.current_step: int = 0
        self.day_of_week: int = 0
        self.current_price: float = self.price_path[0]
        self.portfolio: pd.DataFrame = pd.DataFrame(columns=self.portfolio_columns).astype(self.portfolio_dtypes)
        self.realized_pnl: float = 0.0
        self.final_eval_reward: float = 0.0
        self.high_water_mark: float = self.initial_cash
        self.illegal_action_count: int = 0
        self.next_creation_id: int = 0

        obs = self._get_observation()

        if self.ignore_legal_actions:
            mcts_action_mask = np.ones(self.action_space_size, dtype=np.int8)
        else:
            mcts_action_mask = self._get_true_action_mask()

        # Add a final debug print to be 100% sure before returning
        #print(f"DEBUG: Price path at end of reset: {self.price_path[:5]}...")

        return {'observation': obs, 'action_mask': mcts_action_mask, 'to_play': -1}

    def _close_position(self, position_index: int) -> None:
        if position_index < 0:
            position_index = len(self.portfolio) + position_index
        if 0 <= position_index < len(self.portfolio):
            pos_to_close = self.portfolio.iloc[position_index]
            mid_price, _, _ = self._get_option_details(self.current_price, pos_to_close['strike_price'], pos_to_close['days_to_expiry'], pos_to_close['type'])
            is_short = pos_to_close['direction'] == 'short'
            exit_premium = self._get_option_price(mid_price, is_buy=is_short)
            entry_premium = pos_to_close['entry_premium']
            if pos_to_close['direction'] == 'long':
                pnl = (exit_premium - entry_premium) * self.lot_size
            else:
                pnl = (entry_premium - exit_premium) * self.lot_size
            self.realized_pnl += pnl
            self.portfolio = self.portfolio.drop(self.portfolio.index[position_index]).reset_index(drop=True)

    def _calculate_shaped_reward(self, equity_before: float, equity_after: float) -> Tuple[float, float]:
        if not math.isfinite(equity_before) or not math.isfinite(equity_after):
            return 0.0, 0.0
        raw_reward = equity_after - equity_before
        self.high_water_mark = max(self.high_water_mark, equity_after)
        drawdown = self.high_water_mark - equity_after
        risk_adjusted_raw_reward = raw_reward - (self.drawdown_penalty_weight * drawdown)
        scaled_reward = risk_adjusted_raw_reward / self.pnl_scaling_factor
        final_reward = math.tanh(scaled_reward)
        assert math.isfinite(final_reward)
        return float(final_reward), raw_reward

    def _advance_time_and_market(self) -> None:
        if self.current_step >= self.total_steps:
            return
        
        time_decay_days = self.decay_per_step_trading
        is_end_of_day = (self.current_step + 1) % self.steps_per_day == 0
        if is_end_of_day:
            trading_day_index = self.current_step // self.steps_per_day
            day_of_week = trading_day_index % self.TRADING_DAYS_IN_WEEK
            if day_of_week == 4:
                time_decay_days += self.decay_weekend
            else:
                time_decay_days += self.decay_overnight
        
        self.current_step += 1
        self._simulate_price_step()
        if not self.portfolio.empty:
            self.portfolio['days_to_expiry'] = (self.portfolio['days_to_expiry'] - time_decay_days).clip(lower=0)
            expired_indices = self.portfolio[self.portfolio['days_to_expiry'] <= 1e-6].index
            if not expired_indices.empty:
                for idx in sorted(expired_indices, reverse=True):
                    self._close_position(idx)
        
        if self.current_step >= 1:
            window_size = self.rolling_vol_window * self.steps_per_day
            start_idx = max(0, self.current_step - window_size)
            prices = self.price_path[start_idx:self.current_step+1]
            if len(prices) > 1:
                log_returns = np.diff(np.log(prices))
                ann_factor = math.sqrt(252 * self.steps_per_day)
                vol = log_returns.std() * ann_factor
                self.realized_vol_series[self.current_step] = vol
            else:
                self.realized_vol_series[self.current_step] = 0.0

    def step(self, action: int) -> BaseEnvTimestep:
        true_legal_actions_mask = self._get_true_action_mask()
        was_illegal_action = true_legal_actions_mask[action] == 0
        if was_illegal_action:
            self.illegal_action_count += 1
            action = self.actions_to_indices['HOLD']
        action_name = self.indices_to_actions.get(action, 'INVALID')
        equity_before = self._get_current_equity()
        portfolio_changed = False
        if action_name.startswith('OPEN_'):
            self._handle_open_action(action_name)
            portfolio_changed = True
        elif action_name.startswith('SHIFT_'):
            self._handle_shift_action(action_name)
            portfolio_changed = True
        elif action_name.startswith('CLOSE_POSITION_'):
            pos_index = int(action_name.split('_')[-1])
            self._close_position(pos_index)
            portfolio_changed = True
        elif action_name == 'CLOSE_ALL':
            if not self.portfolio.empty:
                portfolio_changed = True
            while not self.portfolio.empty:
                self._close_position(-1)
        if portfolio_changed and not self.portfolio.empty:
            self.portfolio = self.portfolio.sort_values(by=['strike_price', 'type', 'creation_id']).reset_index(drop=True)
        self._advance_time_and_market()
        terminated = self.current_step >= self.total_steps
        if terminated:
            while not self.portfolio.empty:
                self._close_position(-1)
        equity_after = self._get_current_equity()
        shaped_reward, raw_reward = self._calculate_shaped_reward(equity_before, equity_after)
        if was_illegal_action:
            final_reward = self.illegal_action_penalty
        else:
            final_reward = shaped_reward
        obs = self._get_observation()
        self.final_eval_reward += raw_reward

        if self.ignore_legal_actions:
            mcts_action_mask = np.ones(self.action_space_size, dtype=np.int8)
        else:
            mcts_action_mask = true_legal_actions_mask

        info = {'price': self.current_price, 'eval_episode_return': self.final_eval_reward, 'illegal_actions_in_episode': self.illegal_action_count}

        observation = {'observation': obs, 'action_mask': mcts_action_mask, 'to_play': -1}
        return BaseEnvTimestep(observation, final_reward, terminated, info)

    def _open_butterfly(self, action_name: str) -> None:
        """Opens a three-strike butterfly spread with a fixed, predefined width."""
        if len(self.portfolio) > self.max_positions - 4:
            return

        # 1. Parse action name for direction, type, and width
        try:
            parts = action_name.split('_')
            direction = parts[1].upper()
            option_type = parts[2].lower()
            width_in_strikes = int(parts[4])
            width_in_price = width_in_strikes * self.strike_distance
        except (ValueError, IndexError):
            print(f"Warning: Could not parse butterfly action name '{action_name}'.")
            return

        # 2. Define Strikes based on the fixed width
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        strike_lower = atm_price - width_in_price
        strike_middle = atm_price
        strike_upper = atm_price + width_in_price

        # 3. Define Legs
        wing_direction = 'long' if direction == 'LONG' else 'short'
        body_direction = 'short' if direction == 'LONG' else 'long'

        legs_to_trade = [
            {'type': option_type, 'direction': wing_direction, 'strike_price': strike_lower},
            {'type': option_type, 'direction': body_direction, 'strike_price': strike_middle},
            {'type': option_type, 'direction': body_direction, 'strike_price': strike_middle},
            {'type': option_type, 'direction': wing_direction, 'strike_price': strike_upper},
        ]

        # 4. Price the legs
        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)

        for leg in legs_to_trade:
            leg['entry_step'] = self.current_step
            leg['days_to_expiry'] = days_to_expiry
            mid_price, _, _ = self._get_option_details(self.current_price, leg['strike_price'], leg['days_to_expiry'], leg['type'])
            leg['entry_premium'] = self._get_option_price(mid_price, is_buy=(leg['direction'] == 'long'))

        # 5. Calculate strategy PnL using the NEW SPECIFIC strategy name
        strategy_name_for_pnl = f"{direction}_BUTTERFLY_{width_in_strikes}"
        pnl = self._calculate_strategy_pnl(legs_to_trade, strategy_name_for_pnl)
        self._execute_trades(legs_to_trade, pnl)

    def _handle_shift_action(self, action_name: str) -> None:
        """
        Handles shifting a position up or down by one strike.
        This is a "roll" action: close the old position, open a new one.
        """
        try:
            # 1. Parse action name to get direction and index
            parts = action_name.split('_')
            direction = parts[1].upper()  # 'UP' or 'DOWN'
            position_index = int(parts[3])

            if not (0 <= position_index < len(self.portfolio)):
                return  # Should not happen if action mask is correct

            # 2. Get the original position's details
            original_pos = self.portfolio.iloc[position_index].copy()

            # 3. Calculate the new strike price
            strike_modifier = self.strike_distance if direction == 'UP' else -self.strike_distance
            new_strike_price = original_pos['strike_price'] + strike_modifier

            # 4. Close the original position (this realizes its PnL)
            self._close_position(position_index)

            # 5. Open the new single-leg position
            # Inherit type, direction, and DTE from the original position
            new_leg = {
                'type': original_pos['type'],
                'direction': original_pos['direction'],
                'strike_price': new_strike_price,
                'entry_step': self.current_step,
                'days_to_expiry': original_pos['days_to_expiry'],
            }

            # Price the new leg
            mid_price, _, _ = self._get_option_details(self.current_price, new_leg['strike_price'], new_leg['days_to_expiry'], new_leg['type'])
            new_leg['entry_premium'] = self._get_option_price(mid_price, is_buy=(new_leg['direction'] == 'long'))

            # Calculate its individual strategy PnL and execute the trade
            strategy_name = f"{new_leg['direction'].upper()}_{new_leg['type'].upper()}"
            strategy_pnl = self._calculate_strategy_pnl([new_leg], strategy_name)
            self._execute_trades([new_leg], strategy_pnl)

        except (ValueError, IndexError) as e:
            # Add a guard against malformed action names or indices
            print(f"Warning: Could not parse shift action '{action_name}'. Error: {e}")
            return

    def _handle_open_action(self, action_name: str) -> None:
        """
        Routes any "OPEN_" action to the correct specialized function.
        This version is robust and prevents misrouting of strategies.
        """
        # --- Route by most complex/specific keywords first ---

        if 'FLY' in action_name and 'IRON' not in action_name:
            self._open_butterfly(action_name)
        elif 'IRON_CONDOR' in action_name:
            self._open_iron_condor(action_name)
        elif 'IRON_FLY' in action_name:
            self._open_iron_fly(action_name)
        elif 'VERTICAL' in action_name:
            self._open_vertical_spread(action_name)
        elif 'STRANGLE' in action_name:
            self._open_strangle(action_name)
        elif 'STRADDLE' in action_name:
            self._open_straddle(action_name)
        elif 'ATM' in action_name:
            # This is the only case that should lead to a single leg.
            self._open_single_leg(action_name)
        else:
            assert 0, "Warning: Unrecognized action format in _handle_open_action: {action_name}"
            # Add a failsafe print for any future actions you might add
            # This helps in debugging.
            print(f"Warning: Unrecognized action format in _handle_open_action: {action_name}")

    def _execute_trades(self, trades_to_execute: List[Dict], strategy_pnl: Dict) -> None:
        if not trades_to_execute: return
        strategy_id = strategy_pnl.get('strategy_id', -1) 
        
        for trade in trades_to_execute:
            trade['creation_id'] = self.next_creation_id
            self.next_creation_id += 1
            trade['strategy_id'] = strategy_id
            trade['strategy_max_profit'] = strategy_pnl.get('max_profit', 0.0)
            trade['strategy_max_loss'] = strategy_pnl.get('max_loss', 0.0)
        new_positions_df = pd.DataFrame(trades_to_execute).astype(self.portfolio_dtypes)
        self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)

    def _calculate_strategy_pnl(self, legs: List[Dict], strategy_name: str) -> Dict:
        """
        Calculates the max profit and max loss for a given list of trade legs
        and a canonical strategy name.

        This refactored version handles:
        1. Single leg positions.
        2. Undefined-risk multi-leg positions (Straddles, Strangles).
        3. Defined-risk multi-leg positions (Verticals, Iron Condors, Iron Flies).
        """
        if not legs:
            return {'max_profit': 0.0, 'max_loss': 0.0}

        # --- Case 1: Handle Single Leg Strategies ---
        if len(legs) == 1:
            leg = legs[0]
            entry_premium_total = leg['entry_premium'] * self.lot_size
            if leg['direction'] == 'long':
                max_loss = -entry_premium_total
                # Profit is theoretically unlimited for calls, capped for puts. We use a placeholder.
                max_profit = self.undefined_risk_cap if leg['type'] == 'call' else \
                           (leg['strike_price'] * self.lot_size) - entry_premium_total
            else:  # Short
                max_profit = entry_premium_total
                # Loss is theoretically unlimited for calls, capped for puts. We use a placeholder.
                max_loss = -self.undefined_risk_cap if leg['type'] == 'call' else \
                         -((leg['strike_price'] * self.lot_size) - entry_premium_total)
            return {'max_profit': max_profit, 'max_loss': max_loss}

        # --- Case 2: Handle All Multi-Leg Strategies ---
        net_premium = sum(leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1) for leg in legs)
        is_debit_spread = net_premium > 0

        max_profit = 0.0
        max_loss = 0.0

        # First, set the "easy" side of the P&L based on debit/credit.
        # If you pay a debit, your max loss is that debit.
        # If you receive a credit, your max profit is that credit.
        if is_debit_spread:
            max_loss = -net_premium * self.lot_size
        else: # is_credit_spread
            max_profit = -net_premium * self.lot_size # net_premium is negative, so this becomes positive

        # Now, determine the other side of the P&L based on strategy type.
        # --- Undefined Risk Strategies ---
        if 'STRADDLE' in strategy_name or 'STRANGLE' in strategy_name:
            if is_debit_spread:  # Long Straddle/Strangle
                max_profit = self.undefined_risk_cap
            else:  # Short Straddle/Strangle
                max_loss = -self.undefined_risk_cap

        # --- Defined Risk Strategies ---
        # A butterfly is a defined-risk spread, so its P&L calculation is identical
        # to that of Verticals and Iron Condors/Flies. We just add it to the condition.
        elif 'VERTICAL' in strategy_name or 'IRON' in strategy_name or 'BUTTERFLY' in strategy_name:
            # For defined-risk spreads, the other side of the P&L is determined by the width of the strikes.
            call_strikes = sorted([leg['strike_price'] for leg in legs if leg['type'] == 'call'])
            put_strikes = sorted([leg['strike_price'] for leg in legs if leg['type'] == 'put'])

            # A standard butterfly is all calls or all puts, so one of these will be 0.
            # The max() function correctly finds the width of the butterfly's wings.
            call_width = (call_strikes[-1] - call_strikes[0]) if len(call_strikes) > 1 else 0
            put_width = (put_strikes[-1] - put_strikes[0]) if len(put_strikes) > 1 else 0

            # The effective width is the widest part of the spread.
            max_width = max(call_width, put_width) * self.lot_size

            if is_debit_spread:  # Long Vertical, Long Condor/Fly, Long Butterfly
                # Max profit is the width of the spread minus the debit paid.
                max_profit = max_width + max_loss  # max_loss is already negative
            else:  # Short Vertical, Short Condor/Fly, Short Butterfly
                # Max loss is the width of the spread minus the credit received.
                max_loss = -max_width + max_profit  # max_profit is already positive

        return {'max_profit': max_profit, 'max_loss': max_loss}

    def _get_trade_legs(self, action_name: str) -> List[Dict]:
        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)
        atm_price = 0
        if np.isfinite(self.current_price) and np.isfinite(self.strike_distance) and self.strike_distance > 0:
            atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        direction = 'long' if 'LONG' in action_name else 'short'
        legs = []
        if 'STRADDLE' in action_name:
            legs.append({'type': 'call', 'direction': direction, 'strike_price': atm_price})
            legs.append({'type': 'put', 'direction': direction, 'strike_price': atm_price})
        elif 'IRON_FLY' in action_name:
            straddle_dir = 'long' if direction == 'long' else 'short'
            hedge_dir = 'short' if direction == 'long' else 'long'
            hedge_strike_call = atm_price + 2 * self.strike_distance
            hedge_strike_put = atm_price - 2 * self.strike_distance
            legs.append({'type': 'call', 'direction': straddle_dir, 'strike_price': atm_price})
            legs.append({'type': 'put', 'direction': straddle_dir, 'strike_price': atm_price})
            legs.append({'type': 'call', 'direction': hedge_dir, 'strike_price': hedge_strike_call})
            legs.append({'type': 'put', 'direction': hedge_dir, 'strike_price': hedge_strike_put})
        elif 'IRON_CONDOR' in action_name:
            inner_dir = 'long' if direction == 'long' else 'short'
            outer_dir = 'short' if direction == 'long' else 'long'
            legs.append({'type': 'call', 'direction': inner_dir, 'strike_price': atm_price + 2 * self.strike_distance})
            legs.append({'type': 'put', 'direction': inner_dir, 'strike_price': atm_price - 2 * self.strike_distance})
            legs.append({'type': 'call', 'direction': outer_dir, 'strike_price': atm_price + 3 * self.strike_distance})
            legs.append({'type': 'put', 'direction': outer_dir, 'strike_price': atm_price - 3 * self.strike_distance})
        elif 'ATM' in action_name:
            _, direction, type, strike_str = action_name.split('_')
            offset = int(strike_str.replace('ATM', ''))
            strike_price = atm_price + (offset * self.strike_distance)
            legs.append({'type': type.lower(), 'direction': direction.lower(), 'strike_price': strike_price})
        for leg in legs:
            leg['entry_step'] = self.current_step
            leg['days_to_expiry'] = days_to_expiry
        return legs

    def _open_single_leg(self, action_name: str) -> None:
        if len(self.portfolio) >= self.max_positions: return
        legs = self._get_trade_legs(action_name)
        for leg in legs:
            leg['entry_premium'] = self._get_option_price(self._get_option_details(self.current_price, leg['strike_price'], leg['days_to_expiry'], leg['type'])[0], leg['direction'] == 'long')
        leg = legs[0]
        strategy_name = f"{leg['direction'].upper()}_{leg['type'].upper()}"
        pnl = self._calculate_strategy_pnl(legs, strategy_name)
        self._execute_trades(legs, pnl)

    def _open_straddle(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 2: return
        legs = self._get_trade_legs(action_name)
        for leg in legs:
            leg['entry_premium'] = self._get_option_price(self._get_option_details(self.current_price, leg['strike_price'], leg['days_to_expiry'], leg['type'])[0], leg['direction'] == 'long')
        direction = legs[0]['direction'].upper()
        strategy_name = f"{direction}_STRADDLE"
        pnl = self._calculate_strategy_pnl(legs, strategy_name)
        self._execute_trades(legs, pnl)

    def _open_strangle(self, action_name: str) -> None:
        """Opens a two-leg strangle with a variable width."""
        if len(self.portfolio) > self.max_positions - 2:
            return

        # --- 1. Construct the canonical strategy name ---
        # Turns 'OPEN_LONG_STRANGLE_ATM_1' into 'LONG_STRANGLE_1'
        canonical_strategy_name = action_name.replace('OPEN_', '').replace('_ATM', '')

        # --- 2. Parse the name to get the width ---
        try:
            width_in_strikes = int(canonical_strategy_name.split('_')[-1])
        except (IndexError, ValueError):
            print(f"Warning: Could not parse strangle action name '{action_name}'.")
            return
            
        strike_offset = width_in_strikes * self.strike_distance

        # --- 3. Define the legs ---
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        direction = 'long' if 'LONG' in action_name else 'short'
        legs = [
            {'type': 'call', 'direction': direction, 'strike_price': atm_price + strike_offset},
            {'type': 'put', 'direction': direction, 'strike_price': atm_price - strike_offset}
        ]

        # --- 4. Price and prepare the legs ---
        current_trading_day = self.current_step // self.steps_per_day
        days_to_expiry = (self.time_to_expiry_days - current_trading_day) * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)
        
        for leg in legs:
            leg['entry_step'] = self.current_step
            leg['days_to_expiry'] = days_to_expiry
            mid_price, _, _ = self._get_option_details(self.current_price, leg['strike_price'], leg['days_to_expiry'], leg['type'])
            leg['entry_premium'] = self._get_option_price(mid_price, is_buy=(leg['direction'] == 'long'))

        # --- 5. Calculate PnL and execute ---
        # Note: Strangles have undefined risk/reward, so we use a placeholder cap.
        pnl = self._calculate_strategy_pnl(legs, canonical_strategy_name)
        # Add the correct strategy_id from our lookup
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)

        self._execute_trades(legs, pnl)

    def _open_iron_fly(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 4: return
        legs = self._get_trade_legs(action_name)
        if not legs: return
        for leg in legs:
            leg['entry_premium'] = self._get_option_price(self._get_option_details(self.current_price, leg['strike_price'], leg['days_to_expiry'], leg['type'])[0], leg['direction'] == 'long')
        direction = 'LONG' if 'LONG' in action_name else 'SHORT'
        strategy_name = f"{direction}_IRON_FLY"
        pnl = self._calculate_strategy_pnl(legs, strategy_name)
        self._execute_trades(legs, pnl)

    def _open_iron_condor(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 4: return
        legs = self._get_trade_legs(action_name)
        if not legs: return
        for leg in legs:
            leg['entry_premium'] = self._get_option_price(self._get_option_details(self.current_price, leg['strike_price'], leg['days_to_expiry'], leg['type'])[0], leg['direction'] == 'long')
        direction = 'LONG' if 'LONG' in action_name else 'SHORT'
        strategy_name = f"{direction}_IRON_CONDOR"
        pnl = self._calculate_strategy_pnl(legs, strategy_name)
        self._execute_trades(legs, pnl)

    def _open_vertical_spread(self, action_name: str) -> None:
        """Opens a two-leg vertical spread, with validation for strike limits."""
        if len(self.portfolio) > self.max_positions - 2:
            return

        # --- 1. Parse the action name ---
        try:
            parts = action_name.split('_')
            direction = parts[1]
            option_type = parts[3].lower()
            # Use a clear variable name for the width in strikes (e.g., 1 or 2)
            width_in_strikes = int(parts[4])
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse vertical spread action name '{action_name}'. Error: {e}")
            return

        # --- 2. Calculate the width in price points ---
        width_in_price = width_in_strikes * self.strike_distance

        # --- 2. Define ATM and potential strike prices ---
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        
        # We need to determine the strikes BEFORE creating the legs.
        strike1 = atm_price
        strike2 = 0
        
        if option_type == 'call': # Spreads go up
            strike2 = atm_price + width_in_price
        else: # Puts, spreads go down
            strike2 = atm_price - width_in_price
            
        # --- 3. Define the legs of the spread ---
        legs_to_trade = []
        if direction == 'LONG':
            if option_type == 'call': # Bull Call
                legs_to_trade.append({'type': 'call', 'direction': 'long', 'strike_price': strike1})
                legs_to_trade.append({'type': 'call', 'direction': 'short', 'strike_price': strike2})
            else: # Bear Put
                legs_to_trade.append({'type': 'put', 'direction': 'long', 'strike_price': strike1})
                legs_to_trade.append({'type': 'put', 'direction': 'short', 'strike_price': strike2})
        else: # SHORT
            if option_type == 'call': # Bear Call
                legs_to_trade.append({'type': 'call', 'direction': 'short', 'strike_price': strike1})
                legs_to_trade.append({'type': 'call', 'direction': 'long', 'strike_price': strike2})
            else: # Bull Put
                legs_to_trade.append({'type': 'put', 'direction': 'short', 'strike_price': strike1})
                legs_to_trade.append({'type': 'put', 'direction': 'long', 'strike_price': strike2})

        # --- 3. Get current trading day and DTE ---
        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)

        # --- 4. Price the legs and prepare for execution ---
        for leg in legs_to_trade:
            leg['entry_step'] = self.current_step
            leg['days_to_expiry'] = days_to_expiry
            mid_price, _, _ = self._get_option_details(self.current_price, leg['strike_price'], leg['days_to_expiry'], leg['type'])
            leg['entry_premium'] = self._get_option_price(mid_price, is_buy=(leg['direction'] == 'long'))

        # --- 5. Calculate strategy PnL and execute ---
        # For all vertical spreads, max profit/loss is related to the width and net premium
        net_premium = sum(leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1) for leg in legs_to_trade)
        
        if net_premium > 0: # Debit spread (paid to open)
            max_loss = -net_premium * self.lot_size
            max_profit = (width_in_strikes * self.lot_size) + max_loss
        else: # Credit spread (received credit)
            max_profit = -net_premium * self.lot_size
            max_loss = -(width_in_strikes * self.lot_size) + max_profit

        strategy_pnl = {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'strategy_id': self.strategy_name_to_id.get(action_name, -1) # Use a unique ID later if needed
        }
        
        self._execute_trades(legs_to_trade, strategy_pnl)

    def _get_true_action_mask(self) -> np.ndarray:

        action_mask = np.zeros(self.action_space_size, dtype=np.int8)
        current_trading_day = self.current_step // self.steps_per_day

        # Always HOLD is a legal action.
        action_mask[self.actions_to_indices['HOLD']] = 1

        # This simple boolean check is the key.
        # If the portfolio has ANY position in it, a strategy is considered "open".
        is_strategy_open = not self.portfolio.empty

        # Expiry Day?
        is_expiry_day = current_trading_day >= self.time_to_expiry_days - 1

        # All close actions and Shift actions are allowed on non-expiry days.
        if is_strategy_open and not is_expiry_day:
            atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
            action_mask[self.actions_to_indices['CLOSE_ALL']] = 1
            for i in range(len(self.portfolio)):
                if f'CLOSE_POSITION_{i}' in self.actions_to_indices:
                    action_mask[self.actions_to_indices[f'CLOSE_POSITION_{i}']] = 1

            # --- SHIFT MASKING LOGIC START ---
            for i, pos in self.portfolio.iterrows():
                # Check legality of shifting UP
                new_strike_up = pos['strike_price'] + self.strike_distance
                offset_up = round((new_strike_up - atm_price) / self.strike_distance)
                if self.min_offset <= offset_up <= self.max_offset:
                     action_mask[self.actions_to_indices[f'SHIFT_UP_POS_{i}']] = 1

                # Check legality of shifting DOWN
                new_strike_down = pos['strike_price'] - self.strike_distance
                offset_down = round((new_strike_down - atm_price) / self.strike_distance)
                if self.min_offset <= offset_down <= self.max_offset:
                     action_mask[self.actions_to_indices[f'SHIFT_DOWN_POS_{i}']] = 1
            # --- SHIFT MASKING LOGIC END ---

        # --- ONLY ALLOW OPENING A NEW STRATEGY IF THE PORTFOLIO IS EMPTY ---
        # This entire block is skipped if a strategy is already open.
        if not is_strategy_open and not is_expiry_day:
            # --- Multi-leg strategy rules based on available slots ---
            available_slots = self.max_positions - len(self.portfolio)

            if available_slots >= 4:
                action_mask[self.actions_to_indices['OPEN_LONG_IRON_FLY']] = 1
                action_mask[self.actions_to_indices['OPEN_SHORT_IRON_FLY']] = 1
                action_mask[self.actions_to_indices['OPEN_LONG_IRON_CONDOR']] = 1
                action_mask[self.actions_to_indices['OPEN_SHORT_IRON_CONDOR']] = 1
                # <<< --- UPDATED BUTTERFLY MASKING LOGIC --- >>>
                for width in [1, 2]:
                    wing_offset_pos = +width
                    wing_offset_neg = -width
                    
                    # Check if the wings for this width are within the valid strike range
                    if (self.min_offset <= wing_offset_pos <= self.max_offset and 
                        self.min_offset <= wing_offset_neg <= self.max_offset):
                        
                        # Enable all butterfly actions for THIS SPECIFIC width
                        action_mask[self.actions_to_indices[f'OPEN_LONG_CALL_FLY_{width}']] = 1
                        action_mask[self.actions_to_indices[f'OPEN_SHORT_CALL_FLY_{width}']] = 1
                        action_mask[self.actions_to_indices[f'OPEN_LONG_PUT_FLY_{width}']] = 1
                        action_mask[self.actions_to_indices[f'OPEN_SHORT_PUT_FLY_{width}']] = 1

            if available_slots >= 2:
                # --- Straddle logic (always valid as offset is 0) ---
                action_mask[self.actions_to_indices['OPEN_LONG_STRADDLE_ATM']] = 1
                action_mask[self.actions_to_indices['OPEN_SHORT_STRADDLE_ATM']] = 1

                # --- DYNAMIC STRANGLE MASKING ---
                for width in [1, 2]:
                    call_offset = +width
                    put_offset = -width

                    # Check if this strangle width is legal
                    if (self.min_offset <= call_offset <= self.max_offset and self.min_offset <= put_offset <= self.max_offset):
                        action_name_long = f'OPEN_LONG_STRANGLE_ATM_{width}'
                        action_name_short = f'OPEN_SHORT_STRANGLE_ATM_{width}'

                        if action_name_long in self.actions_to_indices:
                            action_mask[self.actions_to_indices[action_name_long]] = 1
                        if action_name_short in self.actions_to_indices:
                            action_mask[self.actions_to_indices[action_name_short]] = 1

                # --- ENABLE VERTICAL SPREAD ACTIONS ---
                for width_in_strikes in [1, 2]:
                    for option_type in ['call', 'put']:
                        # Calculate the offset of the short/long leg
                        offset = width_in_strikes if option_type == 'call' else -width_in_strikes
                        
                        # Check if this offset is within the valid range
                        is_legal_spread = (self.min_offset <= offset <= self.max_offset)

                        if is_legal_spread:
                            # If the spread is valid, enable its corresponding actions
                            action_name_long = f'OPEN_LONG_VERTICAL_{option_type.upper()}_{width_in_strikes}'
                            action_name_short = f'OPEN_SHORT_VERTICAL_{option_type.upper()}_{width_in_strikes}'
                            
                            if action_name_long in self.actions_to_indices:
                                action_mask[self.actions_to_indices[action_name_long]] = 1
                            if action_name_short in self.actions_to_indices:
                                action_mask[self.actions_to_indices[action_name_short]] = 1

            if available_slots >= 1:
                atm_price = 0
                if np.isfinite(self.current_price) and np.isfinite(self.strike_distance) and self.strike_distance > 0:
                    atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
                trading_days_left = self.time_to_expiry_days - current_trading_day
                days_to_expiry = trading_days_left * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)
                num_long_calls = len(self.portfolio.query("type == 'call' and direction == 'long'"))
                num_short_calls = len(self.portfolio.query("type == 'call' and direction == 'short'"))
                num_long_puts = len(self.portfolio.query("type == 'put' and direction == 'long'"))
                num_short_puts = len(self.portfolio.query("type == 'put' and direction == 'short'"))
                existing_positions = {(p['strike_price'], p['type']): p['direction'] for i, p in self.portfolio.iterrows()}
                for offset in range(-5, 6):
                    strike_price = atm_price + offset * self.strike_distance
                    for option_type in ['call', 'put']:
                        _, signed_delta, _ = self._get_option_details(self.current_price, strike_price, days_to_expiry, option_type)
                        abs_delta = abs(signed_delta)
                        is_far_otm = abs_delta < self.otm_threshold
                        is_far_itm = abs_delta > self.itm_threshold
                        action_name_long = f'OPEN_LONG_{option_type.upper()}_ATM{"+" if offset >=0 else ""}{offset}'
                        if action_name_long in self.actions_to_indices:
                            is_legal = True
                            if is_far_otm or is_far_itm: is_legal = False
                            conflict_type = 'put' if option_type == 'call' else 'call'
                            if existing_positions.get((strike_price, conflict_type)) == 'short': is_legal = False
                            if (option_type == 'call' and num_long_calls > 0) or (option_type == 'put' and num_long_puts > 0): is_legal = False
                            if is_legal: action_mask[self.actions_to_indices[action_name_long]] = 1
                        action_name_short = f'OPEN_SHORT_{option_type.upper()}_ATM{"+" if offset >=0 else ""}{offset}'
                        if action_name_short in self.actions_to_indices:
                            is_legal = True
                            if is_far_itm: is_legal = False
                            conflict_type = 'put' if option_type == 'call' else 'call'
                            if existing_positions.get((strike_price, conflict_type)) == 'long': is_legal = False
                            if (option_type == 'call' and num_short_calls > 0) or (option_type == 'put' and num_short_puts > 0): is_legal = False
                            if is_legal: action_mask[self.actions_to_indices[action_name_short]] = 1

        return action_mask

    def _get_observation(self) -> np.ndarray:
        market_portfolio_vec = np.zeros(self.market_and_portfolio_state_size, dtype=np.float32)
        market_portfolio_vec[0] = (self.current_price / self.start_price) - 1.0
        market_portfolio_vec[1] = (self.total_steps - self.current_step) / self.total_steps
        total_pnl = self._get_total_pnl()
        market_portfolio_vec[2] = math.tanh(total_pnl / self.initial_cash)
        current_realized_vol = self.realized_vol_series[self.current_step]
        garch_vol = self.garch_implied_vol if hasattr(self, 'garch_implied_vol') else 0
        market_portfolio_vec[3] = math.tanh((current_realized_vol / garch_vol) - 1.0) if garch_vol > 0 else 0.0
        epsilon = 1e-8
        if self.current_step > 0:
            log_return = math.log(self.current_price / (self.price_path[self.current_step - 1] + epsilon))
        else:
            log_return = 0.0

        market_portfolio_vec[4] = np.clip(log_return, -0.1, 0.1) * 10
        current_idx = 5
        atm_price = 0
        if np.isfinite(self.current_price) and np.isfinite(self.strike_distance) and self.strike_distance > 0:
            atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        for i, pos in self.portfolio.iterrows():
            if i >= self.max_positions: break
            market_portfolio_vec[current_idx + 0] = 1.0
            market_portfolio_vec[current_idx + 1] = 1.0 if pos['type'] == 'call' else -1.0
            market_portfolio_vec[current_idx + 2] = 1.0 if pos['direction'] == 'long' else -1.0
            market_portfolio_vec[current_idx + 3] = (pos['strike_price'] - atm_price) / (5 * self.strike_distance)
            # <<< FIX: Add a safety check and print for debugging
            entry_step = pos.get('entry_step', self.current_step) # Default to current_step if missing
            days_held_val = (self.current_step - entry_step) / self.total_steps
            if not np.isfinite(days_held_val):
                print(f"DEBUG: NaN detected in days_held calculation!")
                print(f"  current_step: {self.current_step}, entry_step: {entry_step}, total_steps: {self.total_steps}")
                days_held_val = 0.0 # Fallback to a safe value
        garch_vol = self.garch_implied_vol if hasattr(self, 'garch_implied_vol') else 0
        market_portfolio_vec[3] = math.tanh((current_realized_vol / garch_vol) - 1.0) if garch_vol > 0 else 0.0
        if self.current_step > 0: log_return = math.log(self.current_price / self.price_path[self.current_step - 1])
        else: log_return = 0.0
        market_portfolio_vec[4] = np.clip(log_return, -0.1, 0.1) * 10
        current_idx = 5
        atm_price = 0
        if np.isfinite(self.current_price) and np.isfinite(self.strike_distance) and self.strike_distance > 0:
            atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        for i, pos in self.portfolio.iterrows():
            if i >= self.max_positions: break
            market_portfolio_vec[current_idx + 0] = 1.0
            market_portfolio_vec[current_idx + 1] = 1.0 if pos['type'] == 'call' else -1.0
            market_portfolio_vec[current_idx + 2] = 1.0 if pos['direction'] == 'long' else -1.0
            market_portfolio_vec[current_idx + 3] = (pos['strike_price'] - atm_price) / (5 * self.strike_distance)
            # <<< FIX: Add a safety check and print for debugging
            entry_step = pos.get('entry_step', self.current_step) # Default to current_step if missing
            days_held_val = (self.current_step - entry_step) / self.total_steps
            if not np.isfinite(days_held_val):
                print(f"DEBUG: NaN detected in days_held calculation!")
                print(f"  current_step: {self.current_step}, entry_step: {entry_step}, total_steps: {self.total_steps}")
                days_held_val = 0.0 # Fallback to a safe value
            market_portfolio_vec[current_idx + 4] = days_held_val
            _, signed_delta, d2 = self._get_option_details(self.current_price, pos['strike_price'], pos['days_to_expiry'], pos['type'])
            if pos['type'] == 'call': pop = _numba_cdf(d2) if pos['direction'] == 'long' else 1 - _numba_cdf(d2)
            else: pop = 1 - _numba_cdf(d2) if pos['direction'] == 'long' else _numba_cdf(d2)
            market_portfolio_vec[current_idx + 5] = pop
            max_profit = pos['strategy_max_profit']
            max_loss = pos['strategy_max_loss']
            market_portfolio_vec[current_idx + 6] = math.tanh(max_profit / self.initial_cash)
            market_portfolio_vec[current_idx + 7] = math.tanh(max_loss / self.initial_cash)
            market_portfolio_vec[current_idx + 8] = signed_delta
            current_idx += 9
        true_action_mask = self._get_true_action_mask()
        final_obs_vec = np.concatenate((market_portfolio_vec, true_action_mask.astype(np.float32)))
        return final_obs_vec

    def render(self, mode: str = 'human') -> None:
        total_pnl = self._get_total_pnl()
        print(f"Step: {self.current_step:04d} | Day: {self.current_step // self.steps_per_day + 1:02d} | Price: ${self.current_price:9.2f} | Positions: {len(self.portfolio):1d} | Total PnL: ${total_pnl:9.2f}")
        if not self.portfolio.empty:
            print(self.portfolio.to_string())

    @property
    def legal_actions(self): return self._get_true_action_mask()
    @property
    def observation_space(self) -> spaces.Space: return self._observation_space
    @property
    def action_space(self) -> spaces.Space: return self._action_space
    @property
    def reward_space(self) -> spaces.Space: return self._reward_range
    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> list:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]
    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> list:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(evaluator_env_num)]
    def __repr__(self): return "LightZero Options-Zero-Game Env."
