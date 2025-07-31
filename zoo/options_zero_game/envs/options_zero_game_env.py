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
            'LONG_CALL': 0, 'SHORT_CALL': 1, 'LONG_PUT': 2, 'SHORT_PUT': 3,
            'LONG_STRADDLE': 4, 'SHORT_STRADDLE': 5, 'LONG_STRANGLE': 6, 'SHORT_STRANGLE': 7,
            'LONG_IRON_FLY': 8, 'SHORT_IRON_FLY': 9, 'LONG_IRON_CONDOR': 10, 'SHORT_IRON_CONDOR': 11,
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

        # --- Closing Actions ---
        for j in range(self.max_positions):
            actions[f'CLOSE_POSITION_{j}'] = i; i+=1
        actions['CLOSE_ALL'] = i
        return actions

    def seed(self, seed: int, dynamic_seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _generate_historical_price_path(self) -> None:
        """
        Loads a random ticker from the cache and samples a random segment of its
        price history to serve as the price_path for the episode.
        """
        if not self._available_tickers:
            raise RuntimeError("Historical mode selected but no tickers are available.")

        # 1. Select a random ticker and load its data
        selected_ticker = random.choice(self._available_tickers)
        file_path = os.path.join(self.historical_data_path, f"{selected_ticker}.csv")
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        # 2. Ensure data is long enough for an episode
        if len(data) <= self.total_steps:
            raise RuntimeError("Historical mode selected but data is less than total_steps.")
            # Fallback to GARCH if this specific file is too short
            print(f"Warning: {selected_ticker} data is too short. Falling back to GARCH for this episode.")
            self._generate_price_path() # This calls the original synthetic generator
            return

        # 3. Sample a random continuous segment for the episode
        start_index = random.randint(0, len(data) - self.total_steps - 1)
        end_index = start_index + self.total_steps + 1
        price_segment = data['Close'].iloc[start_index:end_index]
        
        # 4. Set the price path and episode parameters
        self.price_path = price_segment.to_numpy(dtype=np.float32)
        
        # --- CRITICAL: Update environment state to match the historical data ---
        self.start_price = self.price_path[0]
        self.current_regime_name = f"Historical: {selected_ticker}"
        
        # Emulate GARCH parameters based on the sampled historical data
        # for consistency in the observation space.
        log_returns = np.log(self.price_path[1:] / self.price_path[:-1])
        
        # Calculate annualized volatility for the segment
        annualized_vol = np.std(log_returns) * np.sqrt(252 * self.steps_per_day)
        self.garch_implied_vol = annualized_vol if np.isfinite(annualized_vol) else 0.2

        # Calculate drift/trend for the segment
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

    def _handle_open_action(self, action_name: str) -> None:
        if 'IRON_CONDOR' in action_name: self._open_iron_condor(action_name)
        elif 'IRON_FLY' in action_name: self._open_iron_fly(action_name)
        elif 'VERTICAL' in action_name: self._open_vertical_spread(action_name)
        elif 'STRANGLE' in action_name: self._open_strangle(action_name)
        elif 'STRADDLE' in action_name: self._open_straddle(action_name)
        else: self._open_single_leg(action_name)
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
    def _calculate_strategy_pnl(self, legs: List[Dict], strategy_name: str, width1: float = 0, width2: float = 0) -> Dict:
        if not legs: return {}
        net_premium = sum(leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1) for leg in legs)
        
        if strategy_name in ['LONG_CALL', 'SHORT_CALL', 'LONG_PUT', 'SHORT_PUT']:
            leg = legs[0]
            entry_premium_total = leg['entry_premium'] * self.lot_size
            if leg['direction'] == 'long':
                max_loss = -entry_premium_total
                max_profit = (leg['strike_price'] * self.lot_size) - entry_premium_total if leg['type'] == 'put' else self.undefined_risk_cap
            else: # short
                max_profit = entry_premium_total
                max_loss = -((leg['strike_price'] * self.lot_size) - entry_premium_total) if leg['type'] == 'put' else -self.undefined_risk_cap
        elif 'STRADDLE' in strategy_name or 'STRANGLE' in strategy_name:
            direction = legs[0]['direction']
            max_profit = self.undefined_risk_cap if direction == 'long' else net_premium * self.lot_size
            max_loss = -net_premium * self.lot_size if direction == 'long' else -self.undefined_risk_cap
        elif 'IRON_FLY' in strategy_name or 'IRON_CONDOR' in strategy_name:
            direction = legs[0]['direction']
            max_width = max(width1, width2) * self.lot_size
            if direction == 'short':
                max_profit = net_premium * self.lot_size
                max_loss = -max_width + max_profit
            else: # long
                max_loss = -net_premium * self.lot_size
                max_profit = max_width + max_loss
        else:
            max_profit, max_loss = 0.0, 0.0
            
        return {'max_profit': max_profit, 'max_loss': max_loss, 'strategy_id': self.strategy_name_to_id.get(strategy_name, -1)}
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
        if len(self.portfolio) > self.max_positions - 2: return
        # --- 1. Parse action name to get the width ---
        parts = action_name.split('_')
        width_in_strikes = int(parts[-1])
        width_in_price = width_in_strikes * self.strike_distance
        
        # --- 2. Validation (already implemented, now more important) ---
        max_offset = max(int(k) for k in self._cfg.iv_skew_table['call'].keys())
        min_offset = min(int(k) for k in self._cfg.iv_skew_table['call'].keys())

        call_offset = +width_in_strikes
        put_offset = -width_in_strikes
        
        if not (min_offset <= call_offset <= max_offset and min_offset <= put_offset <= max_offset):
            return

        # --- 3. Define the legs of the strangle ---
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        direction = 'long' if 'LONG' in action_name else 'short'
        
        legs_to_trade = [
            {'type': 'call', 'direction': direction, 'strike_price': atm_price + width_in_price},
            {'type': 'put',  'direction': direction, 'strike_price': atm_price - width_in_price},
        ]

        # --- 4. Get DTE, price the legs, and prepare for execution ---
        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)

        for leg in legs_to_trade:
            leg['entry_step'] = self.current_step
            leg['days_to_expiry'] = days_to_expiry
            mid_price, _, _ = self._get_option_details(self.current_price, leg['strike_price'], leg['days_to_expiry'], leg['type'])
            leg['entry_premium'] = self._get_option_price(mid_price, is_buy=(leg['direction'] == 'long'))

        # --- 5. Calculate strategy PnL and execute ---
        strategy_name = f"{direction.upper()}_STRANGLE"
        pnl = self._calculate_strategy_pnl(legs_to_trade, strategy_name)
        self._execute_trades(legs_to_trade, pnl)

    def _open_iron_fly(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 4: return
        legs = self._get_trade_legs(action_name)
        if not legs: return
        for leg in legs:
            leg['entry_premium'] = self._get_option_price(self._get_option_details(self.current_price, leg['strike_price'], leg['days_to_expiry'], leg['type'])[0], leg['direction'] == 'long')
        direction = 'LONG' if 'LONG' in action_name else 'SHORT'
        strategy_name = f"{direction}_IRON_FLY"
        width = abs(legs[2]['strike_price'] - legs[0]['strike_price'])
        pnl = self._calculate_strategy_pnl(legs, strategy_name, width, width)
        self._execute_trades(legs, pnl)
    def _open_iron_condor(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 4: return
        legs = self._get_trade_legs(action_name)
        if not legs: return
        for leg in legs:
            leg['entry_premium'] = self._get_option_price(self._get_option_details(self.current_price, leg['strike_price'], leg['days_to_expiry'], leg['type'])[0], leg['direction'] == 'long')
        direction = 'LONG' if 'LONG' in action_name else 'SHORT'
        strategy_name = f"{direction}_IRON_CONDOR"
        call_width = abs(legs[2]['strike_price'] - legs[0]['strike_price'])
        put_width = abs(legs[3]['strike_price'] - legs[1]['strike_price'])
        pnl = self._calculate_strategy_pnl(legs, strategy_name, call_width, put_width)
        self._execute_trades(legs, pnl)

    def _open_vertical_spread(self, action_name: str) -> None:
        """Opens a two-leg vertical spread, with validation for strike limits."""
        if len(self.portfolio) > self.max_positions - 2:
            return

        # --- 1. Parse the action name ---
        parts = action_name.split('_')
        direction = parts[1]
        option_type = parts[3].lower()
        width_in_strikes = int(parts[4]) # This is the width in strike units (e.g., 1 or 2)
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
            
        # Calculate the offsets of both strikes from ATM.
        # Note: atm_price is the center, so strike1's offset is always 0.
        offset1 = round((strike1 - atm_price) / self.strike_distance)
        offset2 = round((strike2 - atm_price) / self.strike_distance)
        
        # Check if either leg is outside our defined limits (e.g., +/- 5 strikes)
        # We get the limit from the iv_skew_table keys.
        max_offset = max(int(k) for k in self._cfg.iv_skew_table['call'].keys())
        min_offset = min(int(k) for k in self._cfg.iv_skew_table['call'].keys())

        if not (min_offset <= offset1 <= max_offset and min_offset <= offset2 <= max_offset):
            # This action would create a leg outside the defined strike range.
            # Treat it as an invalid action and do nothing.
            # We'll also mask this in _get_true_action_mask.
            # print(f"DEBUG: Invalid spread action '{action_name}'. Offset {offset2} is out of bounds [{min_offset}, {max_offset}].")
            return

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
            max_profit = (width * self.lot_size) + max_loss
        else: # Credit spread (received credit)
            max_profit = -net_premium * self.lot_size
            max_loss = -(width * self.lot_size) + max_profit

        strategy_pnl = {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'strategy_id': self.strategy_name_to_id.get(action_name, -1) # Use a unique ID later if needed
        }
        
        self._execute_trades(legs_to_trade, strategy_pnl)

    def _get_true_action_mask(self) -> np.ndarray:
        action_mask = np.zeros(self.action_space_size, dtype=np.int8)
        current_trading_day = self.current_step // self.steps_per_day
        is_expiry_day = current_trading_day >= self.time_to_expiry_days - 1
        action_mask[self.actions_to_indices['HOLD']] = 1
        if not self.portfolio.empty:
            action_mask[self.actions_to_indices['CLOSE_ALL']] = 1
            for i in range(len(self.portfolio)):
                if f'CLOSE_POSITION_{i}' in self.actions_to_indices:
                    action_mask[self.actions_to_indices[f'CLOSE_POSITION_{i}']] = 1
        if is_expiry_day:
            return action_mask

        # --- Get common parameters for validation ---
        available_slots = self.max_positions - len(self.portfolio)
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        max_offset = max(int(k) for k in self._cfg.iv_skew_table['call'].keys())
        min_offset = min(int(k) for k in self._cfg.iv_skew_table['call'].keys())

        # --- Multi-leg strategy rules based on available slots ---
        available_slots = self.max_positions - len(self.portfolio)

        if available_slots >= 4:
            action_mask[self.actions_to_indices['OPEN_LONG_IRON_FLY']] = 1
            action_mask[self.actions_to_indices['OPEN_SHORT_IRON_FLY']] = 1
            action_mask[self.actions_to_indices['OPEN_LONG_IRON_CONDOR']] = 1
            action_mask[self.actions_to_indices['OPEN_SHORT_IRON_CONDOR']] = 1

        if available_slots >= 2:
            # --- Straddle logic (always valid as offset is 0) ---
            action_mask[self.actions_to_indices['OPEN_LONG_STRADDLE_ATM']] = 1
            action_mask[self.actions_to_indices['OPEN_SHORT_STRADDLE_ATM']] = 1

            # --- DYNAMIC STRANGLE MASKING ---
            for width in [1, 2]:
                call_offset = +width
                put_offset = -width

                # Check if this strangle width is legal
                if (min_offset <= call_offset <= max_offset and min_offset <= put_offset <= max_offset):
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
                    is_legal_spread = (min_offset <= offset <= max_offset)

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
