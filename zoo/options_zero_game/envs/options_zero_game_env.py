import copy
import math
import random
import logging
from typing import Tuple, Dict, Any, List

import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
from arch import arch_model
from easydict import EasyDict
from gym.utils import seeding

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
    if T <= 1e-6:
        if is_call: return max(0.0, S - K)
        else: return max(0.0, K - S)
    if sigma <= 1e-6:
        if is_call: return max(0.0, S - K)
        else: return max(0.0, K - S)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if is_call:
        price = S * _numba_cdf(d1) - K * math.exp(-r * T) * _numba_cdf(d2)
    else:
        price = K * math.exp(-r * T) * _numba_cdf(-d2) - S * _numba_cdf(-d1)
    return price

@jit(nopython=True)
def _numba_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T <= 1e-6 or sigma <= 1e-6:
        if is_call: return 1.0 if S > K else 0.0
        else: return -1.0 if S < K else 0.0
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    if is_call:
        return _numba_cdf(d1)
    else:
        return _numba_cdf(d1) - 1.0

@ENV_REGISTRY.register('options_zero_game')
class OptionsZeroGameEnv(gym.Env):
    """
    Options-Zero-Game: A Reinforcement Learning Environment for Options Trading.

    This environment simulates a realistic options trading scenario where an RL agent
    can learn to manage a portfolio of options under a variety of market conditions
    and a strict set of professional trading rules.
    """
    metadata = {'render.modes': ['human']}

    config = dict(
        start_price=20000.0,
        initial_cash=100000.0,
        market_regimes = [
            {'name': 'Developed_Market', 'mu': 0.00005, 'omega': 0.000005, 'alpha': 0.09, 'beta': 0.90},
        ],
        time_to_expiry_days=30,
        steps_per_day=75,
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
        TRADING_DAYS_IN_WEEK=5,
        TOTAL_DAYS_IN_WEEK=7,
        UNDEFINED_RISK_CAP_MULTIPLIER=10,
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

        self.start_price: float = self._cfg.start_price
        self.initial_cash: float = self._cfg.initial_cash
        self.market_regimes: List[Dict[str, Any]] = self._cfg.market_regimes
        self.rolling_vol_window: int = self._cfg.rolling_vol_window
        self.time_to_expiry_days: int = self._cfg.time_to_expiry_days
        self.steps_per_day: int = self._cfg.steps_per_day
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
        self.trading_days_in_week: int = self._cfg.TRADING_DAYS_IN_WEEK
        self.total_days_in_week: int = self._cfg.TOTAL_DAYS_IN_WEEK
        self.undefined_risk_cap: float = self.initial_cash * self._cfg.UNDEFINED_RISK_CAP_MULTIPLIER
        
        self.iv_bins: Dict[str, Dict[str, np.ndarray]] = self._discretize_iv_skew(self._cfg.iv_skew_table)
        
        self.actions_to_indices: Dict[str, int] = self._build_action_space()
        self.indices_to_actions: Dict[int, str] = {v: k for k, v in self.actions_to_indices.items()}
        self.action_space_size: int = len(self.actions_to_indices)

        self._action_space: spaces.Discrete = spaces.Discrete(self.action_space_size)
        self._observation_space: spaces.Dict = self._create_observation_space()
        self._reward_range: spaces.Box = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.np_random: Any = None
        
        self.portfolio_columns: List[str] = ['type', 'direction', 'entry_step', 'strike_price', 'entry_premium', 'days_to_expiry', 'creation_id', 'strategy_id', 'strategy_max_profit', 'strategy_max_loss']
        self.portfolio: pd.DataFrame = pd.DataFrame(columns=self.portfolio_columns)

    def _discretize_iv_skew(self, skew_table: Dict[str, Dict[str, Tuple[float, float]]], num_bins: int = 5) -> Dict[str, Dict[str, np.ndarray]]:
        binned_ivs = {'call': {}, 'put': {}}
        for option_type, table in skew_table.items():
            for offset_str, iv_range in table.items():
                min_iv, max_iv = iv_range
                binned_ivs[option_type][offset_str] = np.linspace(min_iv, max_iv, num_bins, dtype=np.float32) / 100.0
        return binned_ivs

    def _build_action_space(self) -> Dict[str, int]:
        actions = {'HOLD': 0}; i = 1
        for offset in range(-3, 4):
            sign = '+' if offset >= 0 else ''
            actions[f'OPEN_LONG_CALL_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_SHORT_CALL_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_LONG_PUT_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_SHORT_PUT_ATM{sign}{offset}'] = i; i+=1
        actions['OPEN_LONG_STRADDLE_ATM'] = i; i+=1
        actions['OPEN_SHORT_STRADDLE_ATM'] = i; i+=1
        actions['OPEN_LONG_STRANGLE_ATM_1'] = i; i+=1
        actions['OPEN_SHORT_STRANGLE_ATM_1'] = i; i+=1
        actions['OPEN_LONG_IRON_FLY'] = i; i+=1
        actions['OPEN_SHORT_IRON_FLY'] = i; i+=1
        actions['OPEN_LONG_IRON_CONDOR'] = i; i+=1
        actions['OPEN_SHORT_IRON_CONDOR'] = i; i+=1
        for j in range(self.max_positions):
            actions[f'CLOSE_POSITION_{j}'] = i; i+=1
        actions['CLOSE_ALL'] = i
        return actions

    def _create_observation_space(self) -> spaces.Dict:
        self.market_and_portfolio_state_size = 5 + self.max_positions * 9
        self.obs_vector_size = self.market_and_portfolio_state_size + self.action_space_size
        return spaces.Dict({'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_vector_size,), dtype=np.float32),'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space_size,), dtype=np.int8),'to_play': spaces.Box(low=-1, high=-1, shape=(1,), dtype=np.int8)})

    def seed(self, seed: int, dynamic_seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _generate_price_path(self) -> None:
        """
        Generates a realistic, GARCH(1,1)-based price path for an entire episode.

        This function scales annualized, daily GARCH parameters to an intra-day,
        per-step level. It then uses the arch library to simulate a path of log returns,
        which is efficiently converted to a price path using vectorized NumPy operations.
        """
        trading_days_per_year = 252
        steps_per_year = trading_days_per_year * self.steps_per_day
        
        # Scale daily parameters to the per-step frequency for realism
        mu_step = self.trend / steps_per_year
        omega_step = self.omega / steps_per_year
        alpha_step = self.alpha / steps_per_year # Use the safer, scaled alpha
        beta_step = self.beta # Beta is a persistence coefficient and is not scaled
            
        garch_spec = arch_model(None, mean='Constant', vol='GARCH', p=1, q=1, dist='normal')
        params = np.array([mu_step, omega_step, alpha_step, beta_step], dtype=np.float32)
        
        # Simulate the returns for the entire episode
        sim_result = garch_spec.simulate(params, nobs=self.total_steps, random_state=self.np_random)
        simulated_returns = sim_result.data.to_numpy(dtype=np.float32)
        
        # <<< YOUR SUPERIOR VECTORIZED LOGIC >>>
        # Build the price path efficiently from the simulated returns
        cumulative_returns = np.cumsum(np.concatenate([np.array([0.0], dtype=np.float32), simulated_returns]))
        price_path = self.start_price * np.exp(cumulative_returns)
        self.price_path = price_path.astype(np.float32)
        
        # Compute the realized volatility series
        log_returns = np.diff(np.log(self.price_path))
        returns_series = pd.Series(log_returns, dtype=np.float32)
        rolling_window_steps = self.rolling_vol_window * self.steps_per_day
        self.realized_vol_series = returns_series.rolling(window=rolling_window_steps).std().fillna(0) * np.sqrt(steps_per_year)

    def _simulate_price_step(self) -> None:
        self.current_price = self.price_path[self.current_step]
    
    def _get_implied_volatility(self, offset: int, option_type: str) -> float:
        clamped_offset = max(-5, min(5, offset))
        return self.iv_bins[option_type][str(clamped_offset)][self.iv_bin_index]

    def _get_option_details(self, underlying_price: float, strike_price: float, days_to_expiry: float, option_type: str) -> Tuple[float, float, float]:
        t = days_to_expiry / 365.25
        is_call = option_type == 'call'
        
        if t < 1e-6:
            intrinsic = max(0.0, underlying_price - strike_price) if is_call else max(0.0, strike_price - underlying_price)
            delta_val = 1.0 if intrinsic > 0 else 0.0
            return intrinsic, delta_val, 0.0

        atm_price = int(underlying_price / self.strike_distance + 0.5) * self.strike_distance
        offset = round((strike_price - atm_price) / self.strike_distance)
        vol = self._get_implied_volatility(offset, option_type)

        price = _numba_black_scholes(underlying_price, strike_price, t, self.risk_free_rate, vol, is_call)
        signed_delta = _numba_delta(underlying_price, strike_price, t, self.risk_free_rate, vol, is_call)
        
        d2 = 0.0
        if vol > 1e-6:
            d1 = (math.log(underlying_price / strike_price) + (self.risk_free_rate + 0.5 * vol ** 2) * t) / (vol * math.sqrt(t))
            d2 = d1 - vol * math.sqrt(t)
            
        return price, signed_delta, d2

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

    def reset(self, seed: int = None, **kwargs) -> Dict[str, np.ndarray]:
        if seed is not None: self.seed(seed)
        elif self.np_random is None: self.seed(0)

        chosen_regime = random.choice(self.market_regimes)
        self.current_regime_name = chosen_regime['name']
        self.trend: float = chosen_regime['mu']
        self.omega: float = chosen_regime['omega']
        self.alpha: float = chosen_regime['alpha']
        self.beta: float = chosen_regime['beta']

        if self.alpha + self.beta >= 1.0:
            total = self.alpha + self.beta
            self.alpha = self.alpha / (total + 0.01)
            self.beta = self.beta / (total + 0.01)

        unconditional_variance = self.omega / (1 - self.alpha - self.beta)
        self.garch_implied_vol: float = math.sqrt(unconditional_variance * 252)
        self.iv_bin_index: int = random.randint(0, 4)
        self._generate_price_path()
        self.current_step: int = 0
        self.day_of_week: int = 0
        self.current_price: float = self.price_path[0]
        self.portfolio: pd.DataFrame = pd.DataFrame(columns=self.portfolio_columns)
        self.realized_pnl: float = 0.0
        self._final_eval_reward: float = 0.0
        self.high_water_mark: float = self.initial_cash
        self.illegal_action_count: int = 0
        self.next_creation_id: int = 0
        return self._get_observation()

    def _close_position(self, position_index: int) -> None:
        if position_index < 0:
            position_index = len(self.portfolio) + position_index
        if 0 <= position_index < len(self.portfolio):
            pos_to_close = self.portfolio.iloc[position_index]
            mid_price, _, _ = self._get_option_details(self.current_price, pos_to_close['strike_price'], pos_to_close['days_to_expiry'], pos_to_close['type'])
            exit_premium = self._get_option_price(mid_price, is_buy=(pos_to_close['direction'] == 'short'))
            entry_premium = pos_to_close['entry_premium']
            if pos_to_close['direction'] == 'long': pnl = (exit_premium - entry_premium) * self.lot_size
            else: pnl = (entry_premium - exit_premium) * self.lot_size
            self.realized_pnl += pnl
            self.portfolio = self.portfolio.drop(self.portfolio.index[position_index]).reset_index(drop=True)

    def _calculate_shaped_reward(self, equity_before: float, equity_after: float) -> Tuple[float, float]:
        raw_reward = equity_after - equity_before
        pnl_component = math.tanh(raw_reward / self.pnl_scaling_factor)
        self.high_water_mark = max(self.high_water_mark, equity_after)
        drawdown = self.high_water_mark - equity_after
        drawdown_penalty_component = -self.drawdown_penalty_weight * math.tanh(drawdown / self.initial_cash)
        combined_score = pnl_component + drawdown_penalty_component
        final_reward = math.tanh(combined_score / 2)
        return final_reward, raw_reward

    def _enforce_legal_action(self, action: int) -> int:
        true_legal_actions_mask = self._get_true_action_mask()
        if true_legal_actions_mask[action] == 1:
            return action
        else:
            return self.actions_to_indices['HOLD']

    def _advance_time_and_market(self) -> None:
        self.current_step += 1
        time_decay_days = 1.0 / self.steps_per_day
        is_end_of_day = (self.current_step % self.steps_per_day) == 0
        if is_end_of_day:
            if self.day_of_week == 4:
                self.day_of_week = 0
                time_decay_days += (self.total_days_in_week - self.trading_days_in_week)
            else:
                self.day_of_week += 1
        self._simulate_price_step()
        if not self.portfolio.empty:
            self.portfolio['days_to_expiry'] -= time_decay_days

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
            try:
                pos_index = int(action_name.split('_')[-1])
                self._close_position(pos_index)
                portfolio_changed = True
            except (ValueError, IndexError):
                logging.warning(f"Attempted to close an invalid position index from action: {action_name}")
        elif action_name == 'CLOSE_ALL':
            if not self.portfolio.empty:
                portfolio_changed = True
            while not self.portfolio.empty:
                self._close_position(-1)

        # <<< MODIFIED: Only sort the portfolio when its structure has changed
        if portfolio_changed and not self.portfolio.empty:
            self.portfolio = self.portfolio.sort_values(by=['strike_price', 'type', 'creation_id']).reset_index(drop=True)

        self._advance_time_and_market()

        done = self.current_step >= self.total_steps
        if done:
            while not self.portfolio.empty:
                self._close_position(-1)

        equity_after = self._get_current_equity()
        shaped_reward, raw_reward = self._calculate_shaped_reward(equity_before, equity_after)

        if was_illegal_action:
            final_reward = self.illegal_action_penalty
        else:
            final_reward = shaped_reward

        self._final_eval_reward += raw_reward
        obs = self._get_observation()
        info = {
            'price': self.current_price,
            'eval_episode_return': self._final_eval_reward,
            'illegal_actions_in_episode': self.illegal_action_count
        }
        return BaseEnvTimestep(obs, final_reward, done, info)

    def _handle_open_action(self, action_name: str) -> None:
        if 'IRON_CONDOR' in action_name: self._open_iron_condor(action_name)
        elif 'IRON_FLY' in action_name: self._open_iron_fly(action_name)
        elif 'STRANGLE' in action_name: self._open_strangle(action_name)
        elif 'STRADDLE' in action_name: self._open_straddle(action_name)
        else: self._open_single_leg(action_name)

    def _open_single_leg(self, action_name: str) -> None:
        if len(self.portfolio) >= self.max_positions: return
        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (self.total_days_in_week / self.trading_days_in_week)
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        _, direction, type, strike_str = action_name.split('_')
        offset = int(strike_str.replace('ATM', ''))
        strike_price = atm_price + (offset * self.strike_distance)
        is_buy = (direction == 'LONG')
        mid_price, _, _ = self._get_option_details(self.current_price, strike_price, days_to_expiry, type.lower())
        entry_premium = self._get_option_price(mid_price, is_buy)
        new_position = pd.DataFrame([{'type': type.lower(), 'direction': direction.lower(), 'entry_step': self.current_step, 'strike_price': strike_price, 'entry_premium': entry_premium, 'days_to_expiry': days_to_expiry, 'creation_id': self.next_creation_id, 'strategy_id': self.next_creation_id, 'strategy_max_profit': 0.0, 'strategy_max_loss': 0.0}])
        self.next_creation_id += 1
        self.portfolio = pd.concat([self.portfolio, new_position], ignore_index=True)

    def _open_straddle(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 2: return
        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (self.total_days_in_week / self.trading_days_in_week)
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        direction = 'long' if 'LONG' in action_name else 'short'
        is_buy = (direction == 'long')
        trades_to_execute = []
        mid_price_call, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'call')
        trades_to_execute.append({'type': 'call', 'direction': direction, 'strike_price': atm_price, 'entry_premium': self._get_option_price(mid_price_call, is_buy)})
        mid_price_put, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'put')
        trades_to_execute.append({'type': 'put', 'direction': direction, 'strike_price': atm_price, 'entry_premium': self._get_option_price(mid_price_put, is_buy)})
        
        net_premium = sum(t['entry_premium'] for t in trades_to_execute)
        max_profit = self.undefined_risk_cap if direction == 'long' else net_premium * self.lot_size
        max_loss = -net_premium * self.lot_size if direction == 'long' else -self.undefined_risk_cap
        
        strategy_id = self.next_creation_id
        for trade in trades_to_execute:
            trade['creation_id'] = self.next_creation_id
            trade['strategy_id'] = strategy_id
            trade['strategy_max_profit'] = max_profit
            trade['strategy_max_loss'] = max_loss
            self.next_creation_id += 1
            
        if trades_to_execute:
            new_positions_df = pd.DataFrame(trades_to_execute)
            self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)

    def _open_strangle(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 2: return
        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (self.total_days_in_week / self.trading_days_in_week)
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        direction = 'long' if 'LONG' in action_name else 'short'
        is_buy = (direction == 'long')
        trades_to_execute = []
        call_strike = atm_price + (1 * self.strike_distance)
        mid_price_call, _, _ = self._get_option_details(self.current_price, call_strike, days_to_expiry, 'call')
        trades_to_execute.append({'type': 'call', 'direction': direction, 'strike_price': call_strike, 'entry_premium': self._get_option_price(mid_price_call, is_buy)})
        put_strike = atm_price - (1 * self.strike_distance)
        mid_price_put, _, _ = self._get_option_details(self.current_price, put_strike, days_to_expiry, 'put')
        trades_to_execute.append({'type': 'put', 'direction': direction, 'strike_price': put_strike, 'entry_premium': self._get_option_price(mid_price_put, is_buy)})
        
        net_premium = sum(t['entry_premium'] for t in trades_to_execute)
        max_profit = self.undefined_risk_cap if direction == 'long' else net_premium * self.lot_size
        max_loss = -net_premium * self.lot_size if direction == 'long' else -self.undefined_risk_cap
        
        strategy_id = self.next_creation_id
        for trade in trades_to_execute:
            trade['creation_id'] = self.next_creation_id
            trade['strategy_id'] = strategy_id
            trade['strategy_max_profit'] = max_profit
            trade['strategy_max_loss'] = max_loss
            self.next_creation_id += 1
            
        if trades_to_execute:
            new_positions_df = pd.DataFrame(trades_to_execute)
            self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)

    def _open_iron_fly(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 4: return
        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (self.total_days_in_week / self.trading_days_in_week)
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        direction = 'long' if 'LONG' in action_name else 'short'
        trades_to_execute = []
        mid_price_call_atm, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'call')
        premium_call_atm = self._get_option_price(mid_price_call_atm, is_buy=(direction == 'long'))
        mid_price_put_atm, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'put')
        premium_put_atm = self._get_option_price(mid_price_put_atm, is_buy=(direction == 'long'))
        net_premium = premium_call_atm + premium_put_atm
        upper_breakeven = atm_price + net_premium
        lower_breakeven = atm_price - net_premium
        hedge_strike_call = int(upper_breakeven / self.strike_distance + 0.5) * self.strike_distance
        hedge_strike_put = int(lower_breakeven / self.strike_distance + 0.5) * self.strike_distance
        straddle_dir = 'long' if direction == 'long' else 'short'
        hedge_dir = 'short' if direction == 'long' else 'long'
        trades_to_execute.append({'type': 'call', 'direction': straddle_dir, 'strike_price': atm_price, 'entry_premium': premium_call_atm})
        trades_to_execute.append({'type': 'put', 'direction': straddle_dir, 'strike_price': atm_price, 'entry_premium': premium_put_atm})
        mid_price_hedge_call, _, _ = self._get_option_details(self.current_price, hedge_strike_call, days_to_expiry, 'call')
        trades_to_execute.append({'type': 'call', 'direction': hedge_dir, 'strike_price': hedge_strike_call, 'entry_premium': self._get_option_price(mid_price_hedge_call, hedge_dir == 'long')})
        mid_price_hedge_put, _, _ = self._get_option_details(self.current_price, hedge_strike_put, days_to_expiry, 'put')
        trades_to_execute.append({'type': 'put', 'direction': hedge_dir, 'strike_price': hedge_strike_put, 'entry_premium': self._get_option_price(mid_price_hedge_put, hedge_dir == 'long')})
        
        net_premium_straddle = trades_to_execute[0]['entry_premium'] + trades_to_execute[1]['entry_premium']
        net_premium_hedge = trades_to_execute[2]['entry_premium'] + trades_to_execute[3]['entry_premium']
        net_premium = net_premium_straddle - net_premium_hedge
        
        if direction == 'short':
            max_profit = net_premium * self.lot_size
            max_loss = -((hedge_strike_call - atm_price) * self.lot_size - max_profit)
        else:
            max_loss = -net_premium * self.lot_size
            max_profit = ((hedge_strike_call - atm_price) * self.lot_size) + max_loss

        strategy_id = self.next_creation_id
        for trade in trades_to_execute:
            trade['creation_id'] = self.next_creation_id
            trade['strategy_id'] = strategy_id
            trade['strategy_max_profit'] = max_profit
            trade['strategy_max_loss'] = max_loss
            self.next_creation_id += 1
            
        if trades_to_execute:
            new_positions_df = pd.DataFrame(trades_to_execute)
            self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)

    def _open_iron_condor(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 4: return
        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (self.total_days_in_week / self.trading_days_in_week)
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        direction = 'long' if 'LONG' in action_name else 'short'
        target_delta_short = 0.30
        target_delta_long = 0.10
        
        best_short_call_strike, best_long_call_strike = atm_price, atm_price
        min_delta_diff_short_call, min_delta_diff_long_call = 999, 999
        for offset in range(1, 15):
            strike = atm_price + (offset * self.strike_distance)
            _, d, _ = self._get_option_details(self.current_price, strike, days_to_expiry, 'call')
            if d == 0: break
            if abs(d - target_delta_short) < min_delta_diff_short_call:
                min_delta_diff_short_call = abs(d - target_delta_short)
                best_short_call_strike = strike
            if abs(d - target_delta_long) < min_delta_diff_long_call:
                min_delta_diff_long_call = abs(d - target_delta_long)
                best_long_call_strike = strike
        
        best_short_put_strike, best_long_put_strike = atm_price, atm_price
        min_delta_diff_short_put, min_delta_diff_long_put = 999, 999
        max_put_offset = int((atm_price / self.strike_distance) - 1)
        for offset in range(1, max_put_offset):
            strike = atm_price - (offset * self.strike_distance)
            _, d, _ = self._get_option_details(self.current_price, strike, days_to_expiry, 'put')
            if d == 0: break
            if abs(d - target_delta_short) < min_delta_diff_short_put:
                min_delta_diff_short_put = abs(d - target_delta_short)
                best_short_put_strike = strike
            if abs(d - target_delta_long) < min_delta_diff_long_put:
                min_delta_diff_long_put = abs(d - target_delta_long)
                best_long_put_strike = strike
        
        min_strike = self.strike_distance
        max_strike = self.start_price * 3
        best_short_call_strike = max(min_strike, min(max_strike, best_short_call_strike))
        best_long_call_strike = max(min_strike, min(max_strike, best_long_call_strike))
        best_short_put_strike = max(min_strike, min(max_strike, best_short_put_strike))
        best_long_put_strike = max(min_strike, min(max_strike, best_long_put_strike))

        if best_long_call_strike <= best_short_call_strike:
            best_long_call_strike = best_short_call_strike + self.strike_distance
        if best_long_put_strike >= best_short_put_strike:
            best_long_put_strike = best_short_put_strike - self.strike_distance
        if not (best_long_put_strike < best_short_put_strike < best_short_call_strike < best_long_call_strike):
            return

        trades_to_execute = []
        inner_dir = 'long' if direction == 'long' else 'short'
        outer_dir = 'short' if direction == 'long' else 'long'
        mid_price_call_inner, _, _ = self._get_option_details(self.current_price, best_short_call_strike, days_to_expiry, 'call')
        trades_to_execute.append({'type': 'call', 'direction': inner_dir, 'strike_price': best_short_call_strike, 'entry_premium': self._get_option_price(mid_price_call_inner, inner_dir == 'long')})
        mid_price_put_inner, _, _ = self._get_option_details(self.current_price, best_short_put_strike, days_to_expiry, 'put')
        trades_to_execute.append({'type': 'put', 'direction': inner_dir, 'strike_price': best_short_put_strike, 'entry_premium': self._get_option_price(mid_price_put_inner, inner_dir == 'long')})
        mid_price_call_outer, _, _ = self._get_option_details(self.current_price, best_long_call_strike, days_to_expiry, 'call')
        trades_to_execute.append({'type': 'call', 'direction': outer_dir, 'strike_price': best_long_call_strike, 'entry_premium': self._get_option_price(mid_price_call_outer, outer_dir == 'long')})
        mid_price_put_outer, _, _ = self._get_option_details(self.current_price, best_long_put_strike, days_to_expiry, 'put')
        trades_to_execute.append({'type': 'put', 'direction': outer_dir, 'strike_price': best_long_put_strike, 'entry_premium': self._get_option_price(mid_price_put_outer, outer_dir == 'long')})
        
        net_premium = sum(t['entry_premium'] * (1 if t['direction'] == inner_dir else -1) for t in trades_to_execute)
        
        call_spread_width = best_long_call_strike - best_short_call_strike
        put_spread_width = best_short_put_strike - best_long_put_strike
        
        if direction == 'short':
            max_profit = net_premium * self.lot_size
            max_loss = - (max(call_spread_width, put_spread_width) * self.lot_size - max_profit)
        else:
            max_loss = -net_premium * self.lot_size
            max_profit = (max(call_spread_width, put_spread_width) * self.lot_size) + max_loss

        strategy_id = self.next_creation_id
        for trade in trades_to_execute:
            trade['creation_id'] = self.next_creation_id
            trade['strategy_id'] = strategy_id
            trade['strategy_max_profit'] = max_profit
            trade['strategy_max_loss'] = max_loss
            self.next_creation_id += 1
            
        if trades_to_execute:
            new_positions_df = pd.DataFrame(trades_to_execute)
            self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)

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
        
        if len(self.portfolio) <= self.max_positions - 4:
            action_mask[self.actions_to_indices['OPEN_LONG_IRON_FLY']] = 1
            action_mask[self.actions_to_indices['OPEN_SHORT_IRON_FLY']] = 1
            action_mask[self.actions_to_indices['OPEN_LONG_IRON_CONDOR']] = 1
            action_mask[self.actions_to_indices['OPEN_SHORT_IRON_CONDOR']] = 1
        
        if len(self.portfolio) <= self.max_positions - 2:
            action_mask[self.actions_to_indices['OPEN_LONG_STRADDLE_ATM']] = 1
            action_mask[self.actions_to_indices['OPEN_SHORT_STRADDLE_ATM']] = 1
            action_mask[self.actions_to_indices['OPEN_LONG_STRANGLE_ATM_1']] = 1
            action_mask[self.actions_to_indices['OPEN_SHORT_STRANGLE_ATM_1']] = 1
        
        if len(self.portfolio) < self.max_positions:
            atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
            trading_days_left = self.time_to_expiry_days - current_trading_day
            days_to_expiry = trading_days_left * (self.total_days_in_week / self.trading_days_in_week)
            
            num_long_calls = len(self.portfolio.query("type == 'call' and direction == 'long'"))
            num_short_calls = len(self.portfolio.query("type == 'call' and direction == 'short'"))
            num_long_puts = len(self.portfolio.query("type == 'put' and direction == 'long'"))
            num_short_puts = len(self.portfolio.query("type == 'put' and direction == 'short'"))
            existing_positions = {(p['strike_price'], p['type']): p['direction'] for i, p in self.portfolio.iterrows()}
            
            for offset in range(-3, 4):
                strike_price = atm_price + (offset * self.strike_distance)
                for option_type in ['call', 'put']:
                    _, delta, _ = self._get_option_details(self.current_price, strike_price, days_to_expiry, option_type)
                    abs_delta = abs(delta)
                    is_far_otm = abs_delta < self.otm_threshold
                    is_far_itm = abs_delta > self.itm_threshold
                    
                    action_name_long = f'OPEN_LONG_{option_type.upper()}_ATM{"+" if offset >=0 else ""}{offset}'
                    is_legal_long = True
                    is_legal_long &= not (is_far_otm or is_far_itm)
                    if option_type == 'put': is_legal_long &= not existing_positions.get((strike_price, 'call')) == 'short'
                    else: is_legal_long &= not existing_positions.get((strike_price, 'put')) == 'short'
                    if option_type == 'call': is_legal_long &= num_long_calls == 0
                    else: is_legal_long &= num_long_puts == 0
                    if is_legal_long and action_name_long in self.actions_to_indices:
                        action_mask[self.actions_to_indices[action_name_long]] = 1
                    
                    action_name_short = f'OPEN_SHORT_{option_type.upper()}_ATM{"+" if offset >=0 else ""}{offset}'
                    is_legal_short = True
                    is_legal_short &= not is_far_itm
                    if option_type == 'put': is_legal_short &= not existing_positions.get((strike_price, 'call')) == 'long'
                    else: is_legal_short &= not existing_positions.get((strike_price, 'put')) == 'long'
                    if option_type == 'call': is_legal_short &= num_short_calls == 0
                    else: is_legal_short &= num_short_puts == 0
                    if is_legal_short and action_name_short in self.actions_to_indices:
                        action_mask[self.actions_to_indices[action_name_short]] = 1
        
        return action_mask

    def _get_observation(self) -> Dict[str, np.ndarray]:
        market_portfolio_vec = np.zeros(self.market_and_portfolio_state_size, dtype=np.float32)
        market_portfolio_vec[0] = (self.current_price / self.start_price) - 1.0
        market_portfolio_vec[1] = (self.total_steps - self.current_step) / self.total_steps
        total_pnl = self._get_total_pnl()
        market_portfolio_vec[2] = math.tanh(total_pnl / self.initial_cash)
        current_realized_vol = self.realized_vol_series.iloc[self.current_step - 1] if self.current_step > 0 else 0
        market_portfolio_vec[3] = math.tanh((current_realized_vol / self.garch_implied_vol) - 1.0) if self.garch_implied_vol > 0 else 0.0
        if self.current_step > 0: log_return = math.log(self.current_price / self.price_path[self.current_step - 1])
        else: log_return = 0.0
        market_portfolio_vec[4] = np.clip(log_return, -0.1, 0.1) * 10
        current_idx = 5
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        for i, pos in self.portfolio.iterrows():
            if i >= self.max_positions: break
            market_portfolio_vec[current_idx + 0] = 1.0
            market_portfolio_vec[current_idx + 1] = 1.0 if pos['type'] == 'call' else -1.0
            market_portfolio_vec[current_idx + 2] = 1.0 if pos['direction'] == 'long' else -1.0
            market_portfolio_vec[current_idx + 3] = (pos['strike_price'] - atm_price) / (3 * self.strike_distance)
            market_portfolio_vec[current_idx + 4] = (self.current_step - pos['entry_step']) / self.total_steps
            
            _, signed_delta, d2 = self._get_option_details(self.current_price, pos['strike_price'], pos['days_to_expiry'], pos['type'])
            
            if pos['type'] == 'call': pop = _numba_cdf(d2) if pos['direction'] == 'long' else 1 - _numba_cdf(d2)
            else: pop = 1 - _numba_cdf(d2) if pos['direction'] == 'long' else _numba_cdf(d2)
            market_portfolio_vec[current_idx + 5] = pop
            
            max_profit, max_loss = self._calculate_max_profit_loss(pos)
            market_portfolio_vec[current_idx + 6] = math.tanh(max_profit / self.initial_cash)
            market_portfolio_vec[current_idx + 7] = math.tanh(max_loss / self.initial_cash)
            market_portfolio_vec[current_idx + 8] = signed_delta
            current_idx += 9
            
        true_action_mask = self._get_true_action_mask()
        final_obs_vec = np.concatenate((market_portfolio_vec, true_action_mask.astype(np.float32)))
        
        if self.ignore_legal_actions:
            mcts_action_mask = np.ones(self.action_space_size, dtype=np.int8)
        else:
            mcts_action_mask = true_action_mask
            
        return {'observation': final_obs_vec, 'action_mask': mcts_action_mask, 'to_play': np.array([-1], dtype=np.int8)}

    def _calculate_max_profit_loss(self, position: pd.Series) -> Tuple[float, float]:
        if position['strategy_id'] != -1 and position['strategy_max_loss'] != 0:
            return position['strategy_max_profit'], position['strategy_max_loss']
        else:
            entry_premium = position['entry_premium'] * self.lot_size
            if position['direction'] == 'long':
                max_profit = self.undefined_risk_cap
                max_loss = -entry_premium
            else:
                max_profit = entry_premium
                max_loss = -self.undefined_risk_cap
            return max_profit, max_loss

    def render(self, mode: str = 'human') -> None:
        total_pnl = self._get_total_pnl()
        print(f"Step: {self.current_step:04d} | Day: {self.current_step // self.steps_per_day + 1:02d} | Price: ${self.current_price:9.2f} | Positions: {len(self.portfolio):1d} | Total PnL: ${total_pnl:9.2f}")
        # <<< NEW: Add detailed portfolio rendering
        if not self.portfolio.empty:
            print(self.portfolio.to_string())
    
    @property
    def observation_space(self) -> gym.spaces.Space: return self._observation_space
    @property
    def action_space(self) -> gym.spaces.Space: return self._action_space
    @property
    def reward_space(self) -> gym.spaces.Space: return self._reward_range
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
