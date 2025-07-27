import copy
import math
import random
from typing import Tuple, Dict, Any, List

import gym
import numpy as np
import pandas as pd
from arch import arch_model
from easydict import EasyDict
from gym import spaces
from gymnasium.utils import seeding
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta
from scipy.stats import norm

from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY

# Optional: For JIT compilation. If numba is installed, this will significantly speed up the math-heavy functions.
try:
    from numba import jit
except ImportError:
    # If numba is not installed, create a dummy decorator that does nothing.
    def jit(nopython=False):
        def decorator(func):
            return func
        return decorator

@ENV_REGISTRY.register('options_zero_game')
class OptionsZeroGameEnv(gym.Env):
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
        # <<< NEW: Configurable thresholds instead of magic numbers
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

        # ... (Standard parameters)
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
        # <<< NEW: Store the thresholds as attributes
        self.otm_threshold: float = self._cfg.otm_delta_threshold
        self.itm_threshold: float = self._cfg.itm_delta_threshold
        
        self.iv_bins: Dict[str, Dict[str, np.ndarray]] = self._discretize_iv_skew(self._cfg.iv_skew_table)
        
        self.actions_to_indices: Dict[str, int] = self._build_action_space()
        self.indices_to_actions: Dict[int, str] = {v: k for k, v in self.actions_to_indices.items()}
        self.action_space_size: int = len(self.actions_to_indices)

        self._action_space: spaces.Discrete = spaces.Discrete(self.action_space_size)
        self._observation_space: spaces.Dict = self._create_observation_space()
        self._reward_range: spaces.Box = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.np_random: Any = None
        
        self.portfolio_columns: List[str] = ['type', 'direction', 'entry_step', 'strike_price', 'entry_premium', 'days_to_expiry']
        self.portfolio: pd.DataFrame = pd.DataFrame(columns=self.portfolio_columns)

    def _discretize_iv_skew(self, skew_table: Dict[str, Dict[str, Tuple[float, float]]], num_bins: int = 5) -> Dict[str, Dict[str, np.ndarray]]:
        binned_ivs = {'call': {}, 'put': {}}
        for option_type, table in skew_table.items():
            for offset_str, iv_range in table.items():
                min_iv, max_iv = iv_range
                binned_ivs[option_type][offset_str] = np.linspace(min_iv, max_iv, num_bins) / 100.0
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
        self.market_and_portfolio_state_size = 5 + self.max_positions * 8
        self.obs_vector_size = self.market_and_portfolio_state_size + self.action_space_size
        return spaces.Dict({'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_vector_size,), dtype=np.float32),'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space_size,), dtype=np.int8),'to_play': spaces.Box(low=-1, high=-1, shape=(1,), dtype=np.int8)})

    def seed(self, seed: int, dynamic_seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _generate_price_path(self) -> None:
        garch_spec = arch_model(np.zeros(100), mean='Constant', vol='GARCH', p=1, q=1)
        mu_step = self.trend / self.steps_per_day
        omega_step = self.omega / self.steps_per_day
        alpha_step = self.alpha / self.steps_per_day
        beta_step = self.beta
        params = np.array([mu_step, omega_step, alpha_step, beta_step])
        sim_data = garch_spec.simulate(params, nobs=self.total_steps + 1)
        price_path = np.zeros(self.total_steps + 1)
        price_path[0] = self.start_price
        for i in range(1, self.total_steps + 1):
            price_path[i] = price_path[i - 1] * np.exp(sim_data['data'][i - 1])
        self.price_path = price_path
        log_returns = np.diff(np.log(price_path))
        returns_series = pd.Series(log_returns)
        rolling_window_steps = self.rolling_vol_window * self.steps_per_day
        self.realized_vol_series = returns_series.rolling(window=rolling_window_steps).std().fillna(0) * np.sqrt(252 * self.steps_per_day)

    def _simulate_price_step(self) -> None:
        self.current_price = self.price_path[self.current_step]
    
    def _get_implied_volatility(self, offset: int, option_type: str) -> float:
        clamped_offset = max(-5, min(5, offset))
        return self.iv_bins[option_type][str(clamped_offset)][self.iv_bin_index]

    # <<< NEW: Candidate for JIT compilation for performance
    @jit(nopython=False)
    def _get_option_details(self, underlying_price: float, strike_price: float, days_to_expiry: float, option_type: str) -> Tuple[float, float, float]:
        t = days_to_expiry / 365.25
        intrinsic_value = 0.0
        if option_type == 'call':
            intrinsic_value = max(0.0, underlying_price - strike_price)
        else: # put
            intrinsic_value = max(0.0, strike_price - underlying_price)

        if t < (1.0 / 365.25):
            abs_delta_at_expiry = 1.0 if intrinsic_value > 0 else 0.0
            return intrinsic_value, abs_delta_at_expiry, 0.0
        
        atm_price = int(underlying_price / self.strike_distance + 0.5) * self.strike_distance
        offset = round((strike_price - atm_price) / self.strike_distance)
        vol = self._get_implied_volatility(offset, option_type)

        if vol < 1e-6:
            abs_delta_at_expiry = 1.0 if intrinsic_value > 0 else 0.0
            return intrinsic_value, abs_delta_at_expiry, 0.0
        
        try:
            d1 = (math.log(underlying_price / strike_price) + (self.risk_free_rate + 0.5 * vol ** 2) * t) / (vol * math.sqrt(t))
            d2 = d1 - vol * math.sqrt(t)
            d = delta(option_type[0], underlying_price, strike_price, t, self.risk_free_rate, vol)
            mid_price = black_scholes(option_type[0], underlying_price, strike_price, t, self.risk_free_rate, vol)
            return mid_price, abs(d), d2
        except (ValueError, ZeroDivisionError):
            abs_delta_at_expiry = 1.0 if intrinsic_value > 0 else 0.0
            return intrinsic_value, abs_delta_at_expiry, 0.0

    def _get_option_price(self, mid_price: float, is_buy: bool) -> float:
        if is_buy: return mid_price * (1 + self.bid_ask_spread_pct)
        else: return mid_price * (1 - self.bid_ask_spread_pct)

    def _get_total_pnl(self) -> float:
        if self.portfolio.empty:
            return self.realized_pnl

        def get_current_premium(row):
            mid_price, _, _ = self._get_option_details(self.current_price, row['strike_price'], row['days_to_expiry'], row['type'])
            return self._get_option_price(mid_price, is_buy=(row['direction'] == 'short'))

        current_premiums = self.portfolio.apply(get_current_premium, axis=1)

        pnl = (current_premiums - self.portfolio['entry_premium']) * self.lot_size

        unrealized_pnl = np.where(self.portfolio['direction'] == 'long', pnl, -pnl).sum()

        return self.realized_pnl + unrealized_pnl

    def _get_current_equity(self) -> float:
        return self.initial_cash + self._get_total_pnl()

    def reset(self, seed: int = None, **kwargs) -> Dict[str, np.ndarray]:
        if seed is not None: self.seed(seed)
        elif self.np_random is None: self.seed(0)

        chosen_regime = random.choice(self.market_regimes)
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
                time_decay_days += 2
            else:
                self.day_of_week += 1
        self._simulate_price_step()
        if not self.portfolio.empty:
            self.portfolio['days_to_expiry'] -= time_decay_days

    def step(self, action: int) -> BaseEnvTimestep:
        true_legal_actions_mask = self._get_true_action_mask()
        was_illegal_action = true_legal_actions_mask[action] == 0
        if was_illegal_action:
            action = self.actions_to_indices['HOLD']

        action_name = self.indices_to_actions.get(action, 'INVALID')
        equity_before = self._get_current_equity()

        if action_name.startswith('OPEN_'): self._handle_open_action(action_name)
        elif action_name.startswith('CLOSE_POSITION_'):
            try:
                pos_index = int(action_name.split('_')[-1])
                self._close_position(pos_index)
            except (ValueError, IndexError): pass
        elif action_name == 'CLOSE_ALL':
            while not self.portfolio.empty: self._close_position(-1)

        if not self.portfolio.empty:
            self.portfolio = self.portfolio.sort_values(by=['strike_price', 'type']).reset_index(drop=True)

        self._advance_time_and_market()

    def _handle_open_action(self, action_name: str) -> None:
        """
        Orchestrates which helper function to call based on the action name.
        """
        if 'IRON_CONDOR' in action_name:
            self._open_iron_condor(action_name)
        elif 'IRON_FLY' in action_name:
            self._open_iron_fly(action_name)
        elif 'STRANGLE' in action_name:
            self._open_strangle(action_name)
        elif 'STRADDLE' in action_name:
            self._open_straddle(action_name)
        else: # Fallback for all single-leg actions
            self._open_single_leg(action_name)

    def _open_single_leg(self, action_name: str) -> None:
        if len(self.portfolio) >= self.max_positions: return

        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (7.0 / 5.0)
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance

        _, direction, type, strike_str = action_name.split('_')
        offset = int(strike_str.replace('ATM', ''))
        strike_price = atm_price + (offset * self.strike_distance)
        is_buy = (direction == 'LONG')

        mid_price, _, _ = self._get_option_details(self.current_price, strike_price, days_to_expiry, type.lower())
        entry_premium = self._get_option_price(mid_price, is_buy)

        new_position = pd.DataFrame([{'type': type.lower(), 'direction': direction.lower(), 'entry_step': self.current_step, 'strike_price': strike_price, 'entry_premium': entry_premium, 'days_to_expiry': days_to_expiry}])
        self.portfolio = pd.concat([self.portfolio, new_position], ignore_index=True)

    def _open_straddle(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 2: return

        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (7.0 / 5.0)
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        direction = 'long' if 'LONG' in action_name else 'short'
        is_buy = (direction == 'long')

        trades_to_execute = []
        mid_price_call, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'call')
        trades_to_execute.append({'type': 'call', 'direction': direction, 'strike_price': atm_price, 'entry_premium': self._get_option_price(mid_price_call, is_buy)})
        mid_price_put, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'put')
        trades_to_execute.append({'type': 'put', 'direction': direction, 'strike_price': atm_price, 'entry_premium': self._get_option_price(mid_price_put, is_buy)})

        if trades_to_execute:
            new_positions_df = pd.DataFrame(trades_to_execute)
            self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)

    def _open_strangle(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 2: return

        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (7.0 / 5.0)
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

        if trades_to_execute:
            new_positions_df = pd.DataFrame(trades_to_execute)
            self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)

    def _open_iron_fly(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 4: return

        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (7.0 / 5.0)
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

        if trades_to_execute:
            new_positions_df = pd.DataFrame(trades_to_execute)
            self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)

    def _open_iron_condor(self, action_name: str) -> None:
        if len(self.portfolio) > self.max_positions - 4: return

        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_days - current_trading_day
        days_to_expiry = trading_days_left * (7.0 / 5.0)
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        direction = 'long' if 'LONG' in action_name else 'short'
        target_delta_short = 0.30
        target_delta_long = 0.10
        min_strike = self.strike_distance
        max_put_offset = int((atm_price - min_strike) / self.strike_distance)
        put_search_range = range(1, max_put_offset + 1)
        call_search_range = range(1, 15)

        trades_to_execute = []
        best_short_call_strike, best_long_call_strike = atm_price, atm_price
        min_delta_diff_short_call, min_delta_diff_long_call = 999, 999
        for offset in call_search_range:
            strike = atm_price + (offset * self.strike_distance)
            _, d, _ = self._get_option_details(self.current_price, strike, days_to_expiry, 'call')
            if abs(d - target_delta_short) < min_delta_diff_short_call:
                min_delta_diff_short_call = abs(d - target_delta_short)
                best_short_call_strike = strike
            if abs(d - target_delta_long) < min_delta_diff_long_call:
                min_delta_diff_long_call = abs(d - target_delta_long)
                best_long_call_strike = strike
        best_short_put_strike, best_long_put_strike = atm_price, atm_price
        min_delta_diff_short_put, min_delta_diff_long_put = 999, 999
        for offset in put_search_range:
            strike = atm_price - (offset * self.strike_distance)
            _, d, _ = self._get_option_details(self.current_price, strike, days_to_expiry, 'put')
            if abs(d - target_delta_short) < min_delta_diff_short_put:
                min_delta_diff_short_put = abs(d - target_delta_short)
                best_short_put_strike = strike
            if abs(d - target_delta_long) < min_delta_diff_long_put:
                min_delta_diff_long_put = abs(d - target_delta_long)
                best_long_put_strike = strike
        if best_long_call_strike <= best_short_call_strike: best_long_call_strike = best_short_call_strike + self.strike_distance
        if best_long_put_strike >= best_short_put_strike: best_long_put_strike = best_short_put_strike - self.strike_distance

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

        if trades_to_execute:
            new_positions_df = pd.DataFrame(trades_to_execute)
            self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)

    def _get_true_action_mask(self) -> np.ndarray:
        action_mask = np.zeros(self.action_space_size, dtype=np.int8)

        current_trading_day = self.current_step // self.steps_per_day
        is_expiry_day = current_trading_day >= self.time_to_expiry_trading_days - 1

        # --- Always Legal Actions ---
        action_mask[self.actions_to_indices['HOLD']] = 1

        # --- Close Action Logic ---
        if not self.portfolio.empty:
            action_mask[self.actions_to_indices['CLOSE_ALL']] = 1
            for i in range(len(self.portfolio)):
                if f'CLOSE_POSITION_{i}' in self.actions_to_indices:
                    action_mask[self.actions_to_indices[f'CLOSE_POSITION_{i}']] = 1

        # --- Expiry Day Rule ---
        if is_expiry_day:
            return action_mask

        # --- Open Action Logic (Only on non-expiry days) ---
        num_long_calls = len(self.portfolio.query("type == 'call' and direction == 'long'"))
        num_short_calls = len(self.portfolio.query("type == 'call' and direction == 'short'"))
        num_long_puts = len(self.portfolio.query("type == 'put' and direction == 'long'"))
        num_short_puts = len(self.portfolio.query("type == 'put' and direction == 'short'"))
        existing_positions = {(p['strike_price'], p['type']): p['direction'] for i, p in self.portfolio.iterrows()}

        # Multi-leg combo actions
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

        # Single-leg actions
        if len(self.portfolio) < self.max_positions:
            atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
            days_to_expiry = self.time_to_expiry_trading_days - current_trading_day
            for offset in range(-3, 4):
                strike_price = atm_price + (offset * self.strike_distance)
                for option_type in ['call', 'put']:
                    _, d, _ = self._get_option_details(self.current_price, strike_price, days_to_expiry, option_type)

                    # Use the configurable thresholds
                    is_far_otm = d < self.otm_threshold
                    is_far_itm = d > self.itm_threshold

                    # --- Check LONG actions ---
                    action_name_long = f'OPEN_LONG_{option_type.upper()}_ATM{"+" if offset >=0 else ""}{offset}'
                    is_legal_long = (
                        (not (is_far_otm or is_far_itm)) and
                        not (option_type == 'put' and existing_positions.get((strike_price, 'call')) == 'short') and
                        not (option_type == 'call' and existing_positions.get((strike_price, 'put')) == 'short') and
                        not (option_type == 'call' and num_long_calls > 0) and
                        not (option_type == 'put' and num_long_puts > 0)
                    )
                    if is_legal_long:
                        action_mask[self.actions_to_indices[action_name_long]] = 1

                    # --- Check SHORT actions ---
                    action_name_short = f'OPEN_SHORT_{option_type.upper()}_ATM{"+" if offset >=0 else ""}{offset}'
                    is_legal_short = (
                        (not is_far_itm) and
                        not (option_type == 'put' and existing_positions.get((strike_price, 'call')) == 'long') and
                        not (option_type == 'call' and existing_positions.get((strike_price, 'put')) == 'long') and
                        not (option_type == 'call' and num_short_calls > 0) and
                        not (option_type == 'put' and num_short_puts > 0)
                    )
                    if is_legal_short:
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

        if self.current_step > 0:
            log_return = math.log(self.current_price / self.price_path[self.current_step - 1])
        else:
            log_return = 0.0
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
            _, _, d2 = self._get_option_details(self.current_price, pos['strike_price'], pos['days_to_expiry'], pos['type'])
            pop = 0.0
            if pos['type'] == 'call': pop = norm.cdf(d2) if pos['direction'] == 'long' else 1 - norm.cdf(d2)
            else: pop = 1 - norm.cdf(d2) if pos['direction'] == 'long' else norm.cdf(d2)
            market_portfolio_vec[current_idx + 5] = pop
            max_profit, max_loss = self._calculate_max_profit_loss(pos)
            market_portfolio_vec[current_idx + 6] = math.tanh(max_profit / self.initial_cash)
            market_portfolio_vec[current_idx + 7] = math.tanh(max_loss / self.initial_cash)
            current_idx += 8

        true_action_mask = self._get_true_action_mask()

        final_obs_vec = np.concatenate((market_portfolio_vec, true_action_mask.astype(np.float32)))

        mcts_action_mask = np.ones(self.action_space_size, dtype=np.int8)

        return {'observation': final_obs_vec, 'action_mask': mcts_action_mask, 'to_play': np.array([-1], dtype=np.int8)}

    def _calculate_max_profit_loss(self, position: pd.Series) -> Tuple[float, float]:
        entry_premium = position['entry_premium'] * self.lot_size
        if position['direction'] == 'long':
            max_profit = self.initial_cash * 10
            max_loss = -entry_premium
        else:
            max_profit = entry_premium
            max_loss = -self.initial_cash * 10
        return max_profit, max_loss

    def render(self, mode: str = 'human') -> None:
        total_pnl = self._get_total_pnl()
        print(f"Step: {self.current_step:04d} | Day: {self.current_step // self.steps_per_day + 1:02d} | Price: ${self.current_price:9.2f} | Positions: {len(self.portfolio):1d} | Total PnL: ${total_pnl:9.2f}")
    
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
