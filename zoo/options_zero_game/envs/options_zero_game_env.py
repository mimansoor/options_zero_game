# zoo/options_zero_game/envs/options_zero_game_env.py
# <<< FINAL VERSION, incorporating all fixes >>>

import copy
import math
import random
import time
from typing import Tuple, Dict, Any, List

import gymnasium as gym
import numpy as np
from easydict import EasyDict
from gymnasium import spaces
from gymnasium.utils import seeding

from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY

# --- Import the new, specialized manager classes ---
from .black_scholes_manager import BlackScholesManager
from .price_action_manager import PriceActionManager
from .portfolio_manager import PortfolioManager

@ENV_REGISTRY.register('options_zero_game')
class OptionsZeroGameEnv(gym.Env):
    VERSION = "2.1-Refactored-Final"
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
        momentum_window_steps=20,
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
        strategy_name_to_id = {
            # --- Base Strategies ---
            'LONG_CALL': 0, 'SHORT_CALL': 1, 'LONG_PUT': 2, 'SHORT_PUT': 3,
            'LONG_STRADDLE': 4, 'SHORT_STRADDLE': 5,

            # --- Dynamic Strategies (Name includes width) ---
            'LONG_STRANGLE_1': 6, 'SHORT_STRANGLE_1': 7,
            'LONG_STRANGLE_2': 8, 'SHORT_STRANGLE_2': 9,

            'LONG_IRON_FLY': 10, 'SHORT_IRON_FLY': 11,
            'LONG_IRON_CONDOR': 12, 'SHORT_IRON_CONDOR': 13,

            'LONG_VERTICAL_CALL_1': 14, 'SHORT_VERTICAL_CALL_1': 16,
            'LONG_VERTICAL_CALL_2': 15, 'SHORT_VERTICAL_CALL_2': 17,
            'LONG_VERTICAL_PUT_1': 18, 'SHORT_VERTICAL_PUT_1': 20,
            'LONG_VERTICAL_PUT_2': 19, 'SHORT_VERTICAL_PUT_2': 21,

            # Note: Butterflies are often referenced by their action name
            # but can be added here for consistency if needed.
            'LONG_CALL_FLY_1': 22, 'SHORT_CALL_FLY_1': 23,
            'LONG_PUT_FLY_1': 24, 'SHORT_PUT_FLY_1': 25,
            'LONG_CALL_FLY_2': 26, 'SHORT_CALL_FLY_2': 27,
            'LONG_PUT_FLY_2': 28, 'SHORT_PUT_FLY_2': 29,
        }
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None) -> None:
        self._cfg = self.default_config()
        if cfg is not None: self._cfg.update(cfg)
        
        self.np_random, _ = seeding.np_random(None)

        self.bs_manager = BlackScholesManager(self._cfg)
        self.price_manager = PriceActionManager(self._cfg, self.np_random)
        self.portfolio_manager = PortfolioManager(self._cfg, self.bs_manager)
        
        self.actions_to_indices = self._build_action_space()
        self.indices_to_actions = {v: k for k, v in self.actions_to_indices.items()}
        self.action_space_size = len(self.actions_to_indices)

        self.OBS_IDX = {
            'PRICE_NORM': 0, 'TIME_NORM': 1, 'PNL_NORM': 2, 'VOL_MISMATCH_NORM': 3, 'LOG_RETURN': 4,
            'MOMENTUM_NORM': 5, 'PORTFOLIO_DELTA': 6, 'PORTFOLIO_GAMMA': 7, 'PORTFOLIO_THETA': 8, 'PORTFOLIO_VEGA': 9,
            'PORTFOLIO_MAX_PROFIT_NORM': 10, 'PORTFOLIO_MAX_LOSS_NORM': 11, 'PORTFOLIO_RR_RATIO_NORM': 12, 'PORTFOLIO_PROB_PROFIT': 13,
        }
        self.POS_IDX = {
            'IS_OCCUPIED': 0, 'TYPE_NORM': 1, 'DIRECTION_NORM': 2, 'STRIKE_DIST_NORM': 3, 'DAYS_HELD_NORM': 4,
            'PROB_OF_PROFIT': 5, 'MAX_PROFIT_NORM': 6, 'MAX_LOSS_NORM': 7, 'DELTA': 8, 'GAMMA': 9, 'THETA': 10, 'VEGA': 11,
        }
        
        self.MARKET_STATE_SIZE = len(self.OBS_IDX)
        self.PORTFOLIO_STATE_SIZE_PER_POS = len(self.POS_IDX)
        self.PORTFOLIO_START_IDX = self.MARKET_STATE_SIZE
        
        self.market_and_portfolio_state_size = self.MARKET_STATE_SIZE + (self._cfg.max_positions * self.PORTFOLIO_STATE_SIZE_PER_POS)
        self.obs_vector_size = self.market_and_portfolio_state_size + self.action_space_size
        
        self._action_space = spaces.Discrete(self.action_space_size)
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_vector_size,), dtype=np.float32)
        self._reward_range = (-1.0, 1.0)

        self.current_step = 0
        self.total_steps = self._cfg.time_to_expiry_days * self._cfg.steps_per_day
        self.iv_bin_index = 0
        self.final_eval_reward = 0.0
        self.illegal_action_count = 0
        self.realized_vol_series = np.array([])
        
        # --- THE FIX ---
        # Restoring the time constants needed by the orchestrator methods.
        self.TRADING_DAY_IN_MINS = self._cfg.trading_day_in_mins
        self.MINS_IN_DAY = 24 * 60
        self.TRADING_DAYS_IN_WEEK = 5
        self.TOTAL_DAYS_IN_WEEK = 7
        self.decay_per_step_trading = (self.TRADING_DAY_IN_MINS / self._cfg.steps_per_day) / self.MINS_IN_DAY
        self.decay_overnight = (self.MINS_IN_DAY - self.TRADING_DAY_IN_MINS) / self.MINS_IN_DAY
        self.decay_weekend = 2.0

    def seed(self, seed: int, dynamic_seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        self.price_manager.np_random = self.np_random
        return [seed]

    def reset(self, seed: int = None, **kwargs) -> Dict:
        if seed is not None: self.seed(seed)
        else:
            # If the framework or user does not provide a seed, we create a
            # random one. This is useful for standalone testing.
            # During normal training, the framework will ALWAYS provide a seed.
            new_seed = int(time.time())
            self.seed(new_seed)
        
        self.price_manager.reset()
        self.portfolio_manager.reset()
        
        self.current_step = 0
        self.final_eval_reward = 0.0
        self.illegal_action_count = 0
        self.realized_vol_series = np.zeros(self.total_steps + 1, dtype=np.float32)
        self.iv_bin_index = random.randint(0, len(self.bs_manager.iv_bins['call']['0']) - 1)

        obs = self._get_observation()
        action_mask = self._get_true_action_mask() if not self._cfg.ignore_legal_actions else np.ones(self.action_space_size, dtype=np.int8)

        return {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

    def step(self, action: int) -> BaseEnvTimestep:
        equity_before = self.portfolio_manager.get_current_equity(self.price_manager.current_price, self.iv_bin_index)
        was_illegal_action = self._handle_action(action)
        self.current_step += 1
        self.price_manager.step(self.current_step)
        self._update_realized_vol()
        
        time_decay_days = self._calculate_time_decay()
        self.portfolio_manager.update_positions_after_time_step(time_decay_days, self.price_manager.current_price, self.iv_bin_index)
        
        terminated = self.current_step >= self.total_steps
        if terminated: self.portfolio_manager.close_all_positions(self.price_manager.current_price, self.iv_bin_index)

        equity_after = self.portfolio_manager.get_current_equity(self.price_manager.current_price, self.iv_bin_index)
        shaped_reward, raw_reward = self._calculate_shaped_reward(equity_after, equity_before)
        self.final_eval_reward += raw_reward
        final_reward = self._cfg.illegal_action_penalty if was_illegal_action else shaped_reward

        obs = self._get_observation()
        action_mask = self._get_true_action_mask() if not self._cfg.ignore_legal_actions else np.ones(self.action_space_size, dtype=np.int8)
        info = {
            'price': self.price_manager.current_price,
            'eval_episode_return': self.final_eval_reward,
            'illegal_actions_in_episode': self.illegal_action_count,
            'was_illegal_action': was_illegal_action
        }

        return BaseEnvTimestep({'observation': obs, 'action_mask': action_mask, 'to_play': -1}, final_reward, terminated, info)

    def _handle_action(self, action: int) -> bool:
        """Checks action legality and delegates to the PortfolioManager."""
        if self._get_true_action_mask()[action] == 0:
            self.illegal_action_count += 1
            return True

        action_name = self.indices_to_actions.get(action, 'INVALID')
        
        if action_name.startswith('OPEN_'):
            # This logic is correct now because the constants exist on self
            current_day = self.current_step // self._cfg.steps_per_day
            days_to_expiry = (self._cfg.time_to_expiry_days - current_day) * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)
            self.portfolio_manager.open_strategy(action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step, days_to_expiry)
        elif action_name.startswith('CLOSE_POSITION_'):
            self.portfolio_manager.close_position(int(action_name.split('_')[-1]), self.price_manager.current_price, self.iv_bin_index)
        elif action_name == 'CLOSE_ALL':
            self.portfolio_manager.close_all_positions(self.price_manager.current_price, self.iv_bin_index)
        elif action_name.startswith('SHIFT_'):
            self.portfolio_manager.shift_position(action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step)

        self.portfolio_manager.sort_portfolio()
        return False

    def _get_observation(self) -> np.ndarray:
        # Market State
        vec = np.zeros(self.market_and_portfolio_state_size, dtype=np.float32)
        vec[self.OBS_IDX['PRICE_NORM']] = (self.price_manager.current_price / self.price_manager.start_price) - 1.0
        vec[self.OBS_IDX['TIME_NORM']] = (self.total_steps - self.current_step) / self.total_steps
        vec[self.OBS_IDX['PNL_NORM']] = math.tanh(self.portfolio_manager.get_total_pnl(self.price_manager.current_price, self.iv_bin_index) / self._cfg.initial_cash)
        garch_vol = self.price_manager.garch_implied_vol
        vec[self.OBS_IDX['VOL_MISMATCH_NORM']] = math.tanh((self.realized_vol_series[self.current_step] / garch_vol) - 1.0) if garch_vol > 0 else 0.0
        log_return = math.log(self.price_manager.current_price / (self.price_manager.price_path[self.current_step - 1] + 1e-8)) if self.current_step > 0 else 0.0
        vec[self.OBS_IDX['LOG_RETURN']] = np.clip(log_return, -0.1, 0.1) * 10
        vec[self.OBS_IDX['MOMENTUM_NORM']] = self.price_manager.momentum_signal

        # Portfolio Greeks
        greeks = self.portfolio_manager.get_portfolio_greeks(self.price_manager.current_price, self.iv_bin_index)
        vec[self.OBS_IDX['PORTFOLIO_DELTA']] = greeks['delta_norm']
        vec[self.OBS_IDX['PORTFOLIO_GAMMA']] = greeks['gamma_norm']
        vec[self.OBS_IDX['PORTFOLIO_THETA']] = greeks['theta_norm']
        vec[self.OBS_IDX['PORTFOLIO_VEGA']] = greeks['vega_norm']

        summary = self.portfolio_manager.get_portfolio_summary(self.price_manager.current_price, self.iv_bin_index)

        # Normalize the values before adding them to the state
        vec[self.OBS_IDX['PORTFOLIO_MAX_PROFIT_NORM']] = math.tanh(summary['max_profit'] / self._cfg.initial_cash)
        vec[self.OBS_IDX['PORTFOLIO_MAX_LOSS_NORM']] = math.tanh(summary['max_loss'] / self._cfg.initial_cash)
        # RR Ratio is often spiky, tanh is a good choice
        vec[self.OBS_IDX['PORTFOLIO_RR_RATIO_NORM']] = math.tanh(summary['rr_ratio'])
        # Probability is already in a [0, 1] range, we can scale it to [-1, 1]
        vec[self.OBS_IDX['PORTFOLIO_PROB_PROFIT']] = (summary['prob_profit'] * 2) - 1.0

        # Per-Position State
        self.portfolio_manager.get_positions_state(vec, self.PORTFOLIO_START_IDX, self.PORTFOLIO_STATE_SIZE_PER_POS, self.POS_IDX, self.price_manager.current_price, self.iv_bin_index, self.current_step, self.total_steps)

        # 1. First, concatenate the state vector and the action mask
        true_action_mask = self._get_true_action_mask()
        final_obs_vec = np.concatenate((vec, true_action_mask.astype(np.float32)))

        # 2. Then, run assertions on the FINAL object
        assert final_obs_vec.shape == (self.obs_vector_size,), f"Observation shape mismatch. Expected {self.obs_vector_size}, but got {final_obs_vec.shape}"
        assert np.all(np.isfinite(final_obs_vec)), "Observation vector contains NaN or Inf."

        # 3. Return the fully validated final vector
        return final_obs_vec

    def _calculate_shaped_reward(self, equity_before: float, equity_after: float) -> Tuple[float, float]:
        raw_reward = equity_after - equity_before
        self.portfolio_manager.update_high_water_mark(equity_after)
        drawdown = self.portfolio_manager.high_water_mark - equity_after
        risk_adjusted_raw_reward = raw_reward - (self._cfg.drawdown_penalty_weight * drawdown)
        scaled_reward = risk_adjusted_raw_reward / self._cfg.pnl_scaling_factor

        # --- Defensive Assertion ---
        assert math.isfinite(scaled_reward), f"Calculated a non-finite reward: {scaled_reward}"
        return math.tanh(scaled_reward), raw_reward

    def _calculate_time_decay(self) -> float:
        time_decay = self.decay_per_step_trading
        if (self.current_step + 1) % self._cfg.steps_per_day == 0:
            day_of_week = (self.current_step // self._cfg.steps_per_day) % self.TRADING_DAYS_IN_WEEK
            time_decay += self.decay_weekend if day_of_week == 4 else self.decay_overnight
        return time_decay

    def _update_realized_vol(self):
        window = self._cfg.rolling_vol_window * self._cfg.steps_per_day
        start = max(0, self.current_step - window)
        prices = self.price_manager.price_path[start:self.current_step + 1]
        if len(prices) > 1:
            log_returns = np.diff(np.log(prices))
            vol = np.std(log_returns) * np.sqrt(252 * self._cfg.steps_per_day)
            self.realized_vol_series[self.current_step] = vol if np.isfinite(vol) else 0.0

    def _build_action_space(self) -> Dict[str, int]:
        actions = {'HOLD': 0}; i = 1
        for offset in range(-5, 6):
            sign = '+' if offset >= 0 else ''
            for t in ['CALL', 'PUT']:
                for d in ['LONG', 'SHORT']:
                    actions[f'OPEN_{d}_{t}_ATM{sign}{offset}'] = i; i+=1
        for d in ['LONG', 'SHORT']:
            actions[f'OPEN_{d}_STRADDLE_ATM'] = i; i+=1
        for w in [1, 2]:
            for d in ['LONG', 'SHORT']:
                actions[f'OPEN_{d}_STRANGLE_ATM_{w}'] = i; i+=1
        for w in [1, 2]:
            for t in ['CALL', 'PUT']:
                for d in ['LONG', 'SHORT']:
                    actions[f'OPEN_{d}_VERTICAL_{t}_{w}'] = i; i+=1
        for d in ['LONG', 'SHORT']:
            actions[f'OPEN_{d}_IRON_FLY'] = i; i+=1
            actions[f'OPEN_{d}_IRON_CONDOR'] = i; i+=1
        for w in [1, 2]:
            for t in ['CALL', 'PUT']:
                for d in ['LONG', 'SHORT']:
                    actions[f'OPEN_{d}_{t}_FLY_{w}'] = i; i+=1
        for j in range(self._cfg.max_positions):
            actions[f'CLOSE_POSITION_{j}'] = i; i+=1
        for j in range(self._cfg.max_positions):
            actions[f'SHIFT_UP_POS_{j}'] = i; i+=1
            actions[f'SHIFT_DOWN_POS_{j}'] = i; i+=1
        actions['CLOSE_ALL'] = i
        return actions
        
    def _get_true_action_mask(self) -> np.ndarray:
        """
        Calculates the complete, rule-based action mask with simplified logic.
        This version correctly handles all game states.
        """
        action_mask = np.zeros(self.action_space_size, dtype=np.int8)

        # --- Rule 1: Expiry Day is a special case that overrides everything else ---
        current_day = self.current_step // self._cfg.steps_per_day
        is_expiry_day = current_day >= (self._cfg.time_to_expiry_days - 1)

        if is_expiry_day:
            if self.portfolio_manager.portfolio.empty:
                # If portfolio is empty on expiry, the only thing to do is wait.
                action_mask[self.actions_to_indices['HOLD']] = 1
            else:
                # If there are positions, the only legal actions are to close them.
                action_mask[self.actions_to_indices['CLOSE_ALL']] = 1
                for i in range(len(self.portfolio_manager.portfolio)):
                    if f'CLOSE_POSITION_{i}' in self.actions_to_indices:
                        action_mask[self.actions_to_indices[f'CLOSE_POSITION_{i}']] = 1
            return action_mask

        # --- Rule 2: Main logic for all non-expiry days ---
        
        if not self.portfolio_manager.portfolio.empty:
            # Case A: A position is already open.
            # The agent can hold, close, or roll/shift the position.
            action_mask[self.actions_to_indices['HOLD']] = 1
            action_mask[self.actions_to_indices['CLOSE_ALL']] = 1
            for i in range(len(self.portfolio_manager.portfolio)):
                if f'CLOSE_POSITION_{i}' in self.actions_to_indices:
                    action_mask[self.actions_to_indices[f'CLOSE_POSITION_{i}']] = 1
                if f'SHIFT_UP_POS_{i}' in self.actions_to_indices:
                     action_mask[self.actions_to_indices[f'SHIFT_UP_POS_{i}']] = 1
                if f'SHIFT_DOWN_POS_{i}' in self.actions_to_indices:
                     action_mask[self.actions_to_indices[f'SHIFT_DOWN_POS_{i}']] = 1
        else:
            # Case B: The portfolio is empty. The agent must open a position.
            # This correctly handles BOTH step 0 and any later step where the portfolio is empty.
            if self.current_step == 0:
                # Sub-case B1: It's the first step. Force a DIVERSE opening. HOLD is illegal.
                strategy_families = {
                    "SINGLE_LEG": lambda name: 'ATM' in name, "STRADDLE": lambda name: 'STRADDLE' in name,
                    "STRANGLE": lambda name: 'STRANGLE' in name, "VERTICAL": lambda name: 'VERTICAL' in name,
                    "IRON_FLY_CONDOR": lambda name: 'IRON' in name, "BUTTERFLY": lambda name: 'FLY' in name and 'IRON' not in name,
                }
                chosen_family_name = random.choice(list(strategy_families.keys()))
                is_in_family = strategy_families[chosen_family_name]
                for action_name, index in self.actions_to_indices.items():
                    if action_name.startswith('OPEN_') and is_in_family(action_name):
                        action_mask[index] = 1
                if not np.any(action_mask): # Failsafe
                    for name, index in self.actions_to_indices.items():
                        if name.startswith('OPEN_'): action_mask[index] = 1
            else:
                # Sub-case B2: It's a later step. Allow any opening, and also allow holding.
                action_mask[self.actions_to_indices['HOLD']] = 1
                for name, index in self.actions_to_indices.items():
                    if name.startswith('OPEN_'):
                        action_mask[index] = 1

        return action_mask

    def render(self, mode: str = 'human') -> None:
        self.portfolio_manager.render(self.price_manager.current_price, self.current_step, self.iv_bin_index, self._cfg.steps_per_day)

    # --- Properties and Static Methods ---
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
        return [copy.deepcopy(cfg) for _ in range(collector_env_num)]
    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> list:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        return [copy.deepcopy(cfg) for _ in range(evaluator_env_num)]
    def __repr__(self): return f"LightZero Options-Zero-Game Env v{self.VERSION}"
