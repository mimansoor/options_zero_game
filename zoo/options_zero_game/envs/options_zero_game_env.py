import copy
import math

import gym
import numpy as np
from arch import arch_model
from easydict import EasyDict
from gym import spaces
from gymnasium.utils import seeding
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta

from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('options_zero_game')
class OptionsZeroGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    config = dict(
        start_price=100.0,
        trend=0.0,
        volatility=0.20,
        time_to_expiry=30,
        strike_distance=1.0,
        lot_size=100,
        max_positions=4,
        bid_ask_spread_pct=0.01,
        risk_free_rate=0.05,
        pnl_scaling_factor=1000,
        ignore_legal_actions=True,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None):
        self._cfg = self.default_config()
        if cfg is not None:
            self._cfg.update(cfg)

        self.start_price = self._cfg.start_price
        self.trend = self._cfg.trend
        self.volatility = self._cfg.volatility
        self.total_steps = self._cfg.time_to_expiry
        self.risk_free_rate = self._cfg.risk_free_rate
        self.lot_size = self._cfg.lot_size
        self.strike_distance = self._cfg.strike_distance
        self.max_positions = self._cfg.max_positions
        self.bid_ask_spread_pct = self._cfg.bid_ask_spread_pct
        self.pnl_scaling_factor = self._cfg.pnl_scaling_factor
        self.ignore_legal_actions = self._cfg.ignore_legal_actions
        
        self.actions_to_indices = self._build_action_space()
        self.indices_to_actions = {v: k for k, v in self.actions_to_indices.items()}
        self.action_space_size = len(self.actions_to_indices)

        self._action_space = spaces.Discrete(self.action_space_size)
        self._observation_space = self._create_observation_space()
        self._reward_range = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.np_random = None
        self._initialize_price_simulator()

    def _build_action_space(self):
        actions = {'HOLD': 0}
        i = 1
        for offset in range(-3, 4):
            sign = '+' if offset >= 0 else ''
            actions[f'OPEN_LONG_CALL_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_SHORT_CALL_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_LONG_PUT_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_SHORT_PUT_ATM{sign}{offset}'] = i; i+=1
        for j in range(self.max_positions):
            actions[f'CLOSE_POSITION_{j}'] = i; i+=1
        actions['CLOSE_ALL'] = i
        return actions

    def _create_observation_space(self):
        obs_shape = (3,)
        return spaces.Dict({'observation': spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32), 'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space_size,), dtype=np.int8), 'to_play': spaces.Box(low=-1, high=-1, shape=(1,), dtype=np.int8)})

    def seed(self, seed: int, dynamic_seed: int = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _initialize_price_simulator(self):
        init_returns = np.random.RandomState(0).normal(loc=self.trend, scale=self.volatility / np.sqrt(252), size=1000)
        self.garch_model = arch_model(init_returns * 100, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
        self.garch_fit = self.garch_model.fit(disp='off', show_warning=False)

    def _simulate_price_step(self):
        forecast = self.garch_fit.forecast(horizon=1, reindex=False)
        cond_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        shock = self.np_random.normal(loc=self.trend, scale=cond_vol)
        self.current_price *= (1 + shock)
    
    def _get_option_details(self, underlying_price, strike_price, days_to_expiry, option_type):
        t = days_to_expiry / 365.25
        if t <= 0: return 0, 0
        vol = max(self.volatility, 1e-6)
        try:
            d = delta(option_type[0], underlying_price, strike_price, t, self.risk_free_rate, vol)
            mid_price = black_scholes(option_type[0], underlying_price, strike_price, t, self.risk_free_rate, vol)
            return mid_price, abs(d)
        except Exception:
            return 0, 0

    def _get_option_price(self, mid_price, is_buy):
        if is_buy: return mid_price * (1 + self.bid_ask_spread_pct)
        else: return mid_price * (1 - self.bid_ask_spread_pct)

    def _get_portfolio_value(self):
        unrealized_pnl = 0.0
        for pos in self.portfolio:
            mid_price, _ = self._get_option_details(self.current_price, pos['strike_price'], pos['days_to_expiry'], pos['type'])
            current_premium = self._get_option_price(mid_price, is_buy=(pos['direction'] == 'short'))
            entry_premium = pos['entry_premium']
            if pos['direction'] == 'long': pnl = (current_premium - entry_premium) * self.lot_size
            else: pnl = (entry_premium - current_premium) * self.lot_size
            unrealized_pnl += pnl
        return self.realized_pnl + unrealized_pnl

    def reset(self, seed: int = None, **kwargs):
        if seed is not None: self.seed(seed)
        self.current_step = 0
        self.current_price = self.start_price
        self.portfolio = []
        self.realized_pnl = 0.0
        self._final_eval_reward = 0.0
        return self._get_observation()

    def _close_position(self, position_index):
        if position_index < len(self.portfolio):
            pos_to_close = self.portfolio.pop(position_index)
            mid_price, _ = self._get_option_details(self.current_price, pos_to_close['strike_price'], pos_to_close['days_to_expiry'], pos_to_close['type'])
            exit_premium = self._get_option_price(mid_price, is_buy=(pos_to_close['direction'] == 'short'))
            entry_premium = pos_to_close['entry_premium']
            if pos_to_close['direction'] == 'long': pnl = (exit_premium - entry_premium) * self.lot_size
            else: pnl = (entry_premium - exit_premium) * self.lot_size
            self.realized_pnl += pnl

    def step(self, action: int):
        action_name = self.indices_to_actions.get(action, 'INVALID')
        tpv_before = self._get_portfolio_value()

        if action_name.startswith('OPEN_'):
            if len(self.portfolio) < self.max_positions:
                _, direction, type, strike_str = action_name.split('_')
                offset = int(strike_str.replace('ATM', ''))
                atm_price = round(self.current_price / self.strike_distance) * self.strike_distance
                strike_price = atm_price + (offset * self.strike_distance)
                days_to_expiry = self.total_steps - self.current_step
                is_buy = (direction == 'LONG')
                mid_price, _ = self._get_option_details(self.current_price, strike_price, days_to_expiry, type.lower())
                entry_premium = self._get_option_price(mid_price, is_buy)
                self.portfolio.append({'type': type.lower(), 'direction': direction.lower(), 'entry_step': self.current_step, 'strike_price': strike_price, 'entry_premium': entry_premium, 'days_to_expiry': days_to_expiry})
        elif action_name.startswith('CLOSE_POSITION_'):
            try:
                pos_index = int(action_name.split('_')[-1])
                self._close_position(pos_index)
            except (ValueError, IndexError): pass
        elif action_name == 'CLOSE_ALL':
            while len(self.portfolio) > 0: self._close_position(0)
        
        self.portfolio.sort(key=lambda p: (p['strike_price'], p['type']))
        
        self._simulate_price_step()
        self.current_step += 1
        for pos in self.portfolio: pos['days_to_expiry'] -= 1

        tpv_after = self._get_portfolio_value()
        raw_reward = tpv_after - tpv_before
        final_reward = math.tanh(raw_reward / self.pnl_scaling_factor)

        done = self.current_step >= self.total_steps
        if done:
            while len(self.portfolio) > 0: self._close_position(0)
            tpv_after = self._get_portfolio_value()
            raw_reward = tpv_after - tpv_before
            final_reward = math.tanh(raw_reward / self.pnl_scaling_factor)
        
        obs = self._get_observation()
        self._final_eval_reward += raw_reward
        info = {'price': self.current_price, 'eval_episode_return': self._final_eval_reward}
        return BaseEnvTimestep(obs, final_reward, done, info)

    def _get_observation(self):
        norm_price = (self.current_price / self.start_price) - 1.0
        pos_count = len(self.portfolio) / self.max_positions
        total_pnl = self._get_portfolio_value()
        norm_pnl = math.tanh(total_pnl / (self.pnl_scaling_factor * 10))
        obs_vec = np.array([norm_price, pos_count, norm_pnl], dtype=np.float32)

        if self.ignore_legal_actions:
            action_mask = np.ones(self.action_space_size, dtype=np.int8)
        else:
            action_mask = np.zeros(self.action_space_size, dtype=np.int8)
            action_mask[self.actions_to_indices['HOLD']] = 1
            
            can_open = len(self.portfolio) < self.max_positions
            if can_open:
                atm_price = round(self.current_price / self.strike_distance) * self.strike_distance
                days_to_expiry = self.total_steps - self.current_step
                
                # <<< NEW: Create a quick lookup of existing positions for the anti-synthetic rule
                existing_positions = {(p['strike_price'], p['type']): p['direction'] for p in self.portfolio}

                for offset in range(-3, 4):
                    strike_price = atm_price + (offset * self.strike_distance)
                    
                    for option_type in ['call', 'put']:
                        _, d = self._get_option_details(self.current_price, strike_price, days_to_expiry, option_type)
                        
                        is_far_otm = d < 0.15
                        is_far_itm = d > 0.85
                        
                        # --- Check LONG actions ---
                        action_name = f'OPEN_LONG_{option_type.upper()}_ATM{"+" if offset >=0 else ""}{offset}'
                        is_legal = not (is_far_otm or is_far_itm)
                        
                        # <<< NEW: Anti-Synthetic Rule Check for Long positions
                        # Synthetic Short Stock = Short Call + Long Put. Check if we are about to create one.
                        if option_type == 'put' and existing_positions.get((strike_price, 'call')) == 'short':
                            is_legal = False
                        # Synthetic Long Stock = Long Call + Short Put. Check if we are about to create one.
                        if option_type == 'call' and existing_positions.get((strike_price, 'put')) == 'short':
                            is_legal = False

                        if is_legal:
                            action_mask[self.actions_to_indices[action_name]] = 1
                        
                        # --- Check SHORT actions ---
                        action_name = f'OPEN_SHORT_{option_type.upper()}_ATM{"+" if offset >=0 else ""}{offset}'
                        is_legal = not is_far_itm

                        # <<< NEW: Anti-Synthetic Rule Check for Short positions
                        # Synthetic Long Stock = Long Call + Short Put. Check if we are about to create one.
                        if option_type == 'put' and existing_positions.get((strike_price, 'call')) == 'long':
                            is_legal = False
                        # Synthetic Short Stock = Short Call + Long Put. Check if we are about to create one.
                        if option_type == 'call' and existing_positions.get((strike_price, 'put')) == 'long':
                            is_legal = False

                        if is_legal:
                            action_mask[self.actions_to_indices[action_name]] = 1
            
            if len(self.portfolio) > 0:
                action_mask[self.actions_to_indices['CLOSE_ALL']] = 1
            for i in range(len(self.portfolio)):
                action_mask[self.actions_to_indices[f'CLOSE_POSITION_{i}']] = 1
        
        return {'observation': obs_vec, 'action_mask': action_mask, 'to_play': np.array([-1], dtype=np.int8)}

    def render(self, mode='human'):
        portfolio_val = self._get_portfolio_value()
        print(f"Step: {self.current_step:02d} | Price: ${self.current_price:8.2f} | Positions: {len(self.portfolio):1d} | Total PnL: ${portfolio_val:9.2f}")
    
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
    def __repr__(self): return "LightZero Options-Zero-Game Env"
