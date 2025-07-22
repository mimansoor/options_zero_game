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
from scipy.stats import norm

from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('options_zero_game')
class OptionsZeroGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    config = dict(
        start_price=100.0,
        initial_cash=100000.0,
        trend=0.0,
        volatility=0.20,
        time_to_expiry=30,
        strike_distance=1.0,
        lot_size=100,
        max_positions=4,
        bid_ask_spread_pct=0.01,
        risk_free_rate=0.05,
        pnl_scaling_factor=1000,
        # <<< NEW: Add a parameter to control the strength of the drawdown penalty
        drawdown_penalty_weight=0.1,
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

        # ... (All other parameters are unchanged)
        self.start_price = self._cfg.start_price
        self.initial_cash = self._cfg.initial_cash
        self.trend = self._cfg.trend
        self.volatility = self._cfg.volatility
        self.total_steps = self._cfg.time_to_expiry
        self.risk_free_rate = self._cfg.risk_free_rate
        self.lot_size = self._cfg.lot_size
        self.strike_distance = self._cfg.strike_distance
        self.max_positions = self._cfg.max_positions
        self.bid_ask_spread_pct = self._cfg.bid_ask_spread_pct
        self.pnl_scaling_factor = self._cfg.pnl_scaling_factor
        self.drawdown_penalty_weight = self._cfg.drawdown_penalty_weight
        self.ignore_legal_actions = self._cfg.ignore_legal_actions
        
        self.actions_to_indices = self._build_action_space()
        self.indices_to_actions = {v: k for k, v in self.actions_to_indices.items()}
        self.action_space_size = len(self.actions_to_indices)

        self._action_space = spaces.Discrete(self.action_space_size)
        self._observation_space = self._create_observation_space()
        self._reward_range = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.np_random = None
        self._initialize_price_simulator()

    # ... (build_action_space, _create_observation_space, seed, _initialize_price_simulator, _simulate_price_step are unchanged)
    def _build_action_space(self):
        actions = {'HOLD': 0}; i = 1
        for offset in range(-3, 4):
            sign = '+' if offset >= 0 else ''
            actions[f'OPEN_LONG_CALL_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_SHORT_CALL_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_LONG_PUT_ATM{sign}{offset}'] = i; i+=1
            actions[f'OPEN_SHORT_PUT_ATM{sign}{offset}'] = i; i+=1
        actions['OPEN_LONG_STRADDLE_ATM'] = i; i+=1
        actions['OPEN_SHORT_STRADDLE_ATM'] = i; i+=1
        for j in range(self.max_positions): actions[f'CLOSE_POSITION_{j}'] = i; i+=1
        actions['CLOSE_ALL'] = i
        return actions
    def _create_observation_space(self):
        self.obs_vector_size = 3 + self.max_positions * 8
        return spaces.Dict({'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_vector_size,), dtype=np.float32),'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space_size,), dtype=np.int8),'to_play': spaces.Box(low=-1, high=-1, shape=(1,), dtype=np.int8)})
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
        if t <= 0: return 0, 0, 0
        vol = max(self.volatility, 1e-6)
        try:
            d1 = (math.log(underlying_price / strike_price) + (self.risk_free_rate + 0.5 * vol ** 2) * t) / (vol * math.sqrt(t))
            d2 = d1 - vol * math.sqrt(t)
            d = delta(option_type[0], underlying_price, strike_price, t, self.risk_free_rate, vol)
            mid_price = black_scholes(option_type[0], underlying_price, strike_price, t, self.risk_free_rate, vol)
            return mid_price, abs(d), d2
        except Exception: return 0, 0, 0
    def _get_option_price(self, mid_price, is_buy):
        if is_buy: return mid_price * (1 + self.bid_ask_spread_pct)
        else: return mid_price * (1 - self.bid_ask_spread_pct)
    def _get_portfolio_value(self):
        unrealized_pnl = 0.0
        for pos in self.portfolio:
            mid_price, _, _ = self._get_option_details(self.current_price, pos['strike_price'], pos['days_to_expiry'], pos['type'])
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
        # <<< NEW: Initialize the high-water mark for drawdown calculation
        self.high_water_mark = 0.0 # Start at 0 PnL
        return self._get_observation()

    def _close_position(self, position_index):
        # ... (unchanged)
        if position_index < len(self.portfolio):
            pos_to_close = self.portfolio.pop(position_index)
            mid_price, _, _ = self._get_option_details(self.current_price, pos_to_close['strike_price'], pos_to_close['days_to_expiry'], pos_to_close['type'])
            exit_premium = self._get_option_price(mid_price, is_buy=(pos_to_close['direction'] == 'short'))
            entry_premium = pos_to_close['entry_premium']
            if pos_to_close['direction'] == 'long': pnl = (exit_premium - entry_premium) * self.lot_size
            else: pnl = (entry_premium - exit_premium) * self.lot_size
            self.realized_pnl += pnl

    def step(self, action: int):
        action_name = self.indices_to_actions.get(action, 'INVALID')
        tpv_before = self._get_portfolio_value()

        if action_name.startswith('OPEN_'): self._handle_open_action(action_name)
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
        
        # <<< MODIFIED: Implement the shaped reward with drawdown penalty
        pnl_component = math.tanh(raw_reward / self.pnl_scaling_factor)
        
        # Update high-water mark and calculate drawdown
        self.high_water_mark = max(self.high_water_mark, tpv_after)
        drawdown = self.high_water_mark - tpv_after
        
        # Normalize drawdown and apply penalty
        # We normalize by initial_cash to get a consistent scale
        normalized_drawdown = drawdown / self.initial_cash
        drawdown_penalty = self.drawdown_penalty_weight * normalized_drawdown
        
        # The final reward is the PnL component minus the penalty
        final_reward = pnl_component - drawdown_penalty

        done = self.current_step >= self.total_steps
        if done:
            while len(self.portfolio) > 0: self._close_position(0)
            tpv_after = self._get_portfolio_value()
            raw_reward = tpv_after - tpv_before
            # Recalculate final reward on the last step
            pnl_component = math.tanh(raw_reward / self.pnl_scaling_factor)
            self.high_water_mark = max(self.high_water_mark, tpv_after)
            drawdown = self.high_water_mark - tpv_after
            normalized_drawdown = drawdown / self.initial_cash
            drawdown_penalty = self.drawdown_penalty_weight * normalized_drawdown
            final_reward = pnl_component - drawdown_penalty
        
        obs = self._get_observation()
        self._final_eval_reward += raw_reward # Track true PnL for evaluation
        info = {'price': self.current_price, 'eval_episode_return': self._final_eval_reward}
        return BaseEnvTimestep(obs, final_reward, done, info)

    def _handle_open_action(self, action_name):
        # ... (unchanged)
        days_to_expiry = self.total_steps - self.current_step
        atm_price = round(self.current_price / self.strike_distance) * self.strike_distance
        trades_to_execute = []
        if 'STRADDLE' in action_name:
            if len(self.portfolio) > self.max_positions - 2: return
            direction = 'long' if 'LONG' in action_name else 'short'
            is_buy = (direction == 'long')
            mid_price_call, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'call')
            trades_to_execute.append({'type': 'call', 'direction': direction, 'strike_price': atm_price, 'entry_premium': self._get_option_price(mid_price_call, is_buy)})
            mid_price_put, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'put')
            trades_to_execute.append({'type': 'put', 'direction': direction, 'strike_price': atm_price, 'entry_premium': self._get_option_price(mid_price_put, is_buy)})
        elif 'ATM' in action_name:
            if len(self.portfolio) >= self.max_positions: return
            _, direction, type, strike_str = action_name.split('_')
            offset = int(strike_str.replace('ATM', ''))
            strike_price = atm_price + (offset * self.strike_distance)
            is_buy = (direction == 'LONG')
            mid_price, _, _ = self._get_option_details(self.current_price, strike_price, days_to_expiry, type.lower())
            trades_to_execute.append({'type': type.lower(), 'direction': direction.lower(), 'strike_price': strike_price, 'entry_premium': self._get_option_price(mid_price, is_buy)})
        for trade in trades_to_execute:
            self.portfolio.append({'type': trade['type'], 'direction': trade['direction'], 'entry_step': self.current_step, 'strike_price': trade['strike_price'], 'entry_premium': trade['entry_premium'], 'days_to_expiry': days_to_expiry})

    def _get_observation(self):
        # ... (unchanged)
        obs_vec = np.zeros(self.obs_vector_size, dtype=np.float32)
        obs_vec[0] = (self.current_price / self.start_price) - 1.0
        obs_vec[1] = (self.total_steps - self.current_step) / self.total_steps
        total_pnl = self._get_portfolio_value()
        obs_vec[2] = math.tanh(total_pnl / self.initial_cash)
        current_idx = 3
        atm_price = round(self.current_price / self.strike_distance) * self.strike_distance
        for i in range(self.max_positions):
            if i < len(self.portfolio):
                pos = self.portfolio[i]
                obs_vec[current_idx + 0] = 1.0
                obs_vec[current_idx + 1] = 1.0 if pos['type'] == 'call' else -1.0
                obs_vec[current_idx + 2] = 1.0 if pos['direction'] == 'long' else -1.0
                obs_vec[current_idx + 3] = (pos['strike_price'] - atm_price) / (3 * self.strike_distance)
                obs_vec[current_idx + 4] = (self.current_step - pos['entry_step']) / self.total_steps
                _, _, d2 = self._get_option_details(self.current_price, pos['strike_price'], pos['days_to_expiry'], pos['type'])
                pop = 0.0
                if pos['type'] == 'call': pop = norm.cdf(d2) if pos['direction'] == 'long' else 1 - norm.cdf(d2)
                else: pop = 1 - norm.cdf(d2) if pos['direction'] == 'long' else norm.cdf(d2)
                obs_vec[current_idx + 5] = pop
                max_profit, max_loss = self._calculate_max_profit_loss(pos)
                obs_vec[current_idx + 6] = math.tanh(max_profit / self.initial_cash)
                obs_vec[current_idx + 7] = math.tanh(max_loss / self.initial_cash)
            current_idx += 8
        if self.ignore_legal_actions: action_mask = np.ones(self.action_space_size, dtype=np.int8)
        else: # ... (full masking logic)
            pass
        return {'observation': obs_vec, 'action_mask': action_mask, 'to_play': np.array([-1], dtype=np.int8)}

    def _calculate_max_profit_loss(self, position):
        # ... (unchanged)
        entry_premium = position['entry_premium'] * self.lot_size
        if position['direction'] == 'long':
            max_profit = self.initial_cash * 10
            max_loss = -entry_premium
        else:
            max_profit = entry_premium
            max_loss = -self.initial_cash * 10
        return max_profit, max_loss

    def render(self, mode='human'):
        # ... (unchanged)
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
