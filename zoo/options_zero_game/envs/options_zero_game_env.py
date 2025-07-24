import copy
import math
import random

import gym
import numpy as np
from arch import arch_model
from easydict import EasyDict
from gym import spaces
from gym.utils import seeding
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta
from scipy.stats import norm
from collections import namedtuple

try:
    from ding.envs.env.base_env import BaseEnvTimestep
    from ding.utils import ENV_REGISTRY
except ImportError:
    BaseEnvTimestep = namedtuple('BaseEnvTimestep', ['obs', 'reward', 'done', 'info'])
    class DummyRegistry:
        def register(self, name):
            def decorator(cls):
                return cls
            return decorator
    ENV_REGISTRY = DummyRegistry()


@ENV_REGISTRY.register('options_zero_game')
class OptionsZeroGameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    config = dict(
        start_price=100.0,
        initial_cash=100000.0,
        market_regimes = [
            {'name': 'Developed_Market', 'mu': 0.00005, 'omega': 0.000005, 'alpha': 0.09, 'beta': 0.90},
        ],
        time_to_expiry=30,
        strike_distance=1.0,
        lot_size=100,
        max_positions=4,
        bid_ask_spread_pct=0.01,
        risk_free_rate=0.05,
        pnl_scaling_factor=1000,
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

        self.start_price = self._cfg.start_price
        self.initial_cash = self._cfg.initial_cash
        self.market_regimes = self._cfg.market_regimes
        self.trend = 0.0
        self.volatility = 0.20
        
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

    def _create_observation_space(self):
        self.obs_vector_size = 3 + self.max_positions * 8
        return spaces.Dict({'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_vector_size,), dtype=np.float32),'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space_size,), dtype=np.int8),'to_play': spaces.Box(low=-1, high=-1, shape=(1,), dtype=np.int8)})

    def seed(self, seed: int, dynamic_seed: int = None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _generate_price_path(self):
        garch_spec = arch_model(None, p=1, q=1)
        params = np.array([self.trend, self.omega, self.alpha, self.beta])
        sim_returns = garch_spec.simulate(params, self.total_steps + 1)
        price_path = np.zeros(self.total_steps + 1)
        price_path[0] = self.start_price
        for i in range(1, self.total_steps + 1):
            price_path[i] = price_path[i-1] * (1 + sim_returns['data'][i-1] / 100)
        self.price_path = price_path

    def _simulate_price_step(self):
        self.current_price = self.price_path[self.current_step]
    
    def _get_option_details(self, underlying_price, strike_price, days_to_expiry, option_type):
        t = days_to_expiry / 365.25
        
        if t <= 0:
            intrinsic_value = 0.0
            if option_type == 'call':
                intrinsic_value = max(0, underlying_price - strike_price)
            else: # put
                intrinsic_value = max(0, strike_price - underlying_price)
            
            # At expiry, delta is either 0 or 1 (or -1 for puts), and d2 is effectively +/- infinity.
            # We return simplified, representative values.
            delta_at_expiry = 0.0
            if intrinsic_value > 0:
                delta_at_expiry = 1.0 if option_type == 'call' else -1.0

            return intrinsic_value, abs(delta_at_expiry), 0 # d2 is not well-defined, return 0

        # If not expired, proceed with Black-Scholes
        vol = max(self.volatility, 1e-6)
        try:
            d1 = (math.log(underlying_price / strike_price) + (self.risk_free_rate + 0.5 * vol ** 2) * t) / (vol * math.sqrt(t))
            d2 = d1 - vol * math.sqrt(t)
            d = delta(option_type[0], underlying_price, strike_price, t, self.risk_free_rate, vol)
            mid_price = black_scholes(option_type[0], underlying_price, strike_price, t, self.risk_free_rate, vol)
            return mid_price, abs(d), d2
        except (ValueError, ZeroDivisionError):
            return 0, 0, 0

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
        chosen_regime = random.choice(self.market_regimes)
        self.trend = chosen_regime['mu']
        self.omega = chosen_regime['omega']
        self.alpha = chosen_regime['alpha']
        self.beta = chosen_regime['beta']
        
        if self.alpha + self.beta >= 1.0:
            total = self.alpha + self.beta
            self.alpha = self.alpha / (total + 0.01)
            self.beta = self.beta / (total + 0.01)

        unconditional_variance = self.omega / (1 - self.alpha - self.beta)
        self.volatility = math.sqrt(unconditional_variance * 252)
        
        self._generate_price_path()
        
        self.current_step = 0
        self.current_price = self.price_path[0]
        self.portfolio = []
        self.realized_pnl = 0.0
        self._final_eval_reward = 0.0
        self.high_water_mark = 0.0
        return self._get_observation()

    def _close_position(self, position_index):
        if position_index < len(self.portfolio):
            pos_to_close = self.portfolio.pop(position_index)
            mid_price, _, _ = self._get_option_details(self.current_price, pos_to_close['strike_price'], pos_to_close['days_to_expiry'], pos_to_close['type'])
            exit_premium = self._get_option_price(mid_price, is_buy=(pos_to_close['direction'] == 'short'))
            entry_premium = pos_to_close['entry_premium']
            if pos_to_close['direction'] == 'long': pnl = (exit_premium - entry_premium) * self.lot_size
            else: pnl = (entry_premium - exit_premium) * self.lot_size
            self.realized_pnl += pnl

    def step(self, action: int):
        true_legal_actions_mask = self._get_true_action_mask()
        legal_action_indices = np.where(true_legal_actions_mask == 1)[0]

        if action not in legal_action_indices:
            action = self.actions_to_indices['HOLD']

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
        
        self.current_step += 1
        self._simulate_price_step()

        for pos in self.portfolio: pos['days_to_expiry'] -= 1

        tpv_after = self._get_portfolio_value()
        raw_reward = tpv_after - tpv_before
        pnl_component = math.tanh(raw_reward / self.pnl_scaling_factor)
        self.high_water_mark = max(self.high_water_mark, tpv_after)
        drawdown = self.high_water_mark - tpv_after
        normalized_drawdown = drawdown / self.initial_cash
        drawdown_penalty = self.drawdown_penalty_weight * normalized_drawdown
        final_reward = pnl_component - drawdown_penalty

        done = self.current_step >= self.total_steps
        if done:
            while len(self.portfolio) > 0: self._close_position(0)
            tpv_after = self._get_portfolio_value()
            raw_reward = tpv_after - tpv_before
            pnl_component = math.tanh(raw_reward / self.pnl_scaling_factor)
            self.high_water_mark = max(self.high_water_mark, tpv_after)
            drawdown = self.high_water_mark - tpv_after
            normalized_drawdown = drawdown / self.initial_cash
            drawdown_penalty = self.drawdown_penalty_weight * normalized_drawdown
            final_reward = pnl_component - drawdown_penalty
        
        obs = self._get_observation()
        self._final_eval_reward += raw_reward
        info = {'price': self.current_price, 'eval_episode_return': self._final_eval_reward}
        return BaseEnvTimestep(obs, final_reward, done, info)

    def _handle_open_action(self, action_name):
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
        
        elif 'STRANGLE' in action_name:
            if len(self.portfolio) > self.max_positions - 2: return
            direction = 'long' if 'LONG' in action_name else 'short'
            is_buy = (direction == 'long')
            call_strike = atm_price + (1 * self.strike_distance)
            mid_price_call, _, _ = self._get_option_details(self.current_price, call_strike, days_to_expiry, 'call')
            trades_to_execute.append({'type': 'call', 'direction': direction, 'strike_price': call_strike, 'entry_premium': self._get_option_price(mid_price_call, is_buy)})
            put_strike = atm_price - (1 * self.strike_distance)
            mid_price_put, _, _ = self._get_option_details(self.current_price, put_strike, days_to_expiry, 'put')
            trades_to_execute.append({'type': 'put', 'direction': direction, 'strike_price': put_strike, 'entry_premium': self._get_option_price(mid_price_put, is_buy)})

        elif 'IRON_FLY' in action_name:
            if len(self.portfolio) > self.max_positions - 4: return
            direction = 'long' if 'LONG' in action_name else 'short'
            straddle_dir = 'long' if direction == 'long' else 'short'
            strangle_dir = 'short' if direction == 'long' else 'long'
            mid_price_call_atm, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'call')
            trades_to_execute.append({'type': 'call', 'direction': straddle_dir, 'strike_price': atm_price, 'entry_premium': self._get_option_price(mid_price_call_atm, straddle_dir == 'long')})
            mid_price_put_atm, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'put')
            trades_to_execute.append({'type': 'put', 'direction': straddle_dir, 'strike_price': atm_price, 'entry_premium': self._get_option_price(mid_price_put_atm, straddle_dir == 'long')})
            call_strike_otm = atm_price + (2 * self.strike_distance)
            put_strike_otm = atm_price - (2 * self.strike_distance)
            mid_price_call_otm, _, _ = self._get_option_details(self.current_price, call_strike_otm, days_to_expiry, 'call')
            trades_to_execute.append({'type': 'call', 'direction': strangle_dir, 'strike_price': call_strike_otm, 'entry_premium': self._get_option_price(mid_price_call_otm, strangle_dir == 'long')})
            mid_price_put_otm, _, _ = self._get_option_details(self.current_price, put_strike_otm, days_to_expiry, 'put')
            trades_to_execute.append({'type': 'put', 'direction': strangle_dir, 'strike_price': put_strike_otm, 'entry_premium': self._get_option_price(mid_price_put_otm, strangle_dir == 'long')})

        elif 'IRON_CONDOR' in action_name:
            if len(self.portfolio) > self.max_positions - 4: return
            direction = 'long' if 'LONG' in action_name else 'short'
            inner_dir = 'long' if direction == 'long' else 'short'
            outer_dir = 'short' if direction == 'long' else 'long'
            call_strike_inner = atm_price + (2 * self.strike_distance)
            put_strike_inner = atm_price - (2 * self.strike_distance)
            mid_price_call_inner, _, _ = self._get_option_details(self.current_price, call_strike_inner, days_to_expiry, 'call')
            trades_to_execute.append({'type': 'call', 'direction': inner_dir, 'strike_price': call_strike_inner, 'entry_premium': self._get_option_price(mid_price_call_inner, inner_dir == 'long')})
            mid_price_put_inner, _, _ = self._get_option_details(self.current_price, put_strike_inner, days_to_expiry, 'put')
            trades_to_execute.append({'type': 'put', 'direction': inner_dir, 'strike_price': put_strike_inner, 'entry_premium': self._get_option_price(mid_price_put_inner, inner_dir == 'long')})
            call_strike_outer = atm_price + (3 * self.strike_distance)
            put_strike_outer = atm_price - (3 * self.strike_distance)
            mid_price_call_outer, _, _ = self._get_option_details(self.current_price, call_strike_outer, days_to_expiry, 'call')
            trades_to_execute.append({'type': 'call', 'direction': outer_dir, 'strike_price': call_strike_outer, 'entry_premium': self._get_option_price(mid_price_call_outer, outer_dir == 'long')})
            mid_price_put_outer, _, _ = self._get_option_details(self.current_price, put_strike_outer, days_to_expiry, 'put')
            trades_to_execute.append({'type': 'put', 'direction': outer_dir, 'strike_price': put_strike_outer, 'entry_premium': self._get_option_price(mid_price_put_outer, outer_dir == 'long')})

        else:
            if len(self.portfolio) >= self.max_positions: return
            _, direction, type, strike_str = action_name.split('_')
            offset = int(strike_str.replace('ATM', ''))
            strike_price = atm_price + (offset * self.strike_distance)
            is_buy = (direction == 'LONG')
            mid_price, _, _ = self._get_option_details(self.current_price, strike_price, days_to_expiry, type.lower())
            trades_to_execute.append({'type': type.lower(), 'direction': direction.lower(), 'strike_price': strike_price, 'entry_premium': self._get_option_price(mid_price, is_buy)})

        for trade in trades_to_execute:
            self.portfolio.append({'type': trade['type'], 'direction': trade['direction'], 'entry_step': self.current_step, 'strike_price': trade['strike_price'], 'entry_premium': trade['entry_premium'], 'days_to_expiry': days_to_expiry})

    def _get_true_action_mask(self):
        action_mask = np.zeros(self.action_space_size, dtype=np.int8)
        action_mask[self.actions_to_indices['HOLD']] = 1
        
        num_long_calls = sum(1 for p in self.portfolio if p['type'] == 'call' and p['direction'] == 'long')
        num_short_calls = sum(1 for p in self.portfolio if p['type'] == 'call' and p['direction'] == 'short')
        num_long_puts = sum(1 for p in self.portfolio if p['type'] == 'put' and p['direction'] == 'long')
        num_short_puts = sum(1 for p in self.portfolio if p['type'] == 'put' and p['direction'] == 'short')
        existing_positions = {(p['strike_price'], p['type']): p['direction'] for p in self.portfolio}
        
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
            atm_price = round(self.current_price / self.strike_distance) * self.strike_distance
            days_to_expiry = self.total_steps - self.current_step
            for offset in range(-3, 4):
                strike_price = atm_price + (offset * self.strike_distance)
                for option_type in ['call', 'put']:
                    _, d, _ = self._get_option_details(self.current_price, strike_price, days_to_expiry, option_type)
                    is_far_otm = d < 0.15
                    is_far_itm = d > 0.85
                    action_name_long = f'OPEN_LONG_{option_type.upper()}_ATM{"+" if offset >=0 else ""}{offset}'
                    is_legal_long = not (is_far_otm or is_far_itm)
                    if option_type == 'put' and existing_positions.get((strike_price, 'call')) == 'short': is_legal_long = False
                    if option_type == 'call' and existing_positions.get((strike_price, 'put')) == 'short': is_legal_long = False
                    if option_type == 'call' and num_long_calls > 0: is_legal_long = False
                    if option_type == 'put' and num_long_puts > 0: is_legal_long = False
                    if is_legal_long: action_mask[self.actions_to_indices[action_name_long]] = 1
                    action_name_short = f'OPEN_SHORT_{option_type.upper()}_ATM{"+" if offset >=0 else ""}{offset}'
                    is_legal_short = not is_far_itm
                    if option_type == 'put' and existing_positions.get((strike_price, 'call')) == 'long': is_legal_short = False
                    if option_type == 'call' and existing_positions.get((strike_price, 'put')) == 'long': is_legal_short = False
                    if option_type == 'call' and num_short_calls > 0: is_legal_short = False
                    if option_type == 'put' and num_short_puts > 0: is_legal_short = False
                    if is_legal_short: action_mask[self.actions_to_indices[action_name_short]] = 1
        
        if len(self.portfolio) > 0:
            action_mask[self.actions_to_indices['CLOSE_ALL']] = 1
        for i in range(len(self.portfolio)):
            action_mask[self.actions_to_indices[f'CLOSE_POSITION_{i}']] = 1
            
        return action_mask

    def _get_observation(self):
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
        
        action_mask = np.ones(self.action_space_size, dtype=np.int8)
        
        return {'observation': obs_vec, 'action_mask': action_mask, 'to_play': np.array([-1], dtype=np.int8)}

    def _calculate_max_profit_loss(self, position):
        entry_premium = position['entry_premium'] * self.lot_size
        if position['direction'] == 'long':
            max_profit = self.initial_cash * 10
            max_loss = -entry_premium
        else:
            max_profit = entry_premium
            max_loss = -self.initial_cash * 10
        return max_profit, max_loss

    def render(self, mode='human'):
        portfolio_val = self._get_portfolio_value()
        print(f"Step: {self.current_step:02d} | Price: ${self.current_price:8.2f} | Positions: {len(self.portfolio):1d} | Total PnL: ${portfolio_val:9.2f}")
    
    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space
    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space
    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_range

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
