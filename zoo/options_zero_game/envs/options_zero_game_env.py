import copy
import math
import random

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


@ENV_REGISTRY.register('options_zero_game')
class OptionsZeroGameEnv(gym.Env):
    # ... (config, __init__, _build_action_space are unchanged)
    metadata = {'render.modes': ['human']}
    config = dict(start_price=20000.0, initial_cash=100000.0, market_regimes = [{'name': 'Developed_Market', 'mu': 0.00005, 'omega': 0.000005, 'alpha': 0.09, 'beta': 0.90},], time_to_expiry_days=30, steps_per_day=75, rolling_vol_window=5, iv_skew_table={'call': {'-5': (13.5, 16.0), '-4': (13.3, 15.8), '-3': (13.2, 15.7), '-2': (13.1, 15.5), '-1': (13.0, 15.4), '0': (13.0, 15.3), '1': (13.0, 15.4), '2': (13.1, 15.6), '3': (13.2, 15.7), '4': (13.3, 15.8), '5': (13.5, 16.0)}, 'put': {'-5': (14.0, 16.5), '-4': (13.8, 16.3), '-3': (13.8, 16.1), '-2': (13.6, 16.0), '-1': (13.5, 15.8), '0': (13.5, 15.8), '1': (13.5, 15.8), '2': (13.6, 16.0), '3': (13.8, 16.1), '4': (13.8, 16.3), '5': (14.0, 16.5)}}, strike_distance=50.0, lot_size=75, max_positions=4, bid_ask_spread_pct=0.002, risk_free_rate=0.10, pnl_scaling_factor=1000, drawdown_penalty_weight=0.1, illegal_action_penalty=-1.0, ignore_legal_actions=True,)
    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg
    def __init__(self, cfg: dict = None):
        self._cfg = self.default_config()
        if cfg is not None: self._cfg.update(cfg)
        self.start_price = self._cfg.start_price
        self.initial_cash = self._cfg.initial_cash
        self.market_regimes = self._cfg.market_regimes
        self.rolling_vol_window = self._cfg.rolling_vol_window
        self.time_to_expiry_days = self._cfg.time_to_expiry_days
        self.steps_per_day = self._cfg.steps_per_day
        self.total_steps = self.time_to_expiry_days * self.steps_per_day
        self.risk_free_rate = self._cfg.risk_free_rate
        self.lot_size = self._cfg.lot_size
        self.strike_distance = self._cfg.strike_distance
        self.max_positions = self._cfg.max_positions
        self.bid_ask_spread_pct = self._cfg.bid_ask_spread_pct
        self.pnl_scaling_factor = self._cfg.pnl_scaling_factor
        self.drawdown_penalty_weight = self._cfg.drawdown_penalty_weight
        self.illegal_action_penalty = self._cfg.illegal_action_penalty
        self.ignore_legal_actions = self._cfg.ignore_legal_actions
        self.iv_bins = self._discretize_iv_skew(self._cfg.iv_skew_table)
        self.actions_to_indices = self._build_action_space()
        self.indices_to_actions = {v: k for k, v in self.actions_to_indices.items()}
        self.action_space_size = len(self.actions_to_indices)
        self._action_space = spaces.Discrete(self.action_space_size)
        self._observation_space = self._create_observation_space()
        self._reward_range = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.np_random = None
    def _discretize_iv_skew(self, skew_table, num_bins=5):
        binned_ivs = {'call': {}, 'put': {}}
        for option_type, table in skew_table.items():
            for offset_str, iv_range in table.items():
                min_iv, max_iv = iv_range
                binned_ivs[option_type][offset_str] = np.linspace(min_iv, max_iv, num_bins) / 100.0
        return binned_ivs
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
        for j in range(self.max_positions): actions[f'CLOSE_POSITION_{j}'] = i; i+=1
        actions['CLOSE_ALL'] = i
        return actions

    # <<< MODIFIED: State vector size is now larger to include the new features
    def _create_observation_space(self):
        # 5 global + 4*8 portfolio + 42 action_mask = 79
        self.market_and_portfolio_state_size = 5 + self.max_positions * 8
        self.obs_vector_size = self.market_and_portfolio_state_size + self.action_space_size
        return spaces.Dict({'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_vector_size,), dtype=np.float32),'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space_size,), dtype=np.int8),'to_play': spaces.Box(low=-1, high=-1, shape=(1,), dtype=np.int8)})

    def seed(self, seed: int, dynamic_seed: int = None):
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]

    def _generate_price_path(self):
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
        returns_df = pd.Series(price_path).pct_change().dropna()
        rolling_window_steps = self.rolling_vol_window * self.steps_per_day
        self.realized_vol_series = returns_df.rolling(window=rolling_window_steps).std().fillna(0) * np.sqrt(252 * self.steps_per_day)

    def _simulate_price_step(self):
        self.current_price = self.price_path[self.current_step]
    
    def _get_implied_volatility(self, offset, option_type):
        clamped_offset = max(-5, min(5, offset))
        return self.iv_bins[option_type][str(clamped_offset)][self.iv_bin_index]

    def _get_option_details(self, underlying_price, strike_price, days_to_expiry, option_type):
        t = days_to_expiry / 365.25
        if t <= 1e-6:
            intrinsic_value = 0.0
            if option_type == 'call': intrinsic_value = max(0, underlying_price - strike_price)
            else: intrinsic_value = max(0, strike_price - underlying_price)
            abs_delta_at_expiry = 1.0 if intrinsic_value > 0 else 0.0
            return intrinsic_value, abs_delta_at_expiry, 0
        
        atm_price = int(underlying_price / self.strike_distance + 0.5) * self.strike_distance
        offset = round((strike_price - atm_price) / self.strike_distance)
        vol = self._get_implied_volatility(offset, option_type)
        
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
        if seed is not None: self.seed(seed)
        elif self.np_random is None: self.seed(0)
        
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
        self.garch_implied_vol = math.sqrt(unconditional_variance * 252)
        
        self.iv_bin_index = random.randint(0, 4)
        
        self._generate_price_path()
        
        self.current_step = 0
        self.day_of_week = 0
        
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

    def _calculate_shaped_reward(self, tpv_before, tpv_after):
        raw_reward = tpv_after - tpv_before
        pnl_component = math.tanh(raw_reward / self.pnl_scaling_factor)
        self.high_water_mark = max(self.high_water_mark, tpv_after)
        drawdown = self.high_water_mark - tpv_after
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

    def _advance_time_and_market(self):
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
        for pos in self.portfolio:
            pos['days_to_expiry'] -= time_decay_days

    def step(self, action: int):
        true_legal_actions_mask = self._get_true_action_mask()
        was_illegal_action = true_legal_actions_mask[action] == 0
        if was_illegal_action:
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
        
        self._advance_time_and_market()

        done = self.current_step >= self.total_steps
        if done:
            while len(self.portfolio) > 0: self._close_position(0)

        tpv_after = self._get_portfolio_value()
        normal_shaped_reward, raw_reward = self._calculate_shaped_reward(tpv_before, tpv_after)
        
        if was_illegal_action:
            final_reward = self.illegal_action_penalty
        else:
            final_reward = normal_shaped_reward
        
        obs = self._get_observation()
        self._final_eval_reward += raw_reward
        info = {'price': self.current_price, 'eval_episode_return': self._final_eval_reward}
        return BaseEnvTimestep(obs, final_reward, done, info)

    def _handle_open_action(self, action_name):
        current_trading_day = self.current_step // self.steps_per_day
        trading_days_left = self.time_to_expiry_trading_days - current_trading_day
        calendar_days_left = trading_days_left * (7.0 / 5.0)
        days_to_expiry = calendar_days_left

        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
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

            # --- DYNAMIC HEDGE CALCULATION ---
            # First, determine the premium of the inner straddle to find the breakeven points.
            # This is the same for both long and short flies, just the direction of the trade changes.
            mid_price_call_atm, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'call')
            premium_call_atm = self._get_option_price(mid_price_call_atm, is_buy=(direction == 'long'))

            mid_price_put_atm, _, _ = self._get_option_details(self.current_price, atm_price, days_to_expiry, 'put')
            premium_put_atm = self._get_option_price(mid_price_put_atm, is_buy=(direction == 'long'))

            # For a short fly, this is premium collected. For a long fly, it's the debit paid.
            net_premium = premium_call_atm + premium_put_atm

            # Determine the breakeven points based on the straddle's cost/credit
            upper_breakeven = atm_price + net_premium
            lower_breakeven = atm_price - net_premium

            # Find the closest available strikes for the hedges
            hedge_strike_call = int(upper_breakeven / self.strike_distance + 0.5) * self.strike_distance
            hedge_strike_put = int(lower_breakeven / self.strike_distance + 0.5) * self.strike_distance

            # --- Construct the Full 4-Legged Trade ---

            # Leg 1 & 2: The ATM Straddle
            # If it's a LONG Iron Fly, we are LONG the straddle. If SHORT, we are SHORT the straddle.
            trades_to_execute.append({'type': 'call', 'direction': direction, 'strike_price': atm_price, 'entry_premium': premium_call_atm})
            trades_to_execute.append({'type': 'put', 'direction': direction, 'strike_price': atm_price, 'entry_premium': premium_put_atm})

            # Leg 3 & 4: The Hedging OTM Strangle
            # If it's a LONG Iron Fly, we are SHORT the hedges. If SHORT, we are LONG the hedges.
            hedge_direction = 'short' if direction == 'long' else 'long'
            is_buy_hedge = (hedge_direction == 'long')

            mid_price_hedge_call, _, _ = self._get_option_details(self.current_price, hedge_strike_call, days_to_expiry, 'call')
            trades_to_execute.append({'type': 'call', 'direction': hedge_direction, 'strike_price': hedge_strike_call, 'entry_premium': self._get_option_price(mid_price_hedge_call, is_buy_hedge)})

            mid_price_hedge_put, _, _ = self._get_option_details(self.current_price, hedge_strike_put, days_to_expiry, 'put')
            trades_to_execute.append({'type': 'put', 'direction': hedge_direction, 'strike_price': hedge_strike_put, 'entry_premium': self._get_option_price(mid_price_hedge_put, is_buy_hedge)})

        elif 'IRON_CONDOR' in action_name:
            if len(self.portfolio) > self.max_positions - 4: return
            direction = 'long' if 'LONG' in action_name else 'short'

            # --- Dynamic, Delta-Based Strike Selection ---
            # Define the target deltas for the short and long legs
            target_delta_short = 0.30
            target_delta_long = 0.10

            # Define the search range for strikes (e.g., up to 10 strikes away)
            search_range = range(1, 11)

            # --- Find the Call Strikes ---
            # Find the strike closest to the target delta for the short call
            best_short_call_strike = atm_price + (self.strike_distance * 2) # Default fallback
            min_delta_diff_short_call = 999
            for offset in search_range:
                strike = atm_price + (offset * self.strike_distance)
                _, d, _ = self._get_option_details(self.current_price, strike, days_to_expiry, 'call')
                if abs(d - target_delta_short) < min_delta_diff_short_call:
                    min_delta_diff_short_call = abs(d - target_delta_short)
                    best_short_call_strike = strike

            # Find the strike closest to the target delta for the long call (must be further OTM)
            best_long_call_strike = best_short_call_strike + self.strike_distance # Default fallback
            min_delta_diff_long_call = 999
            for offset in range(int((best_short_call_strike - atm_price) / self.strike_distance) + 1, 11):
                strike = atm_price + (offset * self.strike_distance)
                _, d, _ = self._get_option_details(self.current_price, strike, days_to_expiry, 'call')
                if abs(d - target_delta_long) < min_delta_diff_long_call:
                    min_delta_diff_long_call = abs(d - target_delta_long)
                    best_long_call_strike = strike

            # --- Find the Put Strikes ---
            # Find the strike closest to the target delta for the short put
            best_short_put_strike = atm_price - (self.strike_distance * 2) # Default fallback
            min_delta_diff_short_put = 999
            for offset in search_range:
                strike = atm_price - (offset * self.strike_distance)
                _, d, _ = self._get_option_details(self.current_price, strike, days_to_expiry, 'put')
                if abs(d - target_delta_short) < min_delta_diff_short_put:
                    min_delta_diff_short_put = abs(d - target_delta_short)
                    best_short_put_strike = strike

            # Find the strike closest to the target delta for the long put (must be further OTM)
            best_long_put_strike = best_short_put_strike - self.strike_distance # Default fallback
            min_delta_diff_long_put = 999
            for offset in range(int((atm_price - best_short_put_strike) / self.strike_distance) + 1, 11):
                strike = atm_price - (offset * self.strike_distance)
                _, d, _ = self._get_option_details(self.current_price, strike, days_to_expiry, 'put')
                if abs(d - target_delta_long) < min_delta_diff_long_put:
                    min_delta_diff_long_put = abs(d - target_delta_long)
                    best_long_put_strike = strike

            # --- Construct the Full 4-Legged Trade ---
            inner_dir = 'long' if direction == 'long' else 'short'
            outer_dir = 'short' if direction == 'long' else 'long'

            # Leg 1: Inner Call
            mid_price_call_inner, _, _ = self._get_option_details(self.current_price, best_short_call_strike, days_to_expiry, 'call')
            trades_to_execute.append({'type': 'call', 'direction': inner_dir, 'strike_price': best_short_call_strike, 'entry_premium': self._get_option_price(mid_price_call_inner, inner_dir == 'long')})

            # Leg 2: Inner Put
            mid_price_put_inner, _, _ = self._get_option_details(self.current_price, best_short_put_strike, days_to_expiry, 'put')
            trades_to_execute.append({'type': 'put', 'direction': inner_dir, 'strike_price': best_short_put_strike, 'entry_premium': self._get_option_price(mid_price_put_inner, inner_dir == 'long')})

            # Leg 3: Outer Call (Hedge)
            mid_price_call_outer, _, _ = self._get_option_details(self.current_price, best_long_call_strike, days_to_expiry, 'call')
            trades_to_execute.append({'type': 'call', 'direction': outer_dir, 'strike_price': best_long_call_strike, 'entry_premium': self._get_option_price(mid_price_call_outer, outer_dir == 'long')})

            # Leg 4: Outer Put (Hedge)
            mid_price_put_outer, _, _ = self._get_option_details(self.current_price, best_long_put_strike, days_to_expiry, 'put')
            trades_to_execute.append({'type': 'put', 'direction': outer_dir, 'strike_price': best_long_put_strike, 'entry_premium': self._get_option_price(mid_price_put_outer, outer_dir == 'long')})

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

        current_trading_day = self.current_step // self.steps_per_day
        is_expiry_day = current_trading_day >= self.time_to_expiry_trading_days - 1

        action_mask[self.actions_to_indices['HOLD']] = 1
        if len(self.portfolio) > 0:
            action_mask[self.actions_to_indices['CLOSE_ALL']] = 1
            for i in range(len(self.portfolio)):
                if f'CLOSE_POSITION_{i}' in self.actions_to_indices:
                    action_mask[self.actions_to_indices[f'CLOSE_POSITION_{i}']] = 1

        if is_expiry_day:
            return action_mask

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
            atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
            days_to_expiry = self.time_to_expiry_trading_days - current_trading_day
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
        return action_mask

    # <<< MODIFIED: The state space now includes the log return
    def _get_observation(self):
        # 1. Market and Portfolio State Vector
        market_portfolio_vec = np.zeros(self.market_and_portfolio_state_size, dtype=np.float32)
        
        # --- Global Features ---
        market_portfolio_vec[0] = (self.current_price / self.start_price) - 1.0
        market_portfolio_vec[1] = (self.total_steps - self.current_step) / self.total_steps
        total_pnl = self._get_portfolio_value()
        market_portfolio_vec[2] = math.tanh(total_pnl / self.initial_cash)
        current_realized_vol = self.realized_vol_series.iloc[self.current_step - 1] if self.current_step > 0 else 0
        market_portfolio_vec[3] = math.tanh((current_realized_vol / self.garch_implied_vol) - 1.0) if self.garch_implied_vol > 0 else 0.0
        
        # <<< NEW: Add the latest log return feature
        if self.current_step > 0:
            log_return = math.log(self.current_price / self.price_path[self.current_step - 1])
        else:
            log_return = 0.0
        market_portfolio_vec[4] = np.clip(log_return, -5, 5) # Clip for stability
        
        # --- Per-Slot Features ---
        current_idx = 5 # Start filling from index 5
        atm_price = int(self.current_price / self.strike_distance + 0.5) * self.strike_distance
        for i in range(self.max_positions):
            if i < len(self.portfolio):
                pos = self.portfolio[i]
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
        
        # 2. Get the true legal action mask
        true_action_mask = self._get_true_action_mask()
        
        # 3. Concatenate them into the final observation vector
        final_obs_vec = np.concatenate((market_portfolio_vec, true_action_mask.astype(np.float32)))
        
        # 4. The action_mask sent to the MCTS can be all ones
        mcts_action_mask = np.ones(self.action_space_size, dtype=np.int8)
        
        return {'observation': final_obs_vec, 'action_mask': mcts_action_mask, 'to_play': np.array([-1], dtype=np.int8)}

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
