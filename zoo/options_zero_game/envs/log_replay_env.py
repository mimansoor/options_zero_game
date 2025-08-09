# zoo/options_zero_game/envs/log_replay_env.py
# <<< DEFINITIVE VERSION with Complete Two-Snapshot Logging >>>

import json
import copy
import gymnasium as gym
import pandas as pd
import numpy as np

from ding.utils import ENV_REGISTRY
from .options_zero_game_env import OptionsZeroGameEnv
from ..entry.bias_meter import BiasMeter
from ding.envs.env.base_env import BaseEnvTimestep

# Helper class to handle any stray numpy types during JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NumpyEncoder, self).default(obj)

@ENV_REGISTRY.register('log_replay')
class LogReplayEnv(gym.Wrapper):
    def __init__(self, cfg: dict):
        base_env = OptionsZeroGameEnv(cfg)
        super().__init__(base_env)
        self.log_file_path = cfg.get('log_file_path', 'replay_log.json')
        self._episode_history = []
        self._closed_trades_log = []
        self._realized_pnl_at_last_step = 0.0
        print(f"LogReplayEnv initialized. Replay will be saved to: {self.log_file_path}")

    def reset(self, **kwargs):
        self._episode_history = []
        self._closed_trades_log = []
        self._realized_pnl_at_last_step = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        The definitive step method that logs the day in two halves: 
        1. The state immediately after the agent's action.
        2. The state at the end of the day, after the market has moved.
        """
        # --- 1. Capture Pre-Action State ---
        step_at_action = self.env.current_step
        day_at_action = self.env.current_day
        pre_action_price = self.env.price_manager.current_price
        equity_before = self.env.portfolio_manager.get_current_equity(pre_action_price, self.env.iv_bin_index)
        
        # --- 2. Execute Part 1: Take Action ---
        self.env._take_action_on_state(action)
        
        # Check for a closed trade receipt immediately after the action
        receipt = self.env.portfolio_manager.last_closed_trade_receipt
        if receipt:
            receipt['exit_day'] = day_at_action
            self._closed_trades_log.append(receipt)
            self.env.portfolio_manager.last_closed_trade_receipt = {}

        # --- 3. Log the "Post-Action" Snapshot ---
        self._log_state_snapshot(
            step_num=step_at_action, day_num=day_at_action, is_post_action=True,
            price_for_log=pre_action_price, action_info=self.env.last_action_info
        )

        # --- 4. Execute Part 2: Advance Market ---
        timestep = self.env._advance_market_and_get_outcome(equity_before)
        
        # --- 5. Log the "End-of-Day" Snapshot ---
        self._log_state_snapshot(
            step_num=step_at_action, day_num=day_at_action, is_post_action=False,
            price_for_log=self.env.price_manager.current_price, 
            action_info=self.env.last_action_info, final_timestep=timestep
        )

        # --- 6. Handle Episode End ---
        if timestep.done:
            self.save_log()
            
        return timestep

    def _log_state_snapshot(self, step_num, day_num, is_post_action, price_for_log, action_info, final_timestep=None):
        """
        The complete, definitive method to log a state snapshot.
        It can log either a post-action or an end-of-day state.
        """
        log_step = step_num
        log_day = day_num
        portfolio_to_log = self.env.portfolio_manager.get_post_action_portfolio()

        # --- 1. Determine the context of this log entry ---
        if is_post_action:
            executed_action_name = action_info['final_action_name']
            reward, done = None, False
            obs_for_bias = self.env._get_observation() # Get obs before market move
        else: # End-of-Day
            executed_action_name = "MARKET MOVE / EOD"
            reward, done = final_timestep.reward, final_timestep.done
            obs_for_bias = final_timestep.obs['observation']

        # --- 2. Gather all rich data for the log ---
        serialized_portfolio = self._serialize_portfolio(portfolio_to_log, price_for_log)
        pnl_verification = self.env.portfolio_manager.get_pnl_verification(price_for_log, self.env.iv_bin_index)
        payoff_data = self.env.portfolio_manager.get_payoff_data(price_for_log, self.env.iv_bin_index)
        
        meter = BiasMeter(obs_for_bias[:self.env.market_and_portfolio_state_size], self.env.OBS_IDX)

        # Determine the price from the previous log entry to calculate change
        last_price = self._episode_history[-1]['info']['price'] if self._episode_history else self.env.price_manager.start_price
        last_price_change_pct = ((price_for_log / last_price) - 1) * 100 if last_price > 0 else 0.0

        # --- 3. Build the final, consistent info dictionary ---
        info = {
            'price': float(price_for_log),
            'eval_episode_return': pnl_verification['verified_total_pnl'],
            'initial_cash': self.env.portfolio_manager.initial_cash,
            'last_price_change_pct': last_price_change_pct,
            'start_price': float(self.env.price_manager.start_price),
            'market_regime': str(self.env.price_manager.current_regime_name),
            'executed_action_name': executed_action_name,
            'directional_bias': meter.directional_bias,
            'volatility_bias': meter.volatility_bias,
            'portfolio_stats': self.env.portfolio_manager.get_raw_portfolio_stats(price_for_log, self.env.iv_bin_index),
            'pnl_verification': pnl_verification,
            'payoff_data': payoff_data,
            'closed_trades_log': copy.deepcopy(self._closed_trades_log),
        }
        
        # Add framework-specific info if it exists (only for EOD state)
        if final_timestep:
            info.update(final_timestep.info)

        log_entry = {
            'step': log_step,
            'day': log_day,
            'sub_step_name': "Post-Action" if is_post_action else "End-of-Day", # For the UI
            'portfolio': serialized_portfolio,
            'info': info,
            'reward': float(reward) if reward is not None else None,
            'done': bool(done),
        }
        self._episode_history.append(log_entry)

    def _serialize_portfolio(self, portfolio_df: pd.DataFrame, current_price: float) -> list:
        """Calculates live PnL for the current portfolio and returns a serializable list."""
        serializable_portfolio = []
        if portfolio_df is None or portfolio_df.empty:
            return serializable_portfolio
            
        for index, pos in portfolio_df.iterrows():
            is_call = pos['type'] == 'call'
            atm_price = self.env.market_rules_manager.get_atm_price(current_price)
            offset = round((pos['strike_price'] - atm_price) / self.env.strike_distance)
            vol = self.env.market_rules_manager.get_implied_volatility(offset, pos['type'], self.env.iv_bin_index)
            greeks = self.env.bs_manager.get_all_greeks_and_price(current_price, pos['strike_price'], pos['days_to_expiry'], vol, is_call)
            
            mid_price = greeks['price']
            current_premium = self.env.bs_manager.get_price_with_spread(mid_price, is_buy=(pos['direction'] == 'short'), bid_ask_spread_pct=self.env.bid_ask_spread_pct)
            
            pnl_multiplier = 1 if pos['direction'] == 'long' else -1
            pnl = (current_premium - pos['entry_premium']) * self.env.portfolio_manager.lot_size * pnl_multiplier
                
            serializable_portfolio.append({
                'type': pos['type'], 'direction': pos['direction'],
                'strike_price': round(pos['strike_price'], 2),
                'entry_premium': round(pos['entry_premium'], 2),
                'current_premium': round(current_premium, 2),
                'live_pnl': round(pnl, 2),
                'days_to_expiry': float(pos['days_to_expiry']),
            })
        return serializable_portfolio

    def save_log(self):
        """Saves the complete episode history to a JSON file."""
        print(f"Episode finished. Saving replay log with {len(self._episode_history)} steps...")

        # Create a final log object that contains the episode steps AND the historical data.
        final_log_object = {
            'historical_context': self.env.price_manager.historical_context_path.tolist(),
            'episode_data': self._episode_history
        }

        try:
            with open(self.log_file_path, 'w') as f:
                json.dump(final_log_object, f, indent=2, cls=NumpyEncoder)
            print(f"Successfully saved replay log to {self.log_file_path}")
        except Exception as e:
            print(f"Error saving replay log: {e}")

    # --- Static methods for framework compatibility ---
    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> list:
        return OptionsZeroGameEnv.create_collector_env_cfg(cfg)

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> list:
        return OptionsZeroGameEnv.create_evaluator_env_cfg(cfg)
