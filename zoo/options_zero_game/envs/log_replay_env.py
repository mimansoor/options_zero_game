# zoo/options_zero_game/envs/log_replay_env.py
# <<< DEFINITIVE, FINAL, AND CORRECT VERSION >>>

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
        # The first log entry is created after the first step, not on reset.
        return self.env.reset(**kwargs)

    def step(self, action):
        # --- THE FIX IS HERE: The logic is now in the correct order ---
        # 1. Get the portfolio state *before* the action is taken.
        portfolio_before = self.env.portfolio_manager.portfolio.copy()

        # 2. Execute the step in the main environment.
        timestep = self.env.step(action)

        # 3. Get the portfolio state *after* the action.
        portfolio_after = self.env.portfolio_manager.portfolio

        # 4. Compare the portfolios to find and log any newly closed trades.
        self._update_closed_trades_log(portfolio_before, portfolio_after, self.env.current_day)

        # 5. Log the complete state that resulted from the action.
        self._log_step_outcome(timestep)

        # 6. If the episode just ended, save the complete log.
        if timestep.done:
            self.save_log()

        return timestep

    def _update_closed_trades_log(self, before_df, after_df, exit_day):
        if before_df.empty: return
        closed_ids = set(before_df['creation_id']) - set(after_df['creation_id'])
        if not closed_ids: return

        pnl_change_this_step = self.env.portfolio_manager.realized_pnl - self._realized_pnl_at_last_step
        pnl_per_closed_leg = pnl_change_this_step / len(closed_ids) if closed_ids else 0

        for trade_id in closed_ids:
            closed_trade = before_df[before_df['creation_id'] == trade_id].iloc[0]
            pnl_multiplier = 1 if closed_trade['direction'] == 'long' else -1
            exit_premium = closed_trade['entry_premium'] + ((pnl_per_closed_leg / self.env.portfolio_manager.lot_size) * pnl_multiplier)

            self._closed_trades_log.append({
                'position': f"{closed_trade['direction'].upper()} {closed_trade['type'].upper()}",
                'strike': closed_trade['strike_price'],
                'entry_day': closed_trade['entry_step'] // self.env.steps_per_day + 1,
                'exit_day': exit_day,
                'entry_prem': closed_trade['entry_premium'],
                'exit_prem': exit_premium,
                'realized_pnl': pnl_per_closed_leg,
            })
        self._realized_pnl_at_last_step = self.env.portfolio_manager.realized_pnl

    def _log_step_outcome(self, timestep):
        info = timestep.info
        current_price = info['price']

        info['pnl_verification'] = self.env.portfolio_manager.get_pnl_verification(current_price, self.env.iv_bin_index)
        info['payoff_data'] = self.env.portfolio_manager.get_payoff_data(current_price)
        info['closed_trades_log'] = copy.deepcopy(self._closed_trades_log)

        serialized_portfolio = self._serialize_portfolio(self.env.portfolio_manager.portfolio, current_price)

        log_entry = {
            'step': int(self.env.current_step),
            'day': self.env.current_day,
            'portfolio': serialized_portfolio,
            'action': int(timestep.obs['action_mask'].sum()) if isinstance(timestep.obs, dict) else None,
            'reward': float(timestep.reward) if timestep.reward is not None else None,
            'done': bool(timestep.done),
            'info': info,
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
        try:
            with open(self.log_file_path, 'w') as f:
                # Use the NumpyEncoder to prevent serialization errors
                json.dump(self._episode_history, f, indent=2, cls=NumpyEncoder)
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
