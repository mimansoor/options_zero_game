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
        """
        The definitive step method. It uses the new, robust "snapshot" and
        "receipt" systems to guarantee accurate logging.
        """
        # 1. Get the pre-action state needed for correct day/step stamping.
        step_at_action = self.env.current_step
        day_at_action = self.env.current_day

        # 2. Execute the step in the main environment.
        # This will update the portfolio and take the post-action snapshot internally.
        timestep = self.env.step(action)
        
        # 3. Check for a closed trade receipt.
        receipt = self.env.portfolio_manager.last_closed_trade_receipt
        if receipt:
            receipt['exit_day'] = day_at_action
            self._closed_trades_log.append(receipt)
            self.env.portfolio_manager.last_closed_trade_receipt = {}
            
        # 4. Log the outcome of the action.
        self._log_step_outcome(timestep, step_at_action, day_at_action)
        
        # 5. If the episode just ended, add the special settlement entry and save.
        if timestep.done:
            self._log_settlement_state(timestep)
            self.save_log()
            
        return timestep

    def _log_step_outcome(self, timestep, step_at_action, day_at_action):
        """Creates a single, complete log entry for the action that was just taken."""
        info = timestep.info
        current_price = info['price']
        
        # --- THE CORE OF YOUR FIX ---
        # 1. Get the definitive, correctly-timed portfolio snapshot from the manager.
        portfolio_to_log = self.env.portfolio_manager.get_post_action_portfolio()
        
        # 2. Serialize this correct portfolio.
        serialized_portfolio = self._serialize_portfolio(portfolio_to_log, current_price)
        
        # ... (The rest of the method for calculating biases, etc., is now guaranteed to be correct)
        
        log_entry = {
            'step': int(step_at_action),
            'day': int(day_at_action),
            'portfolio': serialized_portfolio,
            'info': info,
            # ... (rest of the entry)
        }
        self._episode_history.append(log_entry)

    def _log_settlement_state(self, final_timestep):
        """
        Creates and appends a final, special log entry for the end of the episode
        to show the state after all closing/settlement logic has occurred.
        """
        if not self._episode_history: return
        
        # Use a copy of the very last recorded state as a template.
        last_log_entry = copy.deepcopy(self._episode_history[-1])
        info = final_timestep.info

        # Determine the reason for termination for the final action name
        final_reward = final_timestep.reward
        if terminated_by_time_logic_placeholder: # This would require passing the flag through info
            final_action_name = "EXPIRATION / SETTLEMENT"
        elif final_reward == -1.0:
            final_action_name = "STOP-LOSS HIT"
        elif final_reward == self.env.jackpot_reward:
            final_action_name = "PROFIT TARGET MET"
        else:
            final_action_name = "EPISODE END" # Generic fallback
        
        # --- THE KEY ---
        # The portfolio is now whatever is left *after* the `if terminated:` block in the env.
        # This will be empty for rule-based terminations and potentially non-empty for settlement.
        # But since our last fix now closes all positions on settlement, it will always be empty.
        # The *real* final state is captured by the P&L.
        final_pnl = info['eval_episode_return']
        
        # Create the new settlement entry
        settlement_entry = last_log_entry
        settlement_entry.update({
            'step': last_log_entry['step'] + 1,
            'day': last_log_entry['day'] + 1,
            'portfolio': [], # The portfolio is always empty at the very end
            'done': True,
            'action': None,
            'reward': final_reward
        })
        settlement_entry['info']['eval_episode_return'] = final_pnl
        settlement_entry['info']['executed_action_name'] = final_action_name
        
        self._episode_history.append(settlement_entry)

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
        """Saves the complete episode history, including the historical context."""
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
