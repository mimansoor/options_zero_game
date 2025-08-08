# zoo/options_zero_game/envs/log_replay_env.py
# <<< DEFINITIVE VERSION with Correct Settlement Logging >>>

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
    """
    The definitive, stateful logging wrapper. It captures all rich data from the
    environment to create a complete replay log for the advanced visualizer.
    """
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
        The definitive step method that orchestrates the logging process.
        """
        # 1. Capture the pre-action state for accurate timestamping.
        step_at_action = self.env.current_step
        day_at_action = self.env.current_day

        # 2. Execute the step in the main environment. This updates the world
        #    and generates the necessary snapshots and receipts internally.
        timestep = self.env.step(action)
        
        # 3. Check for a closed trade receipt from the PortfolioManager.
        receipt = self.env.portfolio_manager.last_closed_trade_receipt
        if receipt:
            # The receipt is created when the position is closed, which is on the
            # current day of the action.
            receipt['exit_day'] = day_at_action
            self._closed_trades_log.append(receipt)
            # Clear the receipt after we've logged it.
            self.env.portfolio_manager.last_closed_trade_receipt = {}
            
        # 4. Log the outcome of the action that was just taken.
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
        
        # 1. Get the definitive, correctly-timed portfolio snapshot from the manager.
        portfolio_to_log = self.env.portfolio_manager.get_post_action_portfolio()
        
        # 2. Serialize this correct portfolio.
        serialized_portfolio = self._serialize_portfolio(portfolio_to_log, current_price)
        
        # 3. Add all enriched data to the info dict for logging.
        info['pnl_verification'] = self.env.portfolio_manager.get_pnl_verification(current_price, self.env.iv_bin_index)
        info['payoff_data'] = self.env.portfolio_manager.get_payoff_data(current_price, self.env.iv_bin_index)
        info['closed_trades_log'] = copy.deepcopy(self._closed_trades_log)

        log_entry = {
            'step': int(step_at_action),
            'day': int(day_at_action),
            'portfolio': serialized_portfolio,
            'info': info,
            'action': int(timestep.obs['action_mask'].sum()) if isinstance(timestep.obs, dict) else None,
            'reward': float(timestep.reward) if timestep.reward is not None else None,
            'done': bool(timestep.done),
        }
        self._episode_history.append(log_entry)

    def _log_settlement_state(self, final_timestep):
        """
        Creates and appends a final, special log entry that correctly reflects
        the zeroed-out state of a settled/closed portfolio.
        """
        if not self._episode_history: return
        
        last_log_entry = copy.deepcopy(self._episode_history[-1])
        info = final_timestep.info
        final_pnl = info['eval_episode_return']
        
        termination_reason = info.get('termination_reason', 'UNKNOWN')
        
        if termination_reason == "TIME_LIMIT": final_action_name = "EXPIRATION / SETTLEMENT"
        elif termination_reason == "STOP_LOSS": final_action_name = "STOP-LOSS HIT"
        elif termination_reason == "TAKE_PROFIT": final_action_name = "PROFIT TARGET MET"
        else: final_action_name = "EPISODE END"
        
        # --- THE FIX: Create a clean, zeroed-out info dict for the final state ---
        final_info = copy.deepcopy(info)
        # An empty portfolio has no bias and no risk.
        final_info['directional_bias'] = "Neutral"
        final_info['volatility_bias'] = "Neutral / Low Volatility Expected"
        final_info['executed_action_name'] = final_action_name
        final_info['eval_episode_return'] = final_pnl
        # All portfolio stats are now zero.
        final_info['portfolio_stats'] = {
            'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0,
            'max_profit': 0.0, 'max_loss': 0.0, 'rr_ratio': 0.0, 'prob_profit': 1.0 if final_pnl > 0 else 0.0
        }
        # The P&L is now fully realized.
        final_info['pnl_verification'] = {
            'realized_pnl': final_pnl, 'unrealized_pnl': 0.0, 'verified_total_pnl': final_pnl
        }
        # Payoff diagram is flat for an empty portfolio.
        final_info['payoff_data'] = {'expiry_pnl': [], 'current_pnl': [], 'spot_price': info['price'], 'sigma_levels': {}}

        settlement_entry = {
            'step': last_log_entry['step'] + 1,
            'day': last_log_entry['day'] + 1,
            'portfolio': [],
            'done': True,
            'action': None,
            'reward': final_timestep.reward,
            'info': final_info
        }
        
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
        """Saves the complete episode history to a JSON file."""
        print(f"Episode finished. Saving replay log with {len(self._episode_history)} steps...")
        try:
            with open(self.log_file_path, 'w') as f:
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
