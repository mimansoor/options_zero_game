# zoo/options_zero_game/envs/log_replay_env.py
# <<< VERSION 2.1 - REFACTORED AND CORRECTED LOGIC >>>

import json
import gymnasium as gym
from easydict import EasyDict
import copy
import numpy as np

from ding.utils import ENV_REGISTRY
from .options_zero_game_env import OptionsZeroGameEnv
from ding.envs.env.base_env import BaseEnvTimestep

@ENV_REGISTRY.register('log_replay')
class LogReplayEnv(gym.Wrapper):
    """
    A gym.Wrapper that logs all interactions and saves them to a JSON file.
    This version is updated to work with the refactored, modular OptionsZeroGameEnv
    and correctly logs 'HOLD' for illegal actions.
    """
    
    def __init__(self, cfg: dict):
        # Create an instance of our (now refactored) base environment
        base_env = OptionsZeroGameEnv(cfg)
        super().__init__(base_env)
        
        self.log_file_path = cfg.log_file_path
        self._episode_history = []
        self._last_price = None
        print(f"LogReplayEnv initialized. Replay will be saved to: {self.log_file_path}")

    # --- Standard Wrapper Methods ---
    def seed(self, seed: int, dynamic_seed: int = None):
        return self.env.seed(seed, dynamic_seed)

    def reset(self, **kwargs):
        self._episode_history = []
        obs = self.env.reset(**kwargs)
        # Initialize the last price from the price manager
        self._last_price = self.env.price_manager.start_price
        return obs

    def step(self, action):
        # Log the state of the environment *before* the action is taken.
        # This is crucial for the replayer to show the state on which the decision was based.
        if self.env.current_step == 0:
            self._log_step(is_initial_state=True)

        # Execute the step in the main environment. It will return an info dict
        # telling us if the action was illegal.
        timestep = self.env.step(action)
        
        # Log the outcome of the step, passing along the original action and the full info dict.
        self._log_step(action=action, reward=timestep.reward, done=timestep.done, info=timestep.info)
        
        if timestep.done:
            self.save_log()
            
        return timestep

    def _log_step(self, action=None, reward=None, done=False, info=None, is_initial_state=False):
        """Builds and appends a single serializable log entry for the current state."""
        
        # Correctly access the portfolio from the portfolio manager
        portfolio_df = self.env.portfolio_manager.portfolio
        serializable_portfolio = []
        
        # Correctly get current price from the price manager
        current_price = self.env.price_manager.current_price
        
        # Re-calculate live PnL for each position for accurate logging
        for index, pos in portfolio_df.iterrows():
            is_call = pos['type'] == 'call'
            atm_price = self.env.market_rules_manager.get_atm_price(current_price)
            offset = round((pos['strike_price'] - atm_price) / self.env._cfg.strike_distance)
            
            vol = self.env.market_rules_manager.get_implied_volatility(offset, pos['type'], self.env.iv_bin_index)
            greeks = self.env.bs_manager.get_all_greeks_and_price(current_price, pos['strike_price'], pos['days_to_expiry'], vol, is_call)
            
            mid_price = greeks['price']
            current_premium = self.env.bs_manager.get_price_with_spread(mid_price, is_buy=(pos['direction'] == 'short'), bid_ask_spread_pct=self.env._cfg.bid_ask_spread_pct)
            
            pnl_multiplier = 1 if pos['direction'] == 'long' else -1
            pnl = (current_premium - pos['entry_premium']) * self.env.portfolio_manager.lot_size * pnl_multiplier
                
            serializable_portfolio.append({
                'type': pos['type'], 'direction': pos['direction'],
                'strike_price': round(pos['strike_price'], 2),
                'entry_premium': round(pos['entry_premium'], 2),
                'days_to_expiry': round(pos['days_to_expiry'], 2),
                'current_premium': round(current_premium, 2),
                'live_pnl': round(pnl, 2),
            })

        log_info = info if info is not None else {}
        
        # Prioritize the 'executed_action_name' from the info dict if it exists.
        # This correctly handles step 0 overrides and illegal action conversions.
        if 'executed_action_name' in log_info:
            log_info['action_name'] = log_info['executed_action_name']
        else:
            # Fallback for the initial state log, which has no action
            log_info['action_name'] = self.env.indices_to_actions.get(action, 'N/A')
       
        # Calculate last price change for the visualizer
        last_price_change_pct = ((current_price / self._last_price) - 1) * 100 if self._last_price and self._last_price > 0 else 0.0
        self._last_price = current_price
        
        # Populate the rest of the info dict by correctly accessing the managers
        log_info['last_price_change_pct'] = last_price_change_pct
        log_info['price'] = float(current_price)
        log_info['eval_episode_return'] = float(self._get_safe_pnl())
        log_info['start_price'] = float(self.env.price_manager.start_price)
        log_info['volatility'] = float(self.env.price_manager.garch_implied_vol)
        log_info['risk_free_rate'] = float(self.env.bs_manager.risk_free_rate)
        log_info['market_regime'] = str(self.env.price_manager.current_regime_name)
        log_info['illegal_actions_in_episode'] = int(self.env.illegal_action_count)

        log_entry = {
            'step': int(self.env.current_step),
            'day': int(self.env.current_step // self.env._cfg.steps_per_day + 1),
            'portfolio': serializable_portfolio,
            'action': int(action) if action is not None else None, # Log the agent's raw action index
            'reward': float(reward) if reward is not None else None,
            'done': bool(done),
            'info': log_info, # The info dict contains the true action name
        }
        
        self._episode_history.append(log_entry)

    def _get_safe_pnl(self) -> float:
        """Helper to get total PnL safely by calling the portfolio manager."""
        try:
            return self.env.portfolio_manager.get_total_pnl(self.env.price_manager.current_price, self.env.iv_bin_index)
        except Exception:
            return 0.0

    def save_log(self):
        print(f"Episode finished. Saving replay log with {len(self._episode_history)} steps...")
        try:
            with open(self.log_file_path, 'w') as f:
                json.dump(self._episode_history, f, indent=2)
            print(f"Successfully saved replay log to {self.log_file_path}")
        except Exception as e:
            print(f"Error saving replay log: {e}")

    # --- Static methods remain the same for framework compatibility ---
    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> list:
        return OptionsZeroGameEnv.create_collector_env_cfg(cfg)

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> list:
        return OptionsZeroGameEnv.create_evaluator_env_cfg(cfg)
