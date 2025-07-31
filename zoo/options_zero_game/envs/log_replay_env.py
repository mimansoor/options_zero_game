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
    This version is updated to work with the final, refactored OptionsZeroGameEnv.
    """
    
    def __init__(self, cfg: dict):
        base_env = OptionsZeroGameEnv(cfg)
        super().__init__(base_env)
        self.log_file_path = cfg.log_file_path
        self._episode_history = []
        self._last_price = None # <<< NEW: Track the last price for change calculation
        print(f"LogReplayEnv initialized. Replay will be saved to: {self.log_file_path}")

    def seed(self, seed: int, dynamic_seed: int = None):
        return self.env.seed(seed, dynamic_seed)

    def reset(self, **kwargs):
        self._episode_history = []
        obs = self.env.reset(**kwargs)
        self._last_price = self.env.start_price # Initialize last price
        return obs

    def step(self, action):
        # <<< MODIFIED: Check for illegal action *before* logging
        true_legal_actions_mask = self.env._get_true_action_mask()
        was_illegal_action = true_legal_actions_mask[action] == 0
        
        # Get the name of the action the agent *intended* to take
        intended_action_name = self.env.indices_to_actions.get(action, 'INVALID')
        
        # If the action was illegal, we log it as HOLD, as per the new rule
        final_action_name = 'HOLD' if was_illegal_action else intended_action_name

        if self.env.current_step == 0:
            self._log_step(self.env._get_observation(), is_initial_state=True)

        self._log_step(self.env._get_observation(), action=action, info_override={'action_name': final_action_name})

        timestep = self.env.step(action)
        
        self._log_step(timestep.obs, reward=timestep.reward, done=timestep.done, info=timestep.info)
        
        if timestep.done:
            self.save_log()
            
        return timestep

    def _log_step(self, obs, action=None, reward=None, done=False, info=None, is_initial_state=False, info_override=None):
        serializable_portfolio = []
        
        for index, pos in self.env.portfolio.iterrows():
            mid_price, _, _ = self.env._get_option_details(self.env.current_price, pos['strike_price'], pos['days_to_expiry'], pos['type'])
            current_premium = self.env._get_option_price(mid_price, is_buy=(pos['direction'] == 'short'))
            if pos['direction'] == 'long': pnl = (current_premium - pos['entry_premium']) * self.env.lot_size
            else: pnl = (pos['entry_premium'] - current_premium) * self.env.lot_size
            serializable_portfolio.append({
                'type': pos['type'], 'direction': pos['direction'],
                'strike_price': round(pos['strike_price'], 2),
                'entry_premium': round(pos['entry_premium'], 2),
                'days_to_expiry': round(pos['days_to_expiry'], 2),
                'current_premium': round(current_premium, 2),
                'live_pnl': round(pnl, 2),
            })

        log_info = info if info is not None else {}
        if info_override:
            log_info.update(info_override)
            
        # <<< NEW: Calculate and add the last price change percentage
        current_price = self.env.current_price
        last_price_change_pct = ((current_price / self._last_price) - 1) * 100 if self._last_price else 0.0
        self._last_price = current_price # Update for the next step
        log_info['last_price_change_pct'] = last_price_change_pct

        log_info['price'] = float(current_price)
        log_info['eval_episode_return'] = float(self._get_safe_pnl())
        log_info['start_price'] = float(self.env.start_price)
        log_info['volatility'] = float(getattr(self.env, 'garch_implied_vol', 0.0))
        log_info['risk_free_rate'] = float(self.env.risk_free_rate)
        log_info['market_regime'] = str(getattr(self.env, 'current_regime_name', 'N/A'))
        log_info['illegal_actions_in_episode'] = int(self.env.illegal_action_count)

        log_entry = {
            'step': int(self.env.current_step),
            'day': int(self.env.current_step // self.env.steps_per_day + 1),
            'portfolio': serializable_portfolio,
            'action': int(action) if action is not None else None,
            'reward': float(reward) if reward is not None else None,
            'done': bool(done),
            'info': log_info,
        }
        
        self._episode_history.append(log_entry)

    def _get_safe_pnl(self):
        try:
            return self.env._get_total_pnl()
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

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> list:
        return OptionsZeroGameEnv.create_collector_env_cfg(cfg)

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> list:
        return OptionsZeroGameEnv.create_evaluator_env_cfg(cfg)
