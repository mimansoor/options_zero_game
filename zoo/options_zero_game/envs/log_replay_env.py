import json
import gym
from easydict import EasyDict
import copy

from ding.utils import ENV_REGISTRY
from .options_zero_game_env import OptionsZeroGameEnv
from ding.envs.env.base_env import BaseEnvTimestep

@ENV_REGISTRY.register('log_replay')
class LogReplayEnv(gym.Wrapper):
    """
    A gym.Wrapper that logs all interactions and saves them to a JSON file.
    This version implements a two-part log for action and market-close states.
    """
    
    def __init__(self, cfg: dict):
        base_env = OptionsZeroGameEnv(cfg)
        super().__init__(base_env)
        self.log_file_path = cfg.log_file_path
        self._episode_history = []
        print(f"LogReplayEnv initialized. Replay will be saved to: {self.log_file_path}")

    def seed(self, seed: int, dynamic_seed: int = None):
        return self.env.seed(seed, dynamic_seed)

    def reset(self, **kwargs):
        self._episode_history = []
        obs = self.env.reset(**kwargs)
        # Log the initial state (e.g., Day 0 EOD)
        self._log_step(obs, is_initial_state=True)
        return obs

    # <<< MODIFIED: The new, two-part logging step method
    def step(self, action):
        # The agent has made a decision based on the state at the start of the day.
        
        # --- Phase 1: Log the Action ---
        # Get the human-readable action name
        action_name = self.env.indices_to_actions.get(action, 'INVALID')
        
        # Log the state at the moment of the trade, BEFORE the market moves.
        # We pass the action_name to be included in this specific log entry.
        self._log_step(self.env._get_observation(), action=action, info_override={'action_name': action_name})

        # --- Phase 2: Execute the step and Log the Market Close ---
        # Now, call the underlying environment's step function.
        # This will execute the trade, move the market, and calculate the final PnL.
        timestep = self.env.step(action)
        
        # Log the state at the END of the day, after the market has moved.
        self._log_step(timestep.obs, action=None, reward=timestep.reward, done=timestep.done, info=timestep.info)
        
        if timestep.done:
            self.save_log()
            
        return timestep

    def _log_step(self, obs, action=None, reward=None, done=False, info=None, is_initial_state=False, info_override=None):
        """Logs a single timestep of data."""
        
        serializable_portfolio = []
        for pos in self.env.portfolio:
            mid_price, _, _ = self.env._get_option_details(self.env.current_price, pos['strike_price'], pos['days_to_expiry'], pos['type'])
            current_premium = self.env._get_option_price(mid_price, is_buy=(pos['direction'] == 'short'))
            if pos['direction'] == 'long': pnl = (current_premium - pos['entry_premium']) * self.env.lot_size
            else: pnl = (pos['entry_premium'] - current_premium) * self.env.lot_size
            serializable_portfolio.append({
                'type': pos['type'], 'direction': pos['direction'],
                'strike_price': round(pos['strike_price'], 2),
                'entry_premium': round(pos['entry_premium'], 2),
                'days_to_expiry': pos['days_to_expiry'],
                'current_premium': round(current_premium, 2),
                'live_pnl': round(pnl, 2),
            })

        # Use the provided info dict, or create a new one
        log_info = info if info is not None else {}
        # If an override is provided (for the "Action" log), update the info
        if info_override:
            log_info.update(info_override)
            
        # Ensure base keys exist
        log_info.setdefault('price', self.env.current_price)
        log_info.setdefault('eval_episode_return', self.env._get_portfolio_value())
        log_info.setdefault('start_price', self.env.start_price)
        log_info.setdefault('volatility', self.env.volatility)
        log_info.setdefault('risk_free_rate', self.env.risk_free_rate)

        log_entry = {
            'portfolio': serializable_portfolio,
            'action': int(action) if action is not None else None,
            'reward': float(reward) if reward is not None else None,
            'done': done,
            'info': log_info,
        }
        
        self._episode_history.append(log_entry)

    def save_log(self):
        print(f"Episode finished. Saving replay log with {len(self._episode_history)} steps...")
        try:
            with open(self.log_file_path, 'w') as f:
                json.dump(self._episode_history, f, indent=4)
            print(f"Successfully saved replay log to {self.log_file_path}")
        except Exception as e:
            print(f"Error saving replay log: {e}")

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> list:
        return OptionsZeroGameEnv.create_collector_env_cfg(cfg)

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> list:
        return OptionsZeroGameEnv.create_evaluator_env_cfg(cfg)
