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
    This wrapper is now a fully registered DI-engine environment.
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
        self._log_step(obs, is_initial_state=True)
        return obs

    def step(self, action):
        timestep = self.env.step(action)
        self._log_step(timestep.obs, action, timestep.reward, timestep.done, timestep.info)
        if timestep.done:
            self.save_log()
        return timestep

    def _log_step(self, obs, action=None, reward=None, done=False, info=None, is_initial_state=False):
        serializable_obs = {
            'observation': obs['observation'].tolist(),
            'action_mask': obs['action_mask'].tolist(),
            'to_play': obs['to_play'].tolist(),
        }

        # <<< THE FIX: Calculate and log the ground truth from the Python environment
        serializable_portfolio = []
        lot_size = self.env.lot_size
        for pos in self.env.portfolio:
            # Get the live, mark-to-liquidation price from the environment
            mid_price, _, _ = self.env._get_option_details(self.env.current_price, pos['strike_price'], pos['days_to_expiry'], pos['type'])
            current_premium = self.env._get_option_price(mid_price, is_buy=(pos['direction'] == 'short'))

            # Calculate the live PnL for this leg
            if pos['direction'] == 'long':
                pnl = (current_premium - pos['entry_premium']) * lot_size
            else:
                pnl = (pos['entry_premium'] - current_premium) * lot_size

            serializable_portfolio.append({
                'type': pos['type'],
                'direction': pos['direction'],
                'strike_price': round(pos['strike_price'], 2),
                'entry_premium': round(pos['entry_premium'], 2),
                'days_to_expiry': pos['days_to_expiry'],
                # <<< NEW: Add the ground truth data to the log
                'current_premium': round(current_premium, 2),
                'live_pnl': round(pnl, 2),
            })

        if info is None: info = {}
        info['volatility'] = self.env.volatility
        info['risk_free_rate'] = self.env.risk_free_rate
        info['start_price'] = self.env.start_price

        log_entry = {
            'obs': serializable_obs,
            'portfolio': serializable_portfolio,
            'action': int(action) if action is not None else None,
            'reward': float(reward) if reward is not None else None,
            'done': done,
            'info': info,
        }

        if info and 'eval_episode_return' in info:
            log_entry['info']['eval_episode_return'] = float(info['eval_episode_return'])

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
