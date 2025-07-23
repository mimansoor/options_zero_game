import json
import gym
from easydict import EasyDict
import copy

from ding.utils import ENV_REGISTRY
from .options_zero_game_env import OptionsZeroGameEnv
# <<< THE FIX (Part 1): We need the BaseEnvTimestep to return it
from ding.envs.env.base_env import BaseEnvTimestep

@ENV_REGISTRY.register('log_replay')
class LogReplayEnv(gym.Wrapper):
    """
    A gym.Wrapper that logs all interactions and saves them to a JSON file.
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
        # The DI-engine env reset returns a single obs dictionary
        obs = self.env.reset(**kwargs)
        self._log_step(obs, is_initial_state=True)
        return obs

    def step(self, action):
        # The DI-engine env step returns a BaseEnvTimestep object
        timestep = self.env.step(action)
        
        # Log the data from the timestep object
        self._log_step(timestep.obs, action, timestep.reward, timestep.done, timestep.info)
        
        if timestep.done:
            self.save_log()
            
        # <<< THE FIX (Part 2): Return the original, unmodified timestep object
        return timestep

    def _log_step(self, obs, action=None, reward=None, done=False, info=None, is_initial_state=False):
        serializable_obs = {
            'observation': obs['observation'].tolist(),
            'action_mask': obs['action_mask'].tolist(),
            'to_play': obs['to_play'].tolist(),
        }
        log_entry = {
            'obs': serializable_obs,
            'action': int(action) if action is not None else None,
            'reward': float(reward) if reward is not None else None,
            'done': done,
            'info': info if info is not None else {},
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
