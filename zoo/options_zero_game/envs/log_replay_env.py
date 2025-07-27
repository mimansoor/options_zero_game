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
    This version is updated to work with the refactored OptionsZeroGameEnv.
    """
    
    def __init__(self, cfg: dict):
        base_env = OptionsZeroGameEnv(cfg)
        super().__init__(base_env)
        self.log_file_path = cfg.log_file_path
        self._episode_history = []
        self._last_day_price = None
        print(f"LogReplayEnv initialized. Replay will be saved to: {self.log_file_path}")

    def seed(self, seed: int, dynamic_seed: int = None):
        return self.env.seed(seed, dynamic_seed)

    def reset(self, **kwargs):
        self._episode_history = []
        obs = self.env.reset(**kwargs)
        self._last_day_price = self.env.start_price
        self._log_step(obs, is_initial_state=True)
        return obs

    def step(self, action):
        # Note: The main env now handles illegal actions, so we don't need _enforce_legal_action here.
        action_name = self.env.indices_to_actions.get(action, 'INVALID')
        
        # Log the state *before* the action is taken
        self._log_step(self.env._get_observation(), action=action, info_override={'action_name': action_name})

        # Execute the step in the real environment
        timestep = self.env.step(action)
        
        # Calculate daily change for logging
        current_price = timestep.info['price']
        daily_change_pct = ((current_price / self._last_day_price) - 1) * 100 if self._last_day_price else 0.0
        self._last_day_price = current_price
        
        info_with_daily_change = timestep.info.copy()
        info_with_daily_change['daily_change_pct'] = daily_change_pct
        
        # Log the state *after* the action and market move
        self._log_step(timestep.obs, action=None, reward=timestep.reward, done=timestep.done, info=info_with_daily_change)
        
        if timestep.done:
            self.save_log()
            
        return timestep

    def _log_step(self, obs, action=None, reward=None, done=False, info=None, is_initial_state=False, info_override=None):
        serializable_portfolio = []
        
        # <<< THE FIX: Use .iterrows() to correctly iterate over the DataFrame rows
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
            
        # Ensure all necessary info fields are present
        log_info.setdefault('price', self.env.current_price)
        log_info.setdefault('eval_episode_return', self._get_total_pnl())
        log_info.setdefault('start_price', self.env.start_price)
        log_info.setdefault('volatility', self.env.garch_implied_vol)
        log_info.setdefault('risk_free_rate', self.env.risk_free_rate)
        log_info.setdefault('daily_change_pct', 0.0)
        # <<< NEW: Add the market regime name to the log!
        log_info.setdefault('market_regime', self.env.current_regime_name)

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

    # Helper to access the correct PnL method, as the wrapper hides the base env
    def _get_total_pnl(self):
        if hasattr(self.env, '_get_total_pnl'):
            return self.env._get_total_pnl()
        return 0.0

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> list:
        return OptionsZeroGameEnv.create_collector_env_cfg(cfg)

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> list:
        return OptionsZeroGameEnv.create_evaluator_env_cfg(cfg)
