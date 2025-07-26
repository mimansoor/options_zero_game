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

    # <<< MODIFIED: The corrected, two-part logging step method
    def step(self, action):
        # The agent has made a decision based on the state at the start of the day.
        
        # --- Phase 1: Enforce Rules and Log the ACTUAL Action ---
        
        # <<< THE FIX: Ask the underlying environment to validate the action first.
        # This ensures we log the action that will actually be executed.
        corrected_action = self.env._enforce_legal_action(action)
        
        # Get the human-readable name of the corrected action
        action_name = self.env.indices_to_actions.get(corrected_action, 'INVALID')
        
        # Log the state at the moment of the trade, BEFORE the market moves.
        # We pass the corrected action and its name to be included in this log entry.
        self._log_step(self.env._get_observation(), action=corrected_action, info_override={'action_name': action_name})

        # --- Phase 2: Execute the step and Log the Market Close ---
        # Now, call the underlying environment's step function with the GUARANTEED-LEGAL action.
        timestep = self.env.step(corrected_action)
        
        # Calculate daily change before logging the market close
        current_price = timestep.info['price']
        daily_change_pct = ((current_price / self._last_day_price) - 1) * 100 if self._last_day_price else 0.0
        self._last_day_price = current_price
        
        info_with_daily_change = timestep.info.copy()
        info_with_daily_change['daily_change_pct'] = daily_change_pct
        
        # Log the state at the END of the day, after the market has moved.
        self._log_step(timestep.obs, action=None, reward=timestep.reward, done=timestep.done, info=info_with_daily_change)
        
        if timestep.done:
            self.save_log()
            
        return timestep

    def _log_step(self, obs, action=None, reward=None, done=False, info=None, is_initial_state=False, info_override=None):
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

        log_info = info if info is not None else {}
        if info_override:
            log_info.update(info_override)
            
        log_info.setdefault('price', self.env.current_price)
        log_info.setdefault('eval_episode_return', self.env._get_portfolio_value())
        log_info.setdefault('start_price', self.env.start_price)
        log_info.setdefault('volatility', self.env.volatility)
        log_info.setdefault('risk_free_rate', self.env.risk_free_rate)
        log_info.setdefault('daily_change_pct', 0.0)

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
