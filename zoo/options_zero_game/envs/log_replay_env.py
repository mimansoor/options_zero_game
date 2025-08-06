# zoo/options_zero_game/envs/log_replay_env.py
# <<< DEFINITIVE VERSION with "Settlement Day" Logging >>>

import json
import gymnasium as gym
import pandas as pd
import copy
from ding.utils import ENV_REGISTRY
from .options_zero_game_env import OptionsZeroGameEnv
from ding.envs.env.base_env import BaseEnvTimestep
from ..entry.bias_meter import BiasMeter

@ENV_REGISTRY.register('log_replay')
class LogReplayEnv(gym.Wrapper):
    def __init__(self, cfg: dict):
        base_env = OptionsZeroGameEnv(cfg)
        super().__init__(base_env)
        self.log_file_path = cfg.get('log_file_path', 'replay_log.json')
        self._episode_history = []
        print(f"LogReplayEnv initialized. Replay will be saved to: {self.log_file_path}")

    def reset(self, **kwargs):
        self._episode_history = []
        obs = self.env.reset(**kwargs)
        self._log_current_state()
        return obs

    def step(self, action):
        timestep = self.env.step(action)
        self._update_previous_log_with_outcome(timestep, action)
        
        if not timestep.done:
            self._log_current_state()
        else:
            # --- THE FIX: Episode is finished, log the special settlement state ---
            self._log_settlement_state(timestep)
            self.save_log()
            
        return timestep

    def _serialize_portfolio(self, portfolio_df: pd.DataFrame, current_price: float) -> list:
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
            current_premium = self.env.bs_manager.get_price_with_spread(mid_price, is_buy=(pos['direction'] == 'short'), bid_ask_spread_pct=self.env._cfg.bid_ask_spread_pct)
            
            pnl_multiplier = 1 if pos['direction'] == 'long' else -1
            pnl = (current_premium - pos['entry_premium']) * self.env.portfolio_manager.lot_size * pnl_multiplier
                
            serializable_portfolio.append({
                'type': pos['type'], 'direction': pos['direction'], 'strike_price': round(pos['strike_price'], 2),
                'entry_premium': round(pos['entry_premium'], 2), 'current_premium': round(current_premium, 2),
                'live_pnl': round(pnl, 2), 'days_to_expiry': float(pos['days_to_expiry']),
                'hedge_status': "Hedged" if pos['is_hedged'] else "Naked"
            })
        return serializable_portfolio

    def _log_current_state(self):
        current_price = self.env.price_manager.current_price
        portfolio_to_log = self.env.portfolio_manager.portfolio
        
        serialized_portfolio = self._serialize_portfolio(portfolio_to_log, current_price)
        pnl_to_log = self._get_safe_pnl(serialized_portfolio)
        
        obs_vector = self.env._get_observation()
        meter = BiasMeter(obs_vector[:self.env.market_and_portfolio_state_size], self.env.OBS_IDX)
        
        last_price = self._episode_history[-1]['info']['price'] if self._episode_history else self.env.price_manager.start_price
        last_price_change_pct = ((current_price / last_price) - 1) * 100 if last_price > 0 else 0.0

        info = {
            'price': float(current_price), 'eval_episode_return': pnl_to_log,
            'start_price': float(self.env.price_manager.start_price),
            'market_regime': str(self.env.price_manager.current_regime_name),
            'last_price_change_pct': last_price_change_pct,
            'directional_bias': meter.directional_bias, 'volatility_bias': meter.volatility_bias,
            'action_name': 'N/A', 'illegal_actions_in_episode': self.env.illegal_action_count
        }

        log_entry = {'step': int(self.env.current_step), 'day': self.env.current_day, 'portfolio': serialized_portfolio, 'info': info, 'action': None, 'reward': None, 'done': False}
        self._episode_history.append(log_entry)

    def _update_previous_log_with_outcome(self, timestep, action_taken):
        if not self._episode_history: return
        log_entry_to_update = self._episode_history[-1]
        info = timestep.info
        
        log_entry_to_update.update({'action': int(action_taken), 'reward': timestep.reward, 'done': timestep.done})
        log_entry_to_update['info']['action_name'] = info.get('executed_action_name', 'N/A')
        log_entry_to_update['info']['illegal_actions_in_episode'] = info.get('illegal_actions_in_episode', 0)
        
        post_action_portfolio = self.env.portfolio_manager.get_pre_step_portfolio()
        price_at_action_time = log_entry_to_update['info']['price']
        
        serialized_post_action_portfolio = self._serialize_portfolio(post_action_portfolio, price_at_action_time)
        log_entry_to_update['portfolio'] = serialized_post_action_portfolio
        log_entry_to_update['info']['eval_episode_return'] = self._get_safe_pnl(serialized_post_action_portfolio)

    def _get_safe_pnl(self, serialized_portfolio: list) -> float:
        """
        Calculates the total PnL by summing the 'live_pnl' from a pre-serialized portfolio list.
        """
        if not serialized_portfolio:
            return self.env.portfolio_manager.realized_pnl
            
        total_live_pnl = sum(p['live_pnl'] for p in serialized_portfolio)
        return self.env.portfolio_manager.realized_pnl + total_live_pnl

    def _log_settlement_state(self, final_timestep):
        """
        Creates and appends a final, special log entry for the end of the episode
        to show the fully realized PnL and an empty portfolio.
        """
        if not self._episode_history: return

        # Use the last known state as a template
        last_log_entry = copy.deepcopy(self._episode_history[-1])
        
        # Create the settlement entry
        settlement_entry = last_log_entry
        
        # Update to reflect the final settlement
        settlement_entry['step'] = last_log_entry['step'] + 1
        settlement_entry['day'] = last_log_entry['day'] + 1
        settlement_entry['portfolio'] = [] # The portfolio is now empty
        settlement_entry['done'] = True
        settlement_entry['action'] = None
        settlement_entry['reward'] = None
        
        # The EOD PnL is the final, realized return from the episode
        settlement_entry['info']['eval_episode_return'] = final_timestep.info['eval_episode_return']
        settlement_entry['info']['action_name'] = "AUTO-SETTLE"
        
        # Carry over the last known biases for context
        settlement_entry['info']['directional_bias'] = last_log_entry['info']['directional_bias']
        settlement_entry['info']['volatility_bias'] = last_log_entry['info']['volatility_bias']
        
        self._episode_history.append(settlement_entry)

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
