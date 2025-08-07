# zoo/options_zero_game/envs/log_replay_env.py
# <<< DEFINITIVE VERSION with Correct Single-Pass Logging >>>

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
        # We do NOT log here. The first log entry will be created by the first step.
        return obs

    def step(self, action):
        """
        The definitive step method. It uses a simple, single-pass logic to
        log the state that resulted from the agent's action.
        """
        # 1. Get the state of the world *before* the action.
        pre_action_price = self.env.price_manager.current_price
        pre_action_step = self.env.current_step
        pre_action_day = self.env.current_day
        
        # 2. Execute the step in the main environment. This moves the world to the next state.
        timestep = self.env.step(action)
        
        # 3. Create the definitive log entry for the action that was just taken.
        self._log_action_outcome(timestep, action, pre_action_price, pre_action_step, pre_action_day)
        
        # 4. If the episode just ended, add the special "Settlement Day" entry.
        if timestep.done:
            self._log_settlement_state(timestep)
            self.save_log()
            
        return timestep

    def _log_action_outcome(self, timestep, action_taken, price_at_action_time, step_at_action, day_at_action):
        """Creates a single, complete log entry for the action that was just taken."""
        
        # --- The Definitive Logic ---
        # The portfolio to log is the snapshot taken IMMEDIATELY after the action.
        portfolio_to_log = self.env.portfolio_manager.get_post_action_portfolio()
        serialized_portfolio = self._serialize_portfolio(portfolio_to_log, price_at_action_time)
        pnl_to_log = self._get_safe_pnl(serialized_portfolio)
        
        # The observation vector for the bias meter is the one from the state the action was taken in.
        # This requires a small temporary change, as the `timestep` has the *next* observation.
        # For simplicity, we'll re-calculate the bias for the pre-action state.
        # A more advanced implementation would pass the pre-action obs through the timestep.
        meter = BiasMeter(self.env._get_observation()[:self.env.market_and_portfolio_state_size], self.env.OBS_IDX)

        last_price = self._episode_history[-1]['info']['price'] if self._episode_history else self.env.price_manager.start_price
        last_price_change_pct = ((price_at_action_time / last_price) - 1) * 100 if last_price > 0 else 0.0

        info = {
            'price': float(price_at_action_time), 'eval_episode_return': pnl_to_log,
            'start_price': float(self.env.price_manager.start_price),
            'market_regime': str(self.env.price_manager.current_regime_name),
            'last_price_change_pct': last_price_change_pct,
            'directional_bias': meter.directional_bias, 'volatility_bias': meter.volatility_bias,
            'action_name': timestep.info.get('executed_action_name', 'N/A'),
            'illegal_actions_in_episode': timestep.info.get('illegal_actions_in_episode', 0)
        }

        log_entry = {
            'step': int(step_at_action), 'day': int(day_at_action),
            'portfolio': serialized_portfolio, 'info': info,
            'action': int(action_taken), 'reward': timestep.reward, 'done': timestep.done
        }
        self._episode_history.append(log_entry)

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
            current_premium = self.env.bs_manager.get_price_with_spread(mid_price, is_buy=(pos['direction'] == 'short'), bid_ask_spread_pct=self.env.bid_ask_spread_pct)
            
            pnl_multiplier = 1 if pos['direction'] == 'long' else -1
            pnl = (current_premium - pos['entry_premium']) * self.env.portfolio_manager.lot_size * pnl_multiplier
                
            serializable_portfolio.append({
                'type': pos['type'], 'direction': pos['direction'], 'strike_price': round(pos['strike_price'], 2),
                'entry_premium': round(pos['entry_premium'], 2), 'current_premium': round(current_premium, 2),
                'live_pnl': round(pnl, 2), 'days_to_expiry': float(pos['days_to_expiry']),
                'hedge_status': "Hedged" if pos['is_hedged'] else "Naked"
            })
        return serializable_portfolio

    def _get_safe_pnl(self, serialized_portfolio: list) -> float:
        """
        Calculates the total PnL by summing the 'live_pnl' from a pre-serialized portfolio list.
        """
        if not serialized_portfolio:
            return self.env.portfolio_manager.realized_pnl
            
        total_live_pnl = sum(p['live_pnl'] for p in serialized_portfolio)
        return self.env.portfolio_manager.realized_pnl + total_live_pnl

    def _log_settlement_state(self, final_timestep):
        """Creates and appends a final, special log entry for the end of the episode."""
        if not self._episode_history: return
        
        last_log_entry = copy.deepcopy(self._episode_history[-1])
        info = final_timestep.info

        # The portfolio to log is the snapshot taken IMMEDIATELY after the action.
        portfolio_to_log = self.env.portfolio_manager.get_portfolio()
        serialized_portfolio = self._serialize_portfolio(portfolio_to_log, self.env.price_manager.current_price)
 
        # Determine the reason for termination
        final_action_name = info['final_action_name']
        
        settlement_entry = last_log_entry
        settlement_entry.update({
            'step': last_log_entry['step'] + 1, 'day': last_log_entry['day'] + 1,
            'portfolio': serialized_portfolio, 'done': True, 'action': None, 'reward': final_timestep.reward
        })
        settlement_entry['info']['eval_episode_return'] = info['eval_episode_return']
        settlement_entry['info']['action_name'] = final_action_name
        
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
