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
        step_at_action = self.env.current_step
        day_at_action = self.env.current_day
        pre_action_price = self.env.price_manager.current_price
        equity_before = self.env.portfolio_manager.get_current_equity(pre_action_price, self.env.iv_bin_index)
        
        self.env._take_action_on_state(action)

        # 3. Check for any closed trade receipts generated during this step.
        receipts = self.env.portfolio_manager.receipts_for_current_step
        if receipts:
            # The exit_day is the same for all trades closed in one step.
            for r in receipts:
                r['exit_day'] = day_at_action
            # Extend the log with all new receipts.
            self._closed_trades_log.extend(receipts)
            
        self._log_state_snapshot(
            step_num=step_at_action, day_num=day_at_action, is_post_action=True,
            price_for_log=pre_action_price, action_info=self.env.last_action_info
        )

        timestep = self.env._advance_market_and_get_outcome(equity_before)
        
        self._log_state_snapshot(
            step_num=step_at_action, day_num=day_at_action, is_post_action=False,
            price_for_log=self.env.price_manager.current_price, 
            action_info=self.env.last_action_info, final_timestep=timestep
        )

        if timestep.done:
            # Add the special final entry to the log.
            self._log_settlement_state(timestep)
            # Then save the completed log.
            self.save_log()
            
        return timestep

    # <<< ADD THIS NEW, CORRECTED METHOD >>>
    def _get_live_pnl(self, portfolio_df: pd.DataFrame, current_price: float, iv_bin_index: int) -> list:
        """
        Calculates the live, un-realized P&L for each individual leg in the
        portfolio for logging purposes. This is a corrected, robust version.
        """
        if portfolio_df.empty:
            return []

        live_pnl_data = []
        atm_price = self.env.market_rules_manager.get_atm_price(current_price)
        lot_size = self.env.portfolio_manager.lot_size

        for _, leg in portfolio_df.iterrows():
            is_call = leg['type'] == 'call'
            
            # 1. Get the current, live market premium for the leg
            offset = round((leg['strike_price'] - atm_price) / self.env.strike_distance)
            vol = self.env.iv_calculator(offset, leg['type'])
            greeks = self.env.bs_manager.get_all_greeks_and_price(
                current_price, leg['strike_price'], leg['days_to_expiry'], vol, is_call
            )
            
            # Determine the exit price, including the bid-ask spread
            current_premium = self.env.bs_manager.get_price_with_spread(
                greeks['price'], is_buy=(leg['direction'] == 'short'), bid_ask_spread_pct=self.env.bid_ask_spread_pct
            )

            # 2. Correctly calculate the P&L based on direction
            direction_multiplier = 1 if leg['direction'] == 'long' else -1
            pnl_per_share = current_premium - leg['entry_premium']
            live_pnl = pnl_per_share * direction_multiplier * lot_size

            # 3. Create a dictionary with all necessary data for the UI
            leg_data = leg.to_dict()
            leg_data['current_premium'] = current_premium
            leg_data['live_pnl'] = live_pnl
            live_pnl_data.append(leg_data)
            
        return live_pnl_data

    def _log_state_snapshot(self, step_num, day_num, is_post_action, price_for_log, action_info, final_timestep=None):
        info = copy.deepcopy(final_timestep.info) if final_timestep else {}
        
        if is_post_action:
            log_step, log_day = step_num, day_num
            portfolio_to_log = self.env.portfolio_manager.get_portfolio()
            info['executed_action_name'] = action_info['final_action_name']
            reward, done = None, False
            obs_for_bias = self.env._get_observation()
        else: # End-of-Day
            log_step, log_day = step_num, day_num
            portfolio_to_log = self.env.portfolio_manager.get_portfolio()
            info['executed_action_name'] = "MARKET MOVE / EOD"
            reward, done = final_timestep.reward, final_timestep.done
            obs_for_bias = final_timestep.obs['observation']

        serialized_portfolio = self._serialize_portfolio(portfolio_to_log, price_for_log)
        pnl_verification = self.env.portfolio_manager.get_pnl_verification(price_for_log, self.env.iv_bin_index)
        payoff_data = self.env.portfolio_manager.get_payoff_data(price_for_log, self.env.iv_bin_index)
        
        meter = BiasMeter(obs_for_bias[:self.env.market_and_portfolio_state_size], self.env.OBS_IDX)
        
        last_price = self._episode_history[-1]['info']['price'] if self._episode_history else self.env.price_manager.start_price
        last_price_change_pct = ((price_for_log / last_price) - 1) * 100 if last_price > 0 else 0.0
        
        info.update({
            'price': float(price_for_log),
            'eval_episode_return': pnl_verification['verified_total_pnl'],
            'last_price_change_pct': last_price_change_pct,
            'directional_bias': meter.directional_bias,
            'volatility_bias': meter.volatility_bias,
            'portfolio_stats': self.env.portfolio_manager.get_raw_portfolio_stats(price_for_log, self.env.iv_bin_index),
            'pnl_verification': pnl_verification,
            'payoff_data': payoff_data,
            'closed_trades_log': copy.deepcopy(self._closed_trades_log)
        })

        log_entry = {
            'step': log_step, 'day': log_day,
            'sub_step_name': "Post-Action" if is_post_action else "End-of-Day",
            'portfolio': serialized_portfolio, 'info': info,
            'reward': float(reward) if reward is not None else None,
            'done': bool(done),
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
        
        final_info = copy.deepcopy(info)
        final_info.update({
            'directional_bias': "Neutral", 'volatility_bias': "Neutral / Low Volatility Expected",
            'executed_action_name': final_action_name, 'eval_episode_return': final_pnl,
            'portfolio_stats': {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'max_profit': 0.0, 'max_loss': 0.0, 'rr_ratio': 0.0, 'prob_profit': 1.0 if final_pnl > 0 else 0.0, 'profit_factor': 0.0},
            'pnl_verification': {'realized_pnl': final_pnl, 'unrealized_pnl': 0.0, 'verified_total_pnl': final_pnl},
            'payoff_data': {'expiry_pnl': [], 'current_pnl': [], 'spot_price': info['price'], 'sigma_levels': {}}
        })

        settlement_entry = {
            'step': last_log_entry['step'], 'day': last_log_entry['day'],
            'sub_step_name': "Settlement",
            'portfolio': [],
            'done': True,
            'action': None,
            'reward': final_timestep.reward,
            'info': final_info
        }
        self._episode_history.append(settlement_entry)

    def _serialize_portfolio(self, portfolio_df: pd.DataFrame, current_price: float) -> list:
        """
        Calculates the live, un-realized P&L for each individual leg in the
        portfolio and returns a list of dictionaries ready for JSON logging.
        This is the corrected, robust version.
        """
        if portfolio_df.empty:
            return []

        serialized_data = []
        atm_price = self.env.market_rules_manager.get_atm_price(current_price)
        lot_size = self.env.portfolio_manager.lot_size

        for _, leg in portfolio_df.iterrows():
            is_call = leg['type'] == 'call'
            
            # --- 1. Get the current, live market premium for the leg ---
            offset = round((leg['strike_price'] - atm_price) / self.env.strike_distance)
            
            # Use the environment's official IV calculator for consistency
            vol = self.env._get_dynamic_iv(offset, leg['type'])
            
            greeks = self.env.bs_manager.get_all_greeks_and_price(
                current_price, leg['strike_price'], leg['days_to_expiry'], vol, is_call
            )
            
            # Determine the current market value (exit price), including the bid-ask spread
            current_premium = self.env.bs_manager.get_price_with_spread(
                greeks['price'], is_buy=(leg['direction'] == 'short'), bid_ask_spread_pct=self.env.bid_ask_spread_pct
            )

            # --- 2. Correctly calculate the Live P&L based on direction ---
            direction_multiplier = 1 if leg['direction'] == 'long' else -1
            pnl_per_share = current_premium - leg['entry_premium']
            live_pnl = pnl_per_share * direction_multiplier * lot_size

            # --- 3. Create a dictionary with all necessary data for the UI ---
            leg_data = leg.to_dict()
            leg_data['current_premium'] = current_premium
            leg_data['live_pnl'] = live_pnl
            serialized_data.append(leg_data)
            
        return serialized_data

    def save_log(self):
        """Saves the complete episode history to a JSON file."""
        print(f"Episode finished. Saving replay log with {len(self._episode_history)} steps...")

        # Create a final log object that contains the episode steps AND the historical data.
        final_log_object = {
            'historical_context': self.env.price_manager.historical_context_path.tolist(),
            'episode_data': self._episode_history
        }

        try:
            with open(self.log_file_path, 'w') as f:
                json.dump(final_log_object, f, indent=2, cls=NumpyEncoder)
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
