# zoo/options_zero_game/envs/options_zero_game_env.py
# <<< FINAL VERSION, incorporating all fixes >>>

import copy
import math
import random
from typing import Tuple, Dict, Any, List
import time

import gymnasium as gym
import numpy as np
from easydict import EasyDict
from gymnasium import spaces
from gymnasium.utils import seeding

from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY

# --- Import the new, specialized manager classes ---
from .black_scholes_manager import BlackScholesManager
from .price_action_manager import PriceActionManager
from .portfolio_manager import PortfolioManager
from .market_rules_manager import MarketRulesManager
from ..entry.bias_meter import BiasMeter

@ENV_REGISTRY.register('options_zero_game')
class OptionsZeroGameEnv(gym.Env):
    VERSION = "2.2-Final"
    metadata = {'render.modes': ['human']}
    
    # --- THIS IS THE FIX ---
    # A complete default configuration that allows the environment to be
    # instantiated on its own without errors.
    config = dict(
        # Price Action Manager Config
        price_source='mixed',
        forced_historical_symbol=None,
        historical_data_path='zoo/options_zero_game/data/market_data_cache',
        market_regimes = [
            {'name': 'Stable_LowVol', 'mu': 0.00005, 'omega': 0.000005, 'alpha': 0.09, 'beta': 0.90, 'overnight_vol_multiplier': 1.5},
            {'name': 'Crisis_HighVol', 'mu': -0.0005, 'omega': 0.0001, 'alpha': 0.15, 'beta': 0.82, 'overnight_vol_multiplier': 2.2},
        ],
        
        # Time and Episode Config
        time_to_expiry_days=40,
        min_time_to_expiry_days=5,
        forced_episode_length=0,
        steps_per_day=1,
        trading_day_in_mins=375,
        
        # Observation Feature Config
        rolling_vol_window=5,
        momentum_window_steps=20,
        
        # Options and Portfolio Config
        start_price=20000.0,
        initial_cash=500000.0,
        lot_size=75,
        max_positions=4,
        strike_distance=50.0,
        max_strike_offset=20,
        bid_ask_spread_pct=0.002,
        brokerage_per_leg=20.0,
        
        # Black-Scholes Manager Config
        risk_free_rate=0.10,

        iv_skew_table={
            'call': {
                '-20':(18.0,20.5),'-19':(17.8,20.3),'-18':(17.6,20.1),'-17':(17.4,19.9),'-16':(17.2,19.7),
                '-15':(17.0,19.5),'-14':(16.8,19.3),'-13':(16.6,19.1),'-12':(16.4,18.9),'-11':(16.2,18.7),
                '-10':(16.0,18.5),'-9':(15.8,18.3),'-8':(15.6,18.1),'-7':(15.4,17.9),'-6':(15.2,17.7),
                '-5':(15.0,17.5),'-4':(14.8,17.3),'-3':(14.6,17.1),'-2':(14.4,16.9),'-1':(14.2,16.7),
                '0': (14.0,16.5),
                '1':(14.2,16.7),'2':(14.4,16.9),'3':(14.6,17.1),'4':(14.8,17.3),'5':(15.0,17.5),
                '6':(15.2,17.7),'7':(15.4,17.9),'8':(15.6,18.1),'9':(15.8,18.3),'10':(16.0,18.5),
                '11':(16.2,18.7),'12':(16.4,18.9),'13':(16.6,19.1),'14':(16.8,19.3),'15':(17.0,19.5),
                '16':(17.2,19.7),'17':(17.4,19.9),'18':(17.6,20.1),'19':(17.8,20.3),'20':(18.0,20.5)
            },
            'put':  {
                '-20':(18.5,21.0),'-19':(18.3,20.8),'-18':(18.1,20.6),'-17':(17.9,20.4),'-16':(17.7,20.2),
                '-15':(17.5,20.0),'-14':(17.3,19.8),'-13':(17.1,19.6),'-12':(16.9,19.4),'-11':(16.7,19.2),
                '-10':(16.5,19.0),'-9':(16.3,18.8),'-8':(16.1,18.6),'-7':(15.9,18.4),'-6':(15.7,18.2),
                '-5':(15.5,18.0),'-4':(15.3,17.8),'-3':(15.1,17.6),'-2':(14.9,17.4),'-1':(14.7,17.2),
                '0': (14.5,17.0),
                '1':(14.7,17.2),'2':(14.9,17.4),'3':(15.1,17.6),'4':(15.3,17.8),'5':(15.5,18.0),
                '6':(15.7,18.2),'7':(15.9,18.4),'8':(16.1,18.6),'9':(16.3,18.8),'10':(16.5,19.0),
                '11':(16.7,19.2),'12':(16.9,19.4),'13':(17.1,19.6),'14':(17.3,19.8),'15':(17.5,20.0),
                '16':(17.7,20.2),'17':(17.9,20.4),'18':(18.1,20.6),'19':(18.3,20.8),'20':(18.5,21.0)
            },
        },

        # Reward and Penalty Config
        pnl_scaling_factor=1000,
        drawdown_penalty_weight=0.1,
        illegal_action_penalty=-1.0,
        
        # Advanced Trading Rules
        profit_target_pct=6.0,

        close_short_leg_on_profit_threshold=2.0,
        jackpot_reward=1.0,

        # --- NEW DYNAMIC TAKE-PROFIT RULES ---
        # For credit strategies, target is a % of the max possible profit (the credit received).
        credit_strategy_take_profit_pct=25.0, # Target 25% of max profit. Set to 0 to disable.
        
        # For debit strategies, target is a multiple of the initial debit paid.
        debit_strategy_take_profit_multiple=2.0, # Target 2x the debit paid (200% return). Set to 0 to disable.

        stop_loss_multiple_of_cost=2.0, # NEW: Added stop loss multiple
        use_stop_loss=True,
        forced_opening_strategy_name=None,
        disable_opening_curriculum=False,
        
        # Agent/Framework Config
        ignore_legal_actions=True,

        # These rules apply ONLY when opening a new LONG single-leg position.
        otm_long_delta_threshold=0.45, # Disallow buying options with delta < 45
        itm_long_delta_threshold=0.55, # Disallow buying options with delta > 55
        itm_short_delta_threshold=0.55, # Disallow shorting options with delta > 55

        strategy_name_to_id={}, # Can be left empty, as it's populated from the main config
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None) -> None:
        self._cfg = self.default_config()
        if cfg is not None: self._cfg.update(cfg)
        
        # The environment now knows its own role.
        self.is_eval_mode = self._cfg.get('is_eval_mode', False)

        self.np_random, _ = seeding.np_random(None)

        self.bs_manager = BlackScholesManager(self._cfg)
        self.price_manager = PriceActionManager(self._cfg, self.np_random)
        self.market_rules_manager = MarketRulesManager(self._cfg)
        
        # The PortfolioManager needs references to the other managers
        self.portfolio_manager = PortfolioManager(self._cfg, self.bs_manager, self.market_rules_manager)
        
        self.actions_to_indices = self._build_action_space()
        self.indices_to_actions = {v: k for k, v in self.actions_to_indices.items()}
        self.action_space_size = len(self.actions_to_indices)

        self.OBS_IDX = {
            'PRICE_NORM': 0, 'TIME_NORM': 1, 'PNL_NORM': 2, 'VOL_MISMATCH_NORM': 3, 'LOG_RETURN': 4,
            'MOMENTUM_NORM': 5, 'PORTFOLIO_DELTA': 6, 'PORTFOLIO_GAMMA': 7, 'PORTFOLIO_THETA': 8, 'PORTFOLIO_VEGA': 9,
            'PORTFOLIO_MAX_PROFIT_NORM': 10, 'PORTFOLIO_MAX_LOSS_NORM': 11, 'PORTFOLIO_RR_RATIO_NORM': 12, 'PORTFOLIO_PROB_PROFIT': 13,
            # --- NEW EXPERT FEATURES ---
            'EXPERT_EMA_RATIO': 14, 'EXPERT_RSI_OVERSOLD': 15, 'EXPERT_RSI_NEUTRAL': 16, 'EXPERT_RSI_OVERBOUGHT': 17, 'EXPERT_VOL_NORM': 18,
            # --- NEW MARKET EXPECTATION FEATURE ---
            'EXPECTED_MOVE_NORM': 19,
        }
        self.POS_IDX = {
            'IS_OCCUPIED': 0, 'TYPE_NORM': 1, 'DIRECTION_NORM': 2, 'STRIKE_DIST_NORM': 3, 'DAYS_HELD_NORM': 4,
            'PROB_OF_PROFIT': 5, 'MAX_PROFIT_NORM': 6, 'MAX_LOSS_NORM': 7, 'DELTA': 8, 'GAMMA': 9, 'THETA': 10, 'VEGA': 11,
            'IS_HEDGED': 12,
        }
        
        self.MARKET_STATE_SIZE = len(self.OBS_IDX)
        self.PORTFOLIO_STATE_SIZE_PER_POS = len(self.POS_IDX)
        self.PORTFOLIO_START_IDX = self.MARKET_STATE_SIZE
        
        self.market_and_portfolio_state_size = self.MARKET_STATE_SIZE + (self._cfg.max_positions * self.PORTFOLIO_STATE_SIZE_PER_POS)
        self.obs_vector_size = self.market_and_portfolio_state_size + self.action_space_size
        
        self._action_space = spaces.Discrete(self.action_space_size)
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_vector_size,), dtype=np.float32)
        self._reward_range = (-1.0, 1.0)

        self.current_step: int = 0
        self.total_steps: int = 0
        self.episode_time_to_expiry: int = 0
        self.iv_bin_index: int = 0
        self.final_eval_reward = 0.0
        self.illegal_action_count: int = 0
        self.realized_vol_series = np.array([])
        
        # --- THE FIX ---
        # Restoring the time constants needed by the orchestrator methods.
        self.TRADING_DAY_IN_MINS = self._cfg.trading_day_in_mins
        self.MINS_IN_DAY = 24 * 60
        self.TRADING_DAYS_IN_WEEK = 5
        self.TOTAL_DAYS_IN_WEEK = 7
        self.decay_per_step_trading = (self.TRADING_DAY_IN_MINS / self._cfg.steps_per_day) / self.MINS_IN_DAY
        self.decay_overnight = (self.MINS_IN_DAY - self.TRADING_DAY_IN_MINS) / self.MINS_IN_DAY
        self.decay_weekend = 2.0

    def seed(self, seed: int, dynamic_seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        self.price_manager.np_random = self.np_random
        return [seed]

    def reset(self, seed: int = None, **kwargs) -> Dict:
        """
        Resets the environment.
        PRIORITY 1: Uses a forced episode length if provided (for evaluation).
        PRIORITY 2: Uses a random episode length (for training).
        """
        if seed is not None: self.seed(seed)
        else:
            # If the framework or user does not provide a seed, we create a
            # random one. This is useful for standalone testing.
            # During normal training, the framework will ALWAYS provide a seed.
            new_seed = int(time.time())
            self.seed(new_seed)

        forced_length = self._cfg.get('forced_episode_length', 0)

        if forced_length > 0:
            # PRIORITY 1: Evaluation mode with a fixed, forced length.
            self.episode_time_to_expiry = forced_length
        else:
            # PRIORITY 2: Training mode with a random length.
            self.episode_time_to_expiry = random.randint(
                self._cfg.min_time_to_expiry_days,
                self._cfg.time_to_expiry_days
            )

        self.total_steps = self.episode_time_to_expiry * self._cfg.steps_per_day
        
        self.price_manager.reset(self.total_steps)
        self.portfolio_manager.reset()
        
        self.current_step = 0
        self.final_eval_reward = 0.0
        self.illegal_action_count = 0
        self.realized_vol_series = np.zeros(self.total_steps + 1, dtype=np.float32)
        self.iv_bin_index = random.randint(0, len(self.market_rules_manager.iv_bins['call']['0']) - 1)

        obs = self._get_observation()
        action_mask = self._get_true_action_mask() if not self._cfg.ignore_legal_actions else np.ones(self.action_space_size, dtype=np.int8)

        return {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

    def step(self, action: int) -> BaseEnvTimestep:
        """
        The complete, definitive step function incorporating all bug fixes and features.
        """
        # --- 1. Get State BEFORE Action ---
        equity_before = self.portfolio_manager.get_current_equity(self.price_manager.current_price, self.iv_bin_index)

        final_action, was_illegal_action = self._handle_action(action)
        
        # --- 2. Determine and Execute the Final Action ---
        final_action_name = self.indices_to_actions.get(final_action, 'INVALID')
        
        if final_action_name.startswith('OPEN_'):
            # Use the DYNAMIC episode length for this calculation, not the static max length.
            days_to_expiry_float = (self.episode_time_to_expiry - self.current_day_index) * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)
            days_to_expiry = int(round(days_to_expiry_float))
            self.portfolio_manager.open_strategy(final_action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step, days_to_expiry)
        elif final_action_name.startswith('CLOSE_POSITION_'):
            self.portfolio_manager.close_position(int(final_action_name.split('_')[-1]), self.price_manager.current_price, self.iv_bin_index)
        elif final_action_name == 'CLOSE_ALL':
            self.portfolio_manager.close_all_positions(self.price_manager.current_price, self.iv_bin_index)
        
        # This new, nested structure prevents mis-routing of SHIFT actions.
        elif final_action_name.startswith('SHIFT_'):
            if 'ATM' in final_action_name:
                self.portfolio_manager.shift_to_atm(final_action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step)
            else: # Must be UP or DOWN
                self.portfolio_manager.shift_position(final_action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step)

        elif final_action_name.startswith('HEDGE_POS_'):
            self.portfolio_manager.add_hedge(final_action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step)

        #print(f"\n--- Portfolio State After Hedge Status Update --- A:{final_action_name}")
        #print(self.portfolio_manager.portfolio.to_string())
        #print("-------------------------------------------------\n")

        self.portfolio_manager.sort_portfolio()
        self.portfolio_manager.take_post_action_portfolio_snapshot()

        # --- 3. Advance Time and Market (CORRECT ORDER) ---
        # First, calculate decay based on the CURRENT step.
        time_decay_days = self._calculate_time_decay()
        # THEN, increment the step counter.
        self.current_step += 1
        # Now, update the market to the new step.
        self.price_manager.step(self.current_step)
        self._update_realized_vol()
        # Finally, apply the correctly calculated decay to the portfolio.
        self.portfolio_manager.update_positions_after_time_step(time_decay_days, self.price_manager.current_price, self.iv_bin_index)

        # --- FINAL, THREE-TIERED TERMINATION LOGIC ---
        terminated_by_rule = False
        final_shaped_reward_override = None
        termination_reason = "RUNNING" # Default state

        current_pnl = self.portfolio_manager.get_total_pnl(self.price_manager.current_price, self.iv_bin_index)

        # 1. Stop-Loss Rule (Highest Priority)
        if self._cfg.use_stop_loss and not self.portfolio_manager.portfolio.empty:
            initial_cost = abs(self.portfolio_manager.initial_net_premium * self.portfolio_manager.lot_size)
            stop_loss_level = initial_cost * self._cfg.stop_loss_multiple_of_cost
            if current_pnl <= -stop_loss_level:
                terminated_by_rule = True
                final_shaped_reward_override = -1.0
                termination_reason = "STOP_LOSS"

        # 2. Take-Profit Rules (Only check if not already stopped out)
        if not terminated_by_rule and not self.portfolio_manager.portfolio.empty:

            # Rule 2a: Fixed Portfolio-Level "Home Run" Target
            fixed_target_pct = self._cfg.profit_target_pct
            if fixed_target_pct > 0:
                pnl_pct = (current_pnl / self.portfolio_manager.initial_cash) * 100
                if pnl_pct >= fixed_target_pct:
                    terminated_by_rule = True
                    final_shaped_reward_override = self._cfg.jackpot_reward
                    termination_reason = "TAKE_PROFIT"

            # Rule 2b: Dynamic Strategy-Level Targets (only check if home run not hit)
            if not terminated_by_rule:
                net_premium = self.portfolio_manager.initial_net_premium
                if net_premium < 0: # Credit strategy
                    credit_target_pct = self._cfg.credit_strategy_take_profit_pct
                    if credit_target_pct > 0:
                        max_profit = self.portfolio_manager.portfolio.iloc[0]['strategy_max_profit']
                        profit_target_pnl = max_profit * (credit_target_pct / 100)
                        if max_profit > 0 and current_pnl >= profit_target_pnl:
                            terminated_by_rule = True
                            final_shaped_reward_override = self._cfg.jackpot_reward
                            termination_reason = "TAKE_PROFIT"
                elif net_premium > 0: # Debit strategy
                    debit_target_multiple = self._cfg.debit_strategy_take_profit_multiple
                    if debit_target_multiple > 0:
                        debit_paid = net_premium * self.portfolio_manager.lot_size
                        profit_target_pnl = debit_paid * debit_target_multiple
                        if current_pnl >= profit_target_pnl:
                            terminated_by_rule = True
                            final_shaped_reward_override = self._cfg.jackpot_reward
                            termination_reason = "TAKE_PROFIT"

        # Final termination condition
        terminated_by_time = self.current_step >= self.total_steps
        terminated = terminated_by_rule or terminated_by_time

        if terminated and termination_reason == "RUNNING":
            termination_reason = "TIME_LIMIT"

        if terminated_by_rule: self.portfolio_manager.close_all_positions(self.price_manager.current_price, self.iv_bin_index)

        # The environment's job is to calculate the final PnL.
        # It does NOT prematurely clean up the portfolio. The logger will handle that.
        equity_after = self.portfolio_manager.get_current_equity(self.price_manager.current_price, self.iv_bin_index)
        shaped_reward, raw_reward = self._calculate_shaped_reward(equity_before, equity_after)
        self.final_eval_reward += raw_reward
        
        if final_shaped_reward_override is not None: final_reward = final_shaped_reward_override
        elif was_illegal_action: final_reward = self._cfg.illegal_action_penalty
        else: final_reward = shaped_reward

        # --- Prepare and Return Timestep ---
        obs = self._get_observation()
        action_mask = self._get_true_action_mask() if not self._cfg.ignore_legal_actions else np.ones(self.action_space_size, dtype=np.int8)
        meter = BiasMeter(obs[:self.market_and_portfolio_state_size], self.OBS_IDX)
        
        # The environment correctly reports the final executed name.
        if terminated_by_time: final_executed_name = "EXPIRATION"
        else: final_executed_name = self.indices_to_actions.get(final_action, 'INVALID')
        
        info = {
            'price': self.price_manager.current_price, 'eval_episode_return': self.final_eval_reward,
            'illegal_actions_in_episode': self.illegal_action_count, 'was_illegal_action': bool(was_illegal_action),
            'initial_cash': self.portfolio_manager.initial_cash,
            'executed_action_name': final_executed_name, 'directional_bias': meter.directional_bias,
            'volatility_bias': meter.volatility_bias,
            'portfolio_stats': self.portfolio_manager.get_raw_portfolio_stats(self.price_manager.current_price, self.iv_bin_index),
            'market_regime': self.price_manager.current_regime_name,
            'termination_reason': termination_reason
        }

        if terminated:
            info['episode_duration'] = self.current_step
        
        # The portfolio is intentionally left intact for the logger to record.
        return BaseEnvTimestep({'observation': obs, 'action_mask': action_mask, 'to_play': -1}, final_reward, terminated, info)

    def _handle_action(self, action: int) -> Tuple[int, bool]:
        """
        The definitive, mode-aware action handler.
        It now correctly uses the _get_true_action_mask as the source of truth for all logic.
        """
        # --- First, get the official list of all legal moves ---
        true_action_mask = self._get_true_action_mask()

        # PRIORITY 1: Handle forced strategy for analysis
        forced_strategy = self._cfg.get('forced_opening_strategy_name')
        if self.current_step == 0 and forced_strategy:
            action_index = self.actions_to_indices.get(forced_strategy)
            assert action_index is not None and true_action_mask[action_index] == 1, \
                f"Forced strategy '{forced_strategy}' is illegal under current market conditions."
            return action_index, False

        # PRIORITY 2: Handle the critical Step 0 logic
        if self.current_step == 0:
            # Get the list of all legal opening moves from the official rulebook
            legal_open_indices = [
                idx for name, idx in self.actions_to_indices.items()
                if name.startswith('OPEN_') and true_action_mask[idx] == 1
            ]
            
            # Case A: Evaluation Mode (Agent's Choice)
            if self._cfg.disable_opening_curriculum:
                # The agent's action is valid if it's in the legal list.
                if action in legal_open_indices:
                    return action, False
                else:
                    # If the agent chose an illegal move (e.g., HOLD), override with a random valid one.
                    return random.choice(legal_open_indices), True # It was an illegal *attempt*
            
            # Case B: Training Mode (The Curriculum)
            else:
                # The curriculum now works with the pre-filtered, legal list of actions.
                strategy_families = {
                    "SINGLE_LEG": lambda name: 'ATM' in name and 'STRADDLE' not in name and 'STRANGLE' not in name,
                    "STRADDLE": lambda name: 'STRADDLE' in name, "STRANGLE": lambda name: 'STRANGLE' in name,
                    "VERTICAL": lambda name: 'VERTICAL' in name, "IRON_FLY_CONDOR": lambda name: 'IRON' in name,
                    "BUTTERFLY": lambda name: 'FLY' in name and 'IRON' not in name,
                }
                chosen_family_name = random.choice(list(strategy_families.keys()))
                is_in_family = strategy_families[chosen_family_name]

                # Filter the already-legal moves by the chosen family
                family_actions = [idx for idx in legal_open_indices if is_in_family(self.indices_to_actions[idx])]
                
                # Failsafe: if the family has no legal moves, pick from any legal opening
                if not family_actions:
                    family_actions = legal_open_indices
                
                final_action = random.choice(family_actions)
                return final_action, False

        # PRIORITY 3: Standard logic for all other steps (current_step > 0)
        is_illegal = true_action_mask[action] == 0
        if is_illegal:
            self.illegal_action_count += 1
            # Special override rule for the liquidation period.
            is_liquidation_period = self.current_day_index >= (self.episode_time_to_expiry - 2)
            if is_liquidation_period and not self.portfolio_manager.portfolio.empty:
                # If the agent makes a mistake during liquidation, the only rational
                # override is to force it to do what it's supposed to do: get flat.
                final_action = self.actions_to_indices['CLOSE_ALL']
                print("DEBUG: Illegal action in liquidation period. Forcing CLOSE_ALL.") # Optional debug print
            else:
                # The standard fallback for all other illegal moves.
                final_action = self.actions_to_indices['HOLD']

            return final_action, True # Return the override and flag the illegal attempt
        else:
            return action, False

    def _get_observation(self) -> np.ndarray:
        # Market State
        vec = np.zeros(self.market_and_portfolio_state_size, dtype=np.float32)
        vec[self.OBS_IDX['PRICE_NORM']] = (self.price_manager.current_price / self.price_manager.start_price) - 1.0
        vec[self.OBS_IDX['TIME_NORM']] = (self.total_steps - self.current_step) / self.total_steps
        vec[self.OBS_IDX['PNL_NORM']] = math.tanh(self.portfolio_manager.get_total_pnl(self.price_manager.current_price, self.iv_bin_index) / self._cfg.initial_cash)
        garch_vol = self.price_manager.garch_implied_vol
        vec[self.OBS_IDX['VOL_MISMATCH_NORM']] = math.tanh((self.realized_vol_series[self.current_step] / garch_vol) - 1.0) if garch_vol > 0 else 0.0
        log_return = math.log(self.price_manager.current_price / (self.price_manager.price_path[self.current_step - 1] + 1e-8)) if self.current_step > 0 else 0.0
        vec[self.OBS_IDX['LOG_RETURN']] = np.clip(log_return, -0.1, 0.1) * 10
        vec[self.OBS_IDX['MOMENTUM_NORM']] = self.price_manager.momentum_signal

        # --- NEW: Calculate the Market's Expected Move ---
        current_price = self.price_manager.current_price
        
        # We need a representative DTE for the market. The time left in the episode is a perfect proxy.
        days_to_expiry = (self.episode_time_to_expiry - self.current_day_index) * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)
        
        expected_move = 0.0
        if days_to_expiry > 0:
            # We use the ATM volatility as the market's consensus IV.
            atm_iv = self.market_rules_manager.get_implied_volatility(offset=0, option_type='call', iv_bin_index=self.iv_bin_index)
            
            # The formula: Price * IV * sqrt(DTE / 365)
            expected_move_points = current_price * atm_iv * math.sqrt(days_to_expiry / 365.25)
            
            # Normalize the move by the current price to get a stable percentage.
            # This prevents huge numbers from destabilizing the network.
            expected_move = expected_move_points / current_price
        
        vec[self.OBS_IDX['EXPECTED_MOVE_NORM']] = expected_move

        # Portfolio Greeks
        greeks = self.portfolio_manager.get_portfolio_greeks(self.price_manager.current_price, self.iv_bin_index)
        vec[self.OBS_IDX['PORTFOLIO_DELTA']] = greeks['delta_norm']
        vec[self.OBS_IDX['PORTFOLIO_GAMMA']] = greeks['gamma_norm']
        vec[self.OBS_IDX['PORTFOLIO_THETA']] = greeks['theta_norm']
        vec[self.OBS_IDX['PORTFOLIO_VEGA']] = greeks['vega_norm']

        summary = self.portfolio_manager.get_portfolio_summary(self.price_manager.current_price, self.iv_bin_index)

        # Normalize the values before adding them to the state
        vec[self.OBS_IDX['PORTFOLIO_MAX_PROFIT_NORM']] = math.tanh(summary['max_profit'] / self._cfg.initial_cash)
        vec[self.OBS_IDX['PORTFOLIO_MAX_LOSS_NORM']] = math.tanh(summary['max_loss'] / self._cfg.initial_cash)
        # RR Ratio is often spiky, tanh is a good choice
        vec[self.OBS_IDX['PORTFOLIO_RR_RATIO_NORM']] = math.tanh(summary['rr_ratio'])
        # Probability is already in a [0, 1] range, we can scale it to [-1, 1]
        vec[self.OBS_IDX['PORTFOLIO_PROB_PROFIT']] = (summary['prob_profit'] * 2) - 1.0

        # --- NEW: Add Holy Trinity predictions ---
        vec[self.OBS_IDX['EXPERT_EMA_RATIO']] = math.tanh(self.price_manager.expert_ema_pred - 1.0)
        
        rsi_probs = self.price_manager.expert_rsi_pred
        vec[self.OBS_IDX['EXPERT_RSI_OVERSOLD']] = rsi_probs[0]
        vec[self.OBS_IDX['EXPERT_RSI_NEUTRAL']] = rsi_probs[1]
        vec[self.OBS_IDX['EXPERT_RSI_OVERBOUGHT']] = rsi_probs[2]
        
        vec[self.OBS_IDX['EXPERT_VOL_NORM']] = math.tanh(self.price_manager.expert_vol_pred)

        # Per-Position State
        self.portfolio_manager.get_positions_state(vec, self.PORTFOLIO_START_IDX, self.PORTFOLIO_STATE_SIZE_PER_POS, self.POS_IDX, self.price_manager.current_price, self.iv_bin_index, self.current_step, self.total_steps)

        # 1. First, concatenate the state vector and the action mask
        true_action_mask = self._get_true_action_mask()
        final_obs_vec = np.concatenate((vec, true_action_mask.astype(np.float32)))

        # 2. Then, run assertions on the FINAL object
        assert final_obs_vec.shape == (self.obs_vector_size,), f"Observation shape mismatch. Expected {self.obs_vector_size}, but got {final_obs_vec.shape}"
        assert np.all(np.isfinite(final_obs_vec)), "Observation vector contains NaN or Inf."

        # 3. Return the fully validated final vector
        return final_obs_vec

    def _calculate_shaped_reward(self, equity_before: float, equity_after: float) -> Tuple[float, float]:
        raw_reward = equity_after - equity_before
        self.portfolio_manager.update_high_water_mark(equity_after)
        drawdown = self.portfolio_manager.high_water_mark - equity_after
        risk_adjusted_raw_reward = raw_reward - (self._cfg.drawdown_penalty_weight * drawdown)
        scaled_reward = risk_adjusted_raw_reward / self._cfg.pnl_scaling_factor

        # --- Defensive Assertion ---
        assert math.isfinite(scaled_reward), f"Calculated a non-finite reward: {scaled_reward}"
        return math.tanh(scaled_reward), raw_reward

    def _calculate_time_decay(self) -> float:
        """
        Calculates the total time decay in days for the current step.
        This version correctly accounts for the Friday overnight period.
        """
        # The decay from the trading session itself
        time_decay = self.decay_per_step_trading

        # We only add overnight/weekend decay at the end of a trading day.
        is_end_of_day = (self.current_step + 1) % self._cfg.steps_per_day == 0
        if not is_end_of_day:
            return time_decay # Return only the intra-day decay

        # --- End of Day Logic ---
        # Add the standard overnight decay for every day
        time_decay += self.decay_overnight

        # Determine the day of the week to check for a weekend
        day_of_week = self.current_day_index % self.TRADING_DAYS_IN_WEEK

        is_friday = (day_of_week == 4)

        # If it's Friday, add the additional 2 full days for the weekend
        if is_friday:
            time_decay += self.decay_weekend

        return time_decay

    def _update_realized_vol(self):
        window = self._cfg.rolling_vol_window * self._cfg.steps_per_day
        start = max(0, self.current_step - window)
        prices = self.price_manager.price_path[start:self.current_step + 1]
        if len(prices) > 1:
            log_returns = np.diff(np.log(prices))
            vol = np.std(log_returns) * np.sqrt(252 * self._cfg.steps_per_day)
            self.realized_vol_series[self.current_step] = vol if np.isfinite(vol) else 0.0

    def _build_action_space(self) -> Dict[str, int]:
        actions = {'HOLD': 0}; i = 1

        # Use the configurable max_strike_offset instead of a hardcoded range.
        for offset in range(-self._cfg.max_strike_offset, self._cfg.max_strike_offset + 1):
            sign = '+' if offset >= 0 else ''
            # We can simplify the action name for offsets > 9, e.g., ATM+10 instead of ATM+ 10
            offset_str = f"{sign}{offset}"
            for t in ['CALL', 'PUT']:
                for d in ['LONG', 'SHORT']:
                    actions[f'OPEN_{d}_{t}_ATM{offset_str}'] = i; i+=1

        for d in ['LONG', 'SHORT']:
            actions[f'OPEN_{d}_STRADDLE_ATM'] = i; i+=1

        for w in [1, 2]:
            for d in ['LONG', 'SHORT']:
                actions[f'OPEN_{d}_STRANGLE_ATM_{w}'] = i; i+=1

        # Generate a range of delta-based strangle actions.
        for delta in range(15, 31, 5): # This creates [15, 20, 25, 30]
            actions[f'OPEN_LONG_STRANGLE_DELTA_{delta}'] = i; i+=1
            actions[f'OPEN_SHORT_STRANGLE_DELTA_{delta}'] = i; i+=1

        for w in [1, 2]:
            for t in ['CALL', 'PUT']:
                for d in ['LONG', 'SHORT']:
                    actions[f'OPEN_{d}_VERTICAL_{t}_{w}'] = i; i+=1

        for d in ['LONG', 'SHORT']:
            actions[f'OPEN_{d}_IRON_FLY'] = i; i+=1
            actions[f'OPEN_{d}_IRON_CONDOR'] = i; i+=1

        for w in [1, 2]:
            for t in ['CALL', 'PUT']:
                for d in ['LONG', 'SHORT']:
                    actions[f'OPEN_{d}_{t}_FLY_{w}'] = i; i+=1

        for j in range(self._cfg.max_positions):
            actions[f'CLOSE_POSITION_{j}'] = i; i+=1

        for j in range(self._cfg.max_positions):
            actions[f'SHIFT_UP_POS_{j}'] = i; i+=1
            actions[f'SHIFT_DOWN_POS_{j}'] = i; i+=1

        # New actions to re-center a position directly to the ATM strike.
        for j in range(self._cfg.max_positions):
            actions[f'SHIFT_TO_ATM_{j}'] = i; i+=1

        # New actions to hedge an existing naked position.
        for j in range(self._cfg.max_positions):
            actions[f'HEDGE_POS_{j}'] = i; i+=1

        actions['CLOSE_ALL'] = i
        return actions
        
    def _get_true_action_mask(self) -> np.ndarray:
        """
        The definitive, correct implementation of the action mask, with a flag to disable
        the opening curriculum for true evaluation.
        """
        action_mask = np.zeros(self.action_space_size, dtype=np.int8)

        # --- Rule 0: Forced Strategy for Analysis (Highest Priority) ---
        forced_strategy = self._cfg.get('forced_opening_strategy_name')
        if self.current_step == 0 and forced_strategy:
            action_index = self.actions_to_indices.get(forced_strategy)
            assert action_index is not None, f"Forced strategy '{forced_strategy}' is invalid."
            action_mask[action_index] = 1
            return action_mask

        # --- Rule 1: Step 0 - Training Curriculum OR Agent's Choice ---
        if self.current_step == 0:
            # For BOTH training and evaluation, the agent MUST open a position.
            # HOLD is never a legal first move.
            for name, index in self.actions_to_indices.items():
                if name.startswith('OPEN_'):
                    action_mask[index] = 1
            return action_mask

        # A day index of N-1 is the final day. N-2 is the second to last day.
        # We will make the last two days (N-2 and N-1) liquidation-only.
        is_liquidation_period = self.current_day_index >= (self.episode_time_to_expiry - 2)
        
        if is_liquidation_period:
            # If we are in the final two days, the agent's job is to get flat.
            if self.portfolio_manager.portfolio.empty:
                # If already flat, the only choice is to wait for the episode to end.
                action_mask[self.actions_to_indices['HOLD']] = 1
            else:
                # If there are positions, the agent MUST close them. Holding is not an option.
                action_mask[self.actions_to_indices['CLOSE_ALL']] = 1
                for i in range(len(self.portfolio_manager.portfolio)):
                    if f'CLOSE_POSITION_{i}' in self.actions_to_indices:
                        action_mask[self.actions_to_indices[f'CLOSE_POSITION_{i}']] = 1
            return action_mask

        # --- Rule 3: Definitive Logic for an Empty Portfolio (Covers Step 0 and Mid-Episode) ---
        # This is the only path left if the portfolio is empty on a non-liquidation day.
        
        # 1. Create the BASE MASK of all opening actions that are legal according to the delta rules.
        base_opening_mask = np.zeros(self.action_space_size, dtype=np.int8)
        atm_price = self.market_rules_manager.get_atm_price(self.price_manager.current_price)
        days_to_expiry = (self.episode_time_to_expiry - self.current_day_index) * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)
        
        for action_name, index in self.actions_to_indices.items():
            if not action_name.startswith('OPEN_'): continue
            
            is_legal = True
            is_single_leg = 'ATM' in action_name and 'STRADDLE' not in action_name and 'STRANGLE' not in action_name
            
            if is_single_leg:
                parts = action_name.split('_')
                direction, option_type, offset_str = parts[1], parts[2].lower(), parts[3].replace('ATM','')
                offset = int(offset_str)
                strike_price = atm_price + (offset * self._cfg.strike_distance)
                vol = self.market_rules_manager.get_implied_volatility(offset, option_type, self.iv_bin_index)
                greeks = self.bs_manager.get_all_greeks_and_price(self.price_manager.current_price, strike_price, days_to_expiry, vol, option_type == 'call')
                abs_delta = abs(greeks['delta'])

                if direction == 'LONG':
                    if abs_delta < self._cfg.otm_long_delta_threshold or abs_delta > self._cfg.itm_long_delta_threshold:
                        is_legal = False
                elif direction == 'SHORT':
                    if abs_delta > self._cfg.itm_short_delta_threshold:
                        is_legal = False
            
            if is_legal:
                base_opening_mask[index] = 1

        # 2. Now, filter this universe based on the number of available slots.
        final_opening_mask = np.zeros(self.action_space_size, dtype=np.int8)
        available_slots = self.portfolio_manager.max_positions - len(self.portfolio_manager.portfolio)

        for index, is_legal in enumerate(base_opening_mask):
            if not is_legal: continue # Skip if it's already illegal by delta rules
            
            action_name = self.indices_to_actions[index]
            
            # Check for 4-leg strategies
            if 'CONDOR' in action_name or 'FLY' in action_name:
                if available_slots >= 4:
                    final_opening_mask[index] = 1
            # Check for 2-leg strategies
            elif 'STRADDLE' in action_name or 'STRANGLE' in action_name or 'VERTICAL' in action_name:
                if available_slots >= 2:
                    final_opening_mask[index] = 1
            # Check for 1-leg strategies
            elif 'ATM' in action_name:
                if available_slots >= 1:
                    final_opening_mask[index] = 1

        # 3. If it's Step 0 and we are in TRAINING mode, apply the curriculum filter.
        if self.current_step == 0 and not self._cfg.disable_opening_curriculum:
            strategy_families = {
                "SINGLE_LEG": lambda name: 'ATM' in name and 'STRADDLE' not in name and 'STRANGLE' not in name,
                "STRADDLE": lambda name: 'STRADDLE' in name, "STRANGLE": lambda name: 'STRANGLE' in name,
                "VERTICAL": lambda name: 'VERTICAL' in name, "IRON_FLY_CONDOR": lambda name: 'IRON' in name,
                "BUTTERFLY": lambda name: 'FLY' in name and 'IRON' not in name,
            }
            chosen_family_name = random.choice(list(strategy_families.keys()))
            is_in_family = strategy_families[chosen_family_name]

            # Filter the already-legal moves by the chosen family
            curriculum_mask = np.zeros(self.action_space_size, dtype=np.int8)
            for index, is_legal in enumerate(final_opening_mask):
                if is_legal and is_in_family(self.indices_to_actions[index]):
                    curriculum_mask[index] = 1
            
            # Failsafe: if the filter results in no legal moves, use the original filtered mask
            if not np.any(curriculum_mask):
                return final_opening_mask
            else:
                return curriculum_mask
        else:
            # For EVALUATION on Step 0, or any MID-EPISODE re-opening, return the full set of slot-legal moves.
            # We also allow HOLD if it's not step 0.
            if self.current_step > 0:
                final_opening_mask[self.actions_to_indices['HOLD']] = 1
            return final_opening_mask

    def render(self, mode: str = 'human') -> None:
        total_pnl = self.portfolio_manager.get_total_pnl(self.price_manager.current_price, self.iv_bin_index)
        print(f"\nStep: {self.current_step:04d} | Day: {self.current_day:02d} | Price: ${self.price_manager.current_price:9.2f} | Positions: {len(self.portfolio_manager.portfolio):1d} | Total PnL: ${total_pnl:9.2f}")
        
        if not self.portfolio_manager.portfolio.empty:
            print(self.portfolio_manager.portfolio.to_string(index=False))

        # --- NEW: Add the Bias Meter Summary ---
        # We get the observation vector but slice off the action mask at the end.
        current_obs_vector = self._get_observation()[:self.market_and_portfolio_state_size]
        meter = BiasMeter(current_obs_vector, self.OBS_IDX)
        meter.summary()

    # --- Properties and Static Methods ---
    @property
    def jackpot_reward(self) -> float:
        """A safe, public, read-only property to access jackpot_reward."""
        return self._cfg.jackpot_reward
    @property
    def bid_ask_spread_pct(self) -> float:
        """A safe, public, read-only property to access bid_ask_pread_pct."""
        return self._cfg.bid_ask_spread_pct
    @property
    def strike_distance(self) -> float:
        """A safe, public, read-only property to access strike_distance."""
        return self._cfg.strike_distance
    @property
    def current_day_index(self) -> int:
        """A safe, public, read-only property to get the current day index (0-indexed)."""
        return int(self.current_step // self.steps_per_day)
    @property
    def current_day(self) -> int:
        """A safe, public, read-only property to get the current day of the episode (1-indexed)."""
        return self.current_day_index + 1
    @property
    def steps_per_day(self) -> int:
        """A safe, public, read-only property to access steps_per_day."""
        return self._cfg.steps_per_day
    @property
    def legal_actions(self): return self._get_true_action_mask()
    @property
    def observation_space(self) -> spaces.Space: return self._observation_space
    @property
    def action_space(self) -> spaces.Space: return self._action_space
    @property
    def reward_space(self) -> spaces.Space: return self._reward_range
    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> list:
        collector_env_num = cfg.pop('collector_env_num')
        return [copy.deepcopy(cfg) for _ in range(collector_env_num)]
    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> list:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        return [copy.deepcopy(cfg) for _ in range(evaluator_env_num)]
    def __repr__(self): return f"LightZero Options-Zero-Game Env v{self.VERSION}"
