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
from .utils import generate_dynamic_iv_skew_table

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
        max_strike_offset=30,
        bid_ask_spread_pct=0.002,
        brokerage_per_leg=25.0,
        days_before_liquidation=1,
        
        # Black-Scholes Manager Config
        risk_free_rate=0.10,

        # Reward and Penalty Config
        pnl_scaling_factor=1000,
        drawdown_penalty_weight=0.1,
        illegal_action_penalty=-1.0,
        
        # Advanced Trading Rules
        profit_target_pct=100.0,

        #Close the short leg if the premium is below 2.0
        close_short_leg_on_profit_threshold=2.0,
        jackpot_reward=1.0,

        # For credit strategies, target is a % of the max possible profit (the credit received).
        credit_strategy_take_profit_pct=0, # Target 25% of max profit. Set to 0 to disable.
        
        # For debit strategies, target is a multiple of the initial debit paid.
        debit_strategy_take_profit_multiple=0, # Target 2x the debit paid (200% return). Set to 0 to disable.

        stop_loss_multiple_of_cost=3.0, # NEW: Added stop loss multiple
        use_stop_loss=True,
        forced_opening_strategy_name=None,
        disable_opening_curriculum=True,
        
        # Agent/Framework Config
        ignore_legal_actions=True,

        # These rules apply ONLY when opening a new LONG single-leg position.
        otm_long_delta_threshold=0.45, # Disallow buying options with delta < 45
        itm_long_delta_threshold=0.55, # Disallow buying options with delta > 55
        itm_short_delta_threshold=0.55, # Disallow shorting options with delta > 55
        short_leg_max_offset=3,

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

        # This dictionary is passed in the config and is needed by both the
        # environment (for the action mask) and the portfolio manager.
        self.strategy_name_to_id = cfg.get('strategy_name_to_id', {})

        self.iv_regimes = self._cfg.get('iv_regimes', [])
        self.current_iv_regime_name = "N/A" # For logging
        self.current_iv_regime_index = 0 # Start with a default
        
        # <<< NEW: Load the Markov Chain model >>>
        try:
            self.iv_transition_matrix = np.load("zoo/options_zero_game/experts/iv_transition_matrix.npy")
            self.iv_stationary_dist = np.load("zoo/options_zero_game/experts/iv_stationary_distribution.npy")
            print("Successfully loaded IV Regime Markov Chain model.")
        except FileNotFoundError:
            print("WARNING: IV Regime model not found. Run iv_regime_analyzer.py. Falling back to random choice.")
            self.iv_transition_matrix = None
            self.iv_stationary_dist = None

        self.bs_manager = BlackScholesManager(self._cfg)
        self.price_manager = PriceActionManager(self._cfg, self.np_random)
        self.market_rules_manager = None
        
        # The PortfolioManager needs references to the other managers
        self.portfolio_manager = None
        
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
            'EXPECTED_MOVE_NORM': 19, 'PORTFOLIO_PROFIT_FACTOR_NORM': 20,
            # --- Realized and MtM Highest and Lowest Profit/Loss
            'MTM_PNL_HIGH_NORM': 21, 'MTM_PNL_LOW_NORM': 22,
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
        self.last_action_info: Dict = {}

        # --- NEW: Curriculum Learning Setup ---
        curriculum_holder = cfg.get('training_curriculum', None)
        self.training_curriculum = curriculum_holder.schedule if curriculum_holder else None
        self.is_training_mode = not cfg.get('is_eval_mode', False)
        #only ignore legal actions if in training mode
        if self.is_training_mode:
            self._cfg.ignore_legal_actions = True
        else:
            self._cfg.ignore_legal_actions = False

        # We need a way to track the approximate training step.
        # We'll use the episode count as a proxy.
        self._episode_count = 0 
        self.volatility_premium_abs = cfg.get('volatility_premium_abs', 0.05)
        self.min_pop_threshold = cfg.get('min_pop_threshold', 0.25)
        self.low_pop_penalty = cfg.get('low_pop_opening_penalty', 0.0)
        self.delta_neutral_threshold = cfg.get('delta_neutral_threshold', 0.1)

    def seed(self, seed: int, dynamic_seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        self.price_manager.np_random = self.np_random
        return [seed]

    def reset(self, seed: int = None, **kwargs) -> Dict:
        """
        The definitive, correct reset method. It correctly prioritizes and
        sequences all initialization logic for a new episode.
        """
        # --- 1. Seeding and Basic State Reset ---
        if seed is not None: self.seed(seed)
        else: self.seed(int(time.time()))
        
        self.current_step = 0
        self.final_eval_reward = 0.0
        self.illegal_action_count = 0

        # --- 2. Determine Forced Opening Strategy (from Curriculum or Config) ---
        if self.is_training_mode and self.training_curriculum:
            self._episode_count += 1
            approx_current_step = self._episode_count * self.total_steps
            active_strategy = 'ALL'
            sorted_phases = sorted(self.training_curriculum.keys())
            for start_step in sorted_phases:
                if approx_current_step >= start_step:
                    active_strategy = self.training_curriculum[start_step]
                else: break
            self.forced_opening_strategy_name = active_strategy if active_strategy != 'ALL' else None
        else:
            self.forced_opening_strategy_name = self._cfg.get('forced_opening_strategy_name')

        # --- 3. Determine Episode Length ---
        forced_length = self._cfg.get('forced_episode_length', 0)
        if forced_length > 0:
            self.episode_time_to_expiry = forced_length
        else:
            self.episode_time_to_expiry = random.randint(self._cfg.min_time_to_expiry_days, self._cfg.time_to_expiry_days)
        self.total_steps = self.episode_time_to_expiry * self._cfg.steps_per_day

        # --- 4. Select IV Regime and Initialize Market Managers ---
        # Select the starting IV regime for this episode
        if self.iv_stationary_dist is not None:
            self.current_iv_regime_index = np.random.choice(len(self.iv_regimes), p=self.iv_stationary_dist)
        else:
            self.current_iv_regime_index = random.randint(0, len(self.iv_regimes) - 1)
        
        # This single helper now correctly creates the MarketRulesManager.
        self._update_market_rules_for_regime()

        # --- 5. Initialize the Portfolio Manager ---
        # This must happen AFTER the MarketRulesManager is created.
        self.portfolio_manager = PortfolioManager(
            cfg=self._cfg,
            bs_manager=self.bs_manager,
            market_rules_manager=self.market_rules_manager,
            iv_calculator_func=self._get_dynamic_iv,
            strategy_name_to_id=self.strategy_name_to_id
        )
        # Note: We do NOT call self.portfolio_manager.reset() because it's a new object.
        
        # --- 6. Reset Remaining State and Get Initial Observation ---
        self.price_manager.reset(self.total_steps)
        self.high_water_mark = self._cfg.initial_cash
        self.iv_bin_index = random.randint(0, len(self.market_rules_manager.iv_bins['call']['0']) - 1)
        self.realized_vol_series = np.zeros(self.total_steps + 1, dtype=np.float32)

        obs = self._get_observation()
        action_mask = self._get_true_action_mask() if not self._cfg.ignore_legal_actions else np.ones(self.action_space_size, dtype=np.int8)

        return {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

    # <<< NEW: Add a helper to handle day changes >>>
    def _handle_day_change(self):
        """Called when a day passes. Evolves the IV regime using the Markov chain."""
        if self.iv_transition_matrix is not None:
            # Get the probability distribution for the next state
            transition_probs = self.iv_transition_matrix[self.current_iv_regime_index]
            # Choose the next regime based on these probabilities
            self.current_iv_regime_index = np.random.choice(len(self.iv_regimes), p=transition_probs)
            
            # Now that the regime has changed, we must rebuild the market rules
            self._update_market_rules_for_regime()
            # We also must re-initialize the PortfolioManager's IV calculator
            self.portfolio_manager.iv_calculator = self._get_dynamic_iv

    # <<< NEW: Create a helper to generate the skew table on the fly >>>
    def _update_market_rules_for_regime(self):
        """Generates and applies the skew table for the current IV regime."""
        chosen_regime = self.iv_regimes[self.current_iv_regime_index]
        self.current_iv_regime_name = chosen_regime['name']
        
        episode_skew_table = generate_dynamic_iv_skew_table(
            max_offset=self._cfg.max_strike_offset,
            atm_iv=chosen_regime['atm_iv'],
            far_otm_put_iv=chosen_regime['far_otm_put_iv'],
            far_otm_call_iv=chosen_regime['far_otm_call_iv']
        )
        
        temp_cfg = self._cfg.copy()
        temp_cfg['iv_skew_table'] = episode_skew_table
        self.market_rules_manager = MarketRulesManager(temp_cfg)

    def step(self, action: int) -> BaseEnvTimestep:
        """
        The main step function, now acting as a clean orchestrator.
        """
        self.portfolio_manager.receipts_for_current_step = []

        # 1. Get the state before any changes.
        equity_before = self.portfolio_manager.get_current_equity(self.price_manager.current_price, self.iv_bin_index)
        
        # 2. Execute the agent's action on the current state.
        self._take_action_on_state(action)
        
        # 3. Advance the market and get the final outcome.
        return self._advance_market_and_get_outcome(equity_before)

    def _get_dynamic_iv(self, offset: int, option_type: str) -> float:
        """
        Calculates the dynamic IV using the pre-trained Volatility Expert's
        forward-looking prediction. This is a more realistic and sophisticated
        approach to modeling implied volatility.
        """
        # 1. Get the floor IV from the static skew table.
        # This preserves the "volatility smile" and provides a safe minimum.
        iv_from_table = self.market_rules_manager.get_implied_volatility(
            offset, option_type, self.iv_bin_index
        )
        
        # 2. Get the expert's forward-looking volatility prediction.
        # This value is already calculated and stored in the PriceActionManager every step.
        predicted_vol = self.price_manager.expert_vol_pred
        
        # We need to ensure the predicted vol is a sensible number (e.g., between 5% and 300%)
        # and has a default value for the initial steps before the expert has enough data.
        if predicted_vol is None or not (0.05 < predicted_vol < 3.0):
            # Fallback to a simple default if the expert isn't ready
            predicted_vol = 0.20 

        # 3. Calculate the dynamic base IV by adding the market's risk premium.
        base_iv = predicted_vol + self.volatility_premium_abs
        
        # 4. Return the greater of the dynamic base and the static floor from the table.
        # This elegantly combines the forward-looking ATM prediction with the static skew.
        return max(base_iv, iv_from_table)

    def _take_action_on_state(self, action: int):
        """
        PART 1 of the step process. It takes the agent's action, enforces all rules,
        updates the portfolio,
        It does NOT advance time or the market price.
        """
        # 1. Determine the final action and if the agent's attempt was illegal.
        final_action, was_illegal_action = self._handle_action(action)
        final_action_name = self.indices_to_actions.get(final_action, 'INVALID')

        # 2. Store this information for the second half of the step.
        self.last_action_info = {
            'final_action': final_action,
            'final_action_name': final_action_name,
            'total_steps_in_episode': self.total_steps,
            'was_illegal_action': was_illegal_action
        }

        # 3. Execute the final action, which modifies the portfolio.
        if final_action_name.startswith('OPEN_'):
            current_day = self.current_step // self._cfg.steps_per_day
            days_to_expiry_float = (self.episode_time_to_expiry - current_day) * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)
            days_to_expiry = int(round(days_to_expiry_float))
            self.portfolio_manager.open_strategy(final_action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step, days_to_expiry)
        elif final_action_name.startswith('CLOSE_POSITION_'):
            self.portfolio_manager.close_position(int(final_action_name.split('_')[-1]), self.price_manager.current_price, self.iv_bin_index)
        elif final_action_name == 'CLOSE_ALL':
            self.portfolio_manager.close_all_positions(self.price_manager.current_price, self.iv_bin_index)
        elif final_action_name.startswith('SHIFT_'):
            if 'ATM' in final_action_name: self.portfolio_manager.shift_to_atm(final_action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step)
            else: self.portfolio_manager.shift_position(final_action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step)
        elif final_action_name == 'ADJUST_TO_DELTA_NEUTRAL':
            self.portfolio_manager.adjust_to_delta_neutral(
                self.price_manager.current_price, self.iv_bin_index, self.current_step
            )
        elif final_action_name == 'CONVERT_TO_IRON_CONDOR':
            self.portfolio_manager.convert_to_iron_condor(
                self.price_manager.current_price, self.iv_bin_index, self.current_step
            )
        elif final_action_name == 'CONVERT_TO_IRON_FLY':
            self.portfolio_manager.convert_to_iron_condor(
                self.price_manager.current_price, self.iv_bin_index, self.current_step
            )
        elif final_action_name == 'CONVERT_TO_STRANGLE':
            self.portfolio_manager.convert_to_strangle(
                self.price_manager.current_price, self.iv_bin_index, self.current_step
            )
        elif final_action_name == 'CONVERT_TO_STRADDLE':
            self.portfolio_manager.convert_to_straddle(
                self.price_manager.current_price, self.iv_bin_index, self.current_step
            )

        # 4. Sort the portfolio and take the crucial snapshot.
        self.portfolio_manager.sort_portfolio()

    def _advance_market_and_get_outcome(self, equity_before: float) -> BaseEnvTimestep:
        """
        PART 2 of the step process. It advances time and the market, calculates
        termination conditions and rewards, and returns the final timestep.
        """
        # --- Advance Time and Market ---
        previous_day_index = self.current_day_index
        self.current_step += 1
        if self.current_day_index > previous_day_index:
            self._handle_day_change()
        time_decay_days = self._calculate_time_decay()
        self.price_manager.step(self.current_step)
        self._update_realized_vol()
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
                    # We use the same credit_strategy_take_profit_pct for the max profit ratio
                    max_profit_target_pct = self._cfg.credit_strategy_take_profit_pct 
                    
                    if debit_target_multiple > 0 or max_profit_target_pct > 0:
                        # Target A: Based on a multiple of the debit paid
                        debit_paid = net_premium * self.portfolio_manager.lot_size
                        target_from_multiple = debit_paid * debit_target_multiple
                        
                        # Target B: Based on a percentage of the max possible profit
                        max_profit = self.portfolio_manager.portfolio.iloc[0]['strategy_max_profit']
                        target_from_ratio = max_profit * (max_profit_target_pct / 100)
                        
                        # The final, achievable target is the MINIMUM of the two.
                        # We only consider targets that are enabled (value > 0)
                        possible_targets = []
                        if debit_target_multiple > 0: possible_targets.append(target_from_multiple)
                        if max_profit_target_pct > 0: possible_targets.append(target_from_ratio)
                        
                        if possible_targets:
                            profit_target_pnl = min(possible_targets)
                            if max_profit > 0 and current_pnl >= profit_target_pnl:
                                terminated_by_rule = True
                                final_shaped_reward_override = self._cfg.jackpot_reward
                                termination_reason = "TAKE_PROFIT"

        # Final termination condition
        terminated_by_time = self.current_step >= self.total_steps
        terminated = terminated_by_rule or terminated_by_time

        if terminated and termination_reason == "RUNNING":
            termination_reason = "TIME_LIMIT"

        if terminated_by_rule: self.portfolio_manager.close_all_positions(self.price_manager.current_price, self.iv_bin_index)

        # --- Calculate Reward ---
        equity_after = self.portfolio_manager.get_current_equity(self.price_manager.current_price, self.iv_bin_index)
        shaped_reward, raw_reward = self._calculate_shaped_reward(equity_before, equity_after)
        self.final_eval_reward += raw_reward

        # Use the stored action info to determine the final reward
        was_illegal_action = self.last_action_info['was_illegal_action']
        if final_shaped_reward_override is not None: final_reward = final_shaped_reward_override
        elif was_illegal_action: final_reward = self._cfg.illegal_action_penalty
        else: final_reward = shaped_reward

        # --- Prepare and Return Timestep ---
        obs = self._get_observation()
        action_mask = self._get_true_action_mask() if not self._cfg.ignore_legal_actions else np.ones(self.action_space_size, dtype=np.int8)
        meter = BiasMeter(obs[:self.market_and_portfolio_state_size], self.OBS_IDX)

        info = {
            'price': self.price_manager.current_price,
            'eval_episode_return': self.final_eval_reward,
            'illegal_actions_in_episode': self.illegal_action_count,
            'was_illegal_action': bool(was_illegal_action),
            'executed_action_name': self.last_action_info['final_action_name'],
            'directional_bias': meter.directional_bias,
            'volatility_bias': meter.volatility_bias,
            'portfolio_stats': self.portfolio_manager.get_raw_portfolio_stats(self.price_manager.current_price, self.iv_bin_index),
            'market_regime': self.price_manager.current_regime_name,
            'total_steps_in_episode': self.total_steps,
            'market_regime': self.current_iv_regime_name,
            'termination_reason': termination_reason
        }

        if terminated: info['episode_duration'] = self.current_step
        return BaseEnvTimestep({'observation': obs, 'action_mask': action_mask, 'to_play': -1}, final_reward, terminated, info)

    def _handle_action(self, action: int) -> Tuple[int, bool]:
        """
        The definitive, correct action handler. It correctly prioritizes all rules
        and ensures the action mask is always respected, with intelligent overrides
        for all game states.
        """
        true_action_mask = self._get_true_action_mask()
        is_illegal = true_action_mask[action] == 0

        # --- PRIORITY 1: Handle a legal action first ---
        if not is_illegal:
            return action, False

        # --- If we reach here, the agent's action was ILLEGAL. We must decide the override. ---
        self.illegal_action_count += 1

        # The curriculum takes highest priority during training.
        if self.is_training_mode and self.forced_opening_strategy_name:
            return self.actions_to_indices[self.forced_opening_strategy_name], True

        # Override Rule 1: Forced Strategy for Analysis (should not be illegal, but a good failsafe)
        forced_strategy = self.forced_opening_strategy_name
        if self.current_step == 0 and forced_strategy:
            return self.actions_to_indices.get(forced_strategy), True

        # Override Rule 2: Liquidation Period
        is_liquidation_period = self.current_day_index >= (self.episode_time_to_expiry - self._cfg.days_before_liquidation)
        if is_liquidation_period and not self.portfolio_manager.portfolio.empty:
            # Let us HOLD if POP > 90% and PF > 1.0
            # We need to get the current POP and Profit Factor
            summary = self.portfolio_manager.get_portfolio_summary(
                self.price_manager.current_price, self.iv_bin_index
            )
            if not (summary['prob_profit'] > 0.9 and summary.get('profit_factor', 0) > 1.0):
                return self.actions_to_indices['CLOSE_ALL'], True

        # Override Rule 3: Step 0 (Training or Evaluation)
        # If the agent attempts an illegal move on step 0, it MUST open a position.
        if self.current_step == 0:
            legal_open_indices = [idx for idx, is_legal in enumerate(true_action_mask) if is_legal]
            if not legal_open_indices: # Failsafe
                raise RuntimeError("No legal opening moves available on Step 0. Check action mask logic.")
            # We override the illegal action with a random valid opening.
            return random.choice(legal_open_indices), True

        # Override Rule 4: Default fallback for all other mid-episode illegal moves
        return self.actions_to_indices['HOLD'], True

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

        # 2. Get the complete portfolio risk profile in one go.
        stats = self.portfolio_manager.get_raw_portfolio_stats(self.price_manager.current_price, self.iv_bin_index)

        # Normalize the profit factor. We divide by 10 as a heuristic to keep the
        # value in a good range for tanh before it saturates.
        vec[self.OBS_IDX['PORTFOLIO_PROFIT_FACTOR_NORM']] = math.tanh(stats['profit_factor'] / 10.0)

        # Normalize by initial cash for a stable representation
        high_water_mark_norm = math.tanh(self.portfolio_manager.mtm_pnl_high / self._cfg.initial_cash)
        max_drawdown_norm = math.tanh(self.portfolio_manager.mtm_pnl_low / self._cfg.initial_cash)
        vec[self.OBS_IDX['MTM_PNL_HIGH_NORM']] = high_water_mark_norm
        vec[self.OBS_IDX['MTM_PNL_LOW_NORM']] = max_drawdown_norm
        

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
        self.portfolio_manager.update_mtm_water_marks(equity_after)
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
        is_end_of_day = (self.current_step) % self._cfg.steps_per_day == 0
        if not is_end_of_day:
            return time_decay # Return only the intra-day decay

        # --- End of Day Logic ---
        # Add the standard overnight decay for every day
        time_decay += self.decay_overnight

        # Check if the day that JUST ENDED was a Friday.
        # We use (self.current_day_index - 1) to get the index of the completed day.
        day_of_week_that_ended = (self.current_day_index - 1) % self.TRADING_DAYS_IN_WEEK
        is_friday = (day_of_week_that_ended == 4)

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
        agent_max_open_offset = self._cfg.get('agent_max_open_offset', 10)

        # Use the configurable max_strike_offset instead of a hardcoded range.
        for offset in range(-agent_max_open_offset, agent_max_open_offset + 1):
            sign = '+' if offset >= 0 else ''
            # We can simplify the action name for offsets > 9, e.g., ATM+10 instead of ATM+ 10
            offset_str = f"{sign}{offset}"
            for t in ['CALL', 'PUT']:
                for d in ['LONG', 'SHORT']:
                    actions[f'OPEN_{d}_{t}_ATM{offset_str}'] = i; i+=1

        actions['OPEN_BULL_CALL_SPREAD'] = i; i+=1
        actions['OPEN_BEAR_CALL_SPREAD'] = i; i+=1
        actions['OPEN_BULL_PUT_SPREAD'] = i; i+=1
        actions['OPEN_BEAR_PUT_SPREAD'] = i; i+=1

        # --- NEW: Strategy Morphing Actions ---
        actions['CONVERT_TO_IRON_CONDOR'] = i; i+=1
        actions['CONVERT_TO_IRON_FLY'] = i; i+=1
        actions['CONVERT_TO_STRANGLE'] = i; i+=1
        actions['CONVERT_TO_STRADDLE'] = i; i+=1
        actions['CONVERT_TO_BULL_CALL_SPREAD'] = i; i+=1
        actions['CONVERT_TO_BEAR_CALL_SPREAD'] = i; i+=1
        actions['CONVERT_TO_BULL_PUT_SPREAD'] = i; i+=1
        actions['CONVERT_TO_BEAR_PUT_SPREAD'] = i; i+=1

        actions['ADJUST_TO_DELTA_NEUTRAL'] = i; i+=1

        for d in ['LONG', 'SHORT']:
            actions[f'OPEN_{d}_STRADDLE_ATM'] = i; i+=1

        for w in [1, 2]:
            for d in ['LONG', 'SHORT']:
                actions[f'OPEN_{d}_STRANGLE_ATM_{w}'] = i; i+=1

        # Generate a range of delta-based strangle actions.
        for delta in range(15, 31, 5): # This creates [15, 20, 25, 30]
            actions[f'OPEN_LONG_STRANGLE_DELTA_{delta}'] = i; i+=1
            actions[f'OPEN_SHORT_STRANGLE_DELTA_{delta}'] = i; i+=1

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
            actions[f'HEDGE_NAKED_POS_{j}'] = i; i+=1

        actions['CLOSE_ALL'] = i
        return actions
        
    def _get_true_action_mask(self) -> np.ndarray:
        """
        Computes the correct action mask, applying all trading rules in priority order.
        """
        action_mask = np.zeros(self.action_space_size, dtype=np.int8)

        # Rule 0: Forced strategy (Step 0 only)
        if self._apply_forced_strategy(action_mask):
            return action_mask

        # Rule 1: Liquidation period
        if self._apply_liquidation_period_rules(action_mask):
            return action_mask

        # Rule 2: Non-empty portfolio
        if not self.portfolio_manager.portfolio.empty:
            return self._get_non_empty_portfolio_mask()

        # Rule 3: Empty portfolio
        return self._get_empty_portfolio_mask()


    # ----------------- HELPER METHODS -----------------

    def _apply_forced_strategy(self, action_mask: np.ndarray) -> bool:
        """Handles Rule 0: Forced strategy for analysis."""
        forced_strategy = self.forced_opening_strategy_name
        if self.current_step == 0 and forced_strategy:
            action_index = self.actions_to_indices.get(forced_strategy)
            assert action_index is not None, f"Forced strategy '{forced_strategy}' is invalid."
            action_mask[action_index] = 1
            return True
        return False

    def _apply_liquidation_period_rules(self, action_mask: np.ndarray) -> bool:
        """Handles Rule 1: Liquidation period logic."""

        is_liquidation_period = self.current_day_index >= (self.episode_time_to_expiry - self._cfg.days_before_liquidation)

        # Check if the user's condition is met
        if is_liquidation_period:
            action_mask[self.actions_to_indices['HOLD']] = 1
            if not self.portfolio_manager.portfolio.empty:
                action_mask[self.actions_to_indices['CLOSE_ALL']] = 1
            return True # Handled the state

        return False # Did not handle the state

    def _get_non_empty_portfolio_mask(self) -> np.ndarray:
        """Handles Rule 2: Logic for when the portfolio is not empty."""
        action_mask = np.zeros(self.action_space_size, dtype=np.int8)
        portfolio_df = self.portfolio_manager.portfolio
        
        # --- Basic actions (unchanged) ---
        action_mask[self.actions_to_indices['HOLD']] = 1
        action_mask[self.actions_to_indices['CLOSE_ALL']] = 1

        # --- Management Actions (unchanged) ---
        atm_price = self.market_rules_manager.get_atm_price(self.price_manager.current_price)
        for i, original_pos in portfolio_df.iterrows():
            self._set_if_exists(action_mask, f'CLOSE_POSITION_{i}')
            if not original_pos['is_hedged']:
                self._set_if_exists(action_mask, f'HEDGE_NAKED_POS_{i}')
            self._set_shift_if_no_conflict(action_mask, i, original_pos, direction="UP")
            self._set_shift_if_no_conflict(action_mask, i, original_pos, direction="DOWN")
            if original_pos['strike_price'] != atm_price:
                self._set_shift_to_atm_if_no_conflict(action_mask, i, original_pos, atm_price)

        if not portfolio_df.empty:
            current_strategy_id = portfolio_df.iloc[0]['strategy_id']
            
            # Use self.strategy_name_to_id for all lookups to ensure consistency.
            s_map = self.strategy_name_to_id 

            strangle_ids = {
                s_map.get('SHORT_STRANGLE_DELTA_15'), s_map.get('SHORT_STRANGLE_DELTA_20'),
                s_map.get('SHORT_STRANGLE_DELTA_25'), s_map.get('SHORT_STRANGLE_DELTA_30'),
            }
            straddle_id = s_map.get('SHORT_STRADDLE')
            condor_id = s_map.get('SHORT_IRON_CONDOR')
            fly_id = s_map.get('SHORT_IRON_FLY')
            call_fly_ids = {s_map.get('LONG_CALL_FLY_1'), s_map.get('SHORT_CALL_FLY_1'), s_map.get('LONG_CALL_FLY_2'), s_map.get('SHORT_CALL_FLY_2')}
            put_fly_ids = {s_map.get('LONG_PUT_FLY_1'), s_map.get('SHORT_PUT_FLY_1'), s_map.get('LONG_PUT_FLY_2'), s_map.get('SHORT_PUT_FLY_2')}

            # Rule: Strangle -> Iron Condor
            if current_strategy_id in strangle_ids:
                if len(portfolio_df) <= self.portfolio_manager.max_positions - 2:
                    self._set_if_exists(action_mask, 'CONVERT_TO_IRON_CONDOR')

            # Rule: Straddle -> Iron Fly
            if current_strategy_id == straddle_id:
                if len(portfolio_df) <= self.portfolio_manager.max_positions - 2:
                    self._set_if_exists(action_mask, 'CONVERT_TO_IRON_FLY')

            # Rule: Iron Condor -> Strangle
            if current_strategy_id == condor_id:
                self._set_if_exists(action_mask, 'CONVERT_TO_STRANGLE')

            # Rule: Iron Fly -> Straddle
            if current_strategy_id == fly_id:
                self._set_if_exists(action_mask, 'CONVERT_TO_STRADDLE')

            # --- NEW: Logic for converting Condors/Flies to Verticals ---
            if current_strategy_id == condor_id or current_strategy_id == fly_id:
                put_legs = portfolio_df[portfolio_df['type'] == 'put']
                call_legs = portfolio_df[portfolio_df['type'] == 'call']

                if not put_legs.empty and not call_legs.empty:
                    # Find the midpoint of each spread
                    put_spread_midpoint = put_legs['strike_price'].mean()
                    call_spread_midpoint = call_legs['strike_price'].mean()
                    
                    current_price = self.price_manager.current_price
                    
                    # If price is closer to the put side, the call side is the winner to close
                    if abs(current_price - put_spread_midpoint) < abs(current_price - call_spread_midpoint):
                        self._set_if_exists(action_mask, 'CONVERT_TO_BULL_PUT_SPREAD')
                    else: # Otherwise, the put side is the winner to close
                        self._set_if_exists(action_mask, 'CONVERT_TO_BEAR_CALL_SPREAD')

            # <<< NEW: Logic for converting Butterflies to Verticals >>>
            elif current_strategy_id in call_fly_ids:
                # A call butterfly can be decomposed into a Bull Call or a Bear Call spread.
                self._set_if_exists(action_mask, 'CONVERT_TO_BULL_CALL_SPREAD')
                self._set_if_exists(action_mask, 'CONVERT_TO_BEAR_CALL_SPREAD')

            elif current_strategy_id in put_fly_ids:
                # A put butterfly can be decomposed into a Bull Put or a Bear Put spread.
                self._set_if_exists(action_mask, 'CONVERT_TO_BULL_PUT_SPREAD')
                self._set_if_exists(action_mask, 'CONVERT_TO_BEAR_PUT_SPREAD')
        
        return action_mask

    def _set_shift_if_no_conflict(self, action_mask, i, original_pos, direction):
        """Set SHIFT_UP or SHIFT_DOWN if no strike conflict."""
        delta = self._cfg.strike_distance if direction == "UP" else -self._cfg.strike_distance
        new_strike = original_pos['strike_price'] + delta
        portfolio_df = self.portfolio_manager.portfolio.drop(i)

        is_conflict = any(
            (pos['strike_price'] == new_strike and pos['type'] == original_pos['type'])
            for _, pos in portfolio_df.iterrows()
        )

        action_name = f'SHIFT_{direction}_POS_{i}'
        if not is_conflict:
            self._set_if_exists(action_mask, action_name)

    def _set_shift_to_atm_if_no_conflict(self, action_mask, i, original_pos, atm_price):
        """Set SHIFT_TO_ATM if no strike conflict."""
        portfolio_df = self.portfolio_manager.portfolio.drop(i)
        is_conflict = any(
            (pos['strike_price'] == atm_price and pos['type'] == original_pos['type'])
            for _, pos in portfolio_df.iterrows()
        )
        if not is_conflict:
            self._set_if_exists(action_mask, f'SHIFT_TO_ATM_{i}')

    def _set_if_exists(self, action_mask, action_name):
        """Safely set action if it exists in actions_to_indices."""
        if action_name in self.actions_to_indices:
            action_mask[self.actions_to_indices[action_name]] = 1

    def _get_empty_portfolio_mask(self) -> np.ndarray:
        """Handles Rule 3: Opening strategies when portfolio is empty."""
        atm_price = self.market_rules_manager.get_atm_price(self.price_manager.current_price)
        days_to_expiry = (self.episode_time_to_expiry - self.current_day_index) * (
            self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK
        )

        base_opening_mask = self._compute_base_opening_mask(atm_price, days_to_expiry)
        final_opening_mask = self._apply_slot_constraints(base_opening_mask)

        # Apply opening curriculum at Step 0 (training only)
        if self.current_step == 0 and not self._cfg.disable_opening_curriculum:
            return self._apply_opening_curriculum(final_opening_mask)

        if self.current_step > 0:
            final_opening_mask[self.actions_to_indices['HOLD']] = 1

        return final_opening_mask

    def _compute_base_opening_mask(self, atm_price, days_to_expiry) -> np.ndarray:
        """Compute legality of each opening action based on greeks/delta rules."""
        base_mask = np.zeros(self.action_space_size, dtype=np.int8)
        for action_name, index in self.actions_to_indices.items():
            if not action_name.startswith('OPEN_'):
                continue
            if self._is_legal_opening_action(action_name, atm_price, days_to_expiry):
                base_mask[index] = 1
        return base_mask

    def _is_legal_opening_action(self, action_name, atm_price, days_to_expiry) -> bool:
        """Check if a given opening action is legal."""
        is_single_leg = 'ATM' in action_name and 'STRADDLE' not in action_name and 'STRANGLE' not in action_name
        if not is_single_leg:
            return True  # multi-leg assumed legal for now

        parts = action_name.split('_')
        direction, option_type, offset_str = parts[1], parts[2].lower(), parts[3].replace('ATM', '')
        offset = int(offset_str)

        if direction == 'SHORT' and abs(offset) > self._cfg.short_leg_max_offset:
            return False

        strike_price = atm_price + (offset * self._cfg.strike_distance)
        vol = self.market_rules_manager.get_implied_volatility(offset, option_type, self.iv_bin_index)
        greeks = self.bs_manager.get_all_greeks_and_price(
            self.price_manager.current_price, strike_price, days_to_expiry, vol, option_type == 'call'
        )
        abs_delta = abs(greeks['delta'])

        if direction == 'LONG':
            return self._cfg.otm_long_delta_threshold <= abs_delta <= self._cfg.itm_long_delta_threshold
        elif direction == 'SHORT':
            return abs_delta <= self._cfg.itm_short_delta_threshold
        return True

    def _apply_slot_constraints(self, base_mask: np.ndarray) -> np.ndarray:
        """Apply max position slot constraints to legal opening actions."""
        final_mask = np.zeros_like(base_mask)
        available_slots = self.portfolio_manager.max_positions

        for index, is_legal in enumerate(base_mask):
            if not is_legal:
                continue
            name = self.indices_to_actions[index]
            if 'CONDOR' in name or 'FLY' in name:
                if available_slots >= 4:
                    final_mask[index] = 1
            elif 'STRADDLE' in name or 'STRANGLE' in name or 'SPREAD' in name:
                if available_slots >= 2:
                    final_mask[index] = 1
            elif 'ATM' in name:
                if available_slots >= 1:
                    final_mask[index] = 1
        return final_mask

    def _apply_opening_curriculum(self, final_mask: np.ndarray) -> np.ndarray:
        """Randomly choose a strategy family for Step 0 training."""
        strategy_families = {
            "SINGLE_LEG": lambda name: 'ATM' in name and 'STRADDLE' not in name and 'STRANGLE' not in name,
            "STRADDLE": lambda name: 'STRADDLE' in name,
            "STRANGLE": lambda name: 'STRANGLE' in name,
            "SPREAD": lambda name: 'SPREAD' in name,
            "IRON_FLY_CONDOR": lambda name: 'IRON' in name,
            "BUTTERFLY": lambda name: 'FLY' in name and 'IRON' not in name,
        }

        chosen_family_name = random.choice(list(strategy_families.keys()))
        is_in_family = strategy_families[chosen_family_name]
        curriculum_mask = np.zeros_like(final_mask)

        for index, is_legal in enumerate(final_mask):
            if is_legal and is_in_family(self.indices_to_actions[index]):
                curriculum_mask[index] = 1

        return curriculum_mask if np.any(curriculum_mask) else final_mask

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
