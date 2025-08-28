# zoo/options_zero_game/envs/options_zero_game_env.py
# <<< FINAL VERSION, incorporating all fixes >>>

import copy
import math
import random
from typing import Tuple, Dict, Any, List
import time

import gymnasium as gym
import numpy as np
import pandas as pd
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
        price_source='historical',
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
        max_strike_offset=50,
        bid_ask_spread_pct=0.002,
        brokerage_per_leg=25.0,
        days_before_liquidation=1,
        
        # Black-Scholes Manager Config
        risk_free_rate=0.10,

        # Reward and Penalty Config
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

        disable_spread_solver=False,
        
        # Agent/Framework Config
        ignore_legal_actions=True,

        # These rules apply ONLY when opening a new LONG single-leg position.
        otm_long_delta_threshold=0.45, # Disallow buying options with delta < 45
        itm_long_delta_threshold=0.55, # Disallow buying options with delta > 55
        itm_short_delta_threshold=0.55, # Disallow shorting options with delta > 55

        # <<< --- NEW: The Liquidity Window Parameter --- >>>
        # The max number of strikes away from the ATM that the portfolio hedge solver
        # is allowed to search. Prevents rolling to illiquid, far OTM options.
        hedge_roll_search_width=20, # ATM +/- 20 strikes seems like a reasonable default

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

        self.is_custom_iv_episode = False

        # Add a flag to track if this specific instance has been reset yet.
        self._has_printed_setup_info = False

        # <<< --- NEW: Add a variable to hold the portfolio definition --- >>>
        self.portfolio_to_setup_on_step_0 = None

        # portfolio_manager, which is not created yet.
        # We will move this logic to the reset method.
        self.pnl_scaling_factor = self._cfg.get('pnl_scaling_factor', 1000) # Use a default for now.
       
        # The environment now knows its own role.
        self.is_eval_mode = self._cfg.get('is_eval_mode', False)

        self.np_random, _ = seeding.np_random(None)

        self.disable_spread_solver = self._cfg.get('disable_spread_solver', False)

        # This dictionary is passed in the config and is needed by both the
        # environment (for the action mask) and the portfolio manager.
        self.strategy_name_to_id = cfg.get('strategy_name_to_id', {})

        # <<< NEW: Load the parameter from the config >>>
        self.capital_preservation_bonus_pct = cfg.get('capital_preservation_bonus_pct', 0.0)

        self.regimes = self._cfg.get('unified_regimes', [])
        self.current_iv_regime_name = "N/A" # For logging
        self.current_iv_regime_index = 0 # Start with a default

        # 1. Define the sizes of all embedding blocks FIRST.
        self.vol_embedding_size = 128
        self.mlp_embedding_size = 64
        
        # 2. THEN, define the layout of the base summary block.
        self.OBS_IDX = {
            'PRICE_NORM': 0, 'TIME_NORM': 1, 'PNL_NORM': 2,
            'LOG_RETURN': 3,
            'EXPECTED_MOVE_NORM': 4,
            'PORTFOLIO_DELTA': 5,
            'PORTFOLIO_GAMMA': 6,
            'PORTFOLIO_THETA': 7,
            'PORTFOLIO_VEGA': 8,
            'PORTFOLIO_MAX_PROFIT_NORM': 9,
            'PORTFOLIO_MAX_LOSS_NORM': 10,
            'PORTFOLIO_RR_RATIO_NORM': 11,
            'PORTFOLIO_PROB_PROFIT': 12,
            'PORTFOLIO_PROFIT_FACTOR_NORM': 13,
            'MTM_PNL_HIGH_NORM': 14,
            'MTM_PNL_LOW_NORM': 15,
            'DIR_EXPERT_PROB_DOWN': 16,
            'DIR_EXPERT_PROB_NEUTRAL': 17,
            'DIR_EXPERT_PROB_UP': 18,
        }
       
        # 3. NOW, we can safely calculate all start indices and total size.
        base_summary_size = len(self.OBS_IDX)
        
        self.OBS_IDX['VOL_EMBEDDING_START'] = base_summary_size
        self.OBS_IDX['TREND_EMBEDDING_START'] = self.OBS_IDX['VOL_EMBEDDING_START'] + self.vol_embedding_size
        self.OBS_IDX['OSCILLATOR_EMBEDDING_START'] = self.OBS_IDX['TREND_EMBEDDING_START'] + self.mlp_embedding_size
        self.OBS_IDX['DEVIATION_EMBEDDING_START'] = self.OBS_IDX['OSCILLATOR_EMBEDDING_START'] + self.mlp_embedding_size
        self.OBS_IDX['CYCLE_EMBEDDING_START'] = self.OBS_IDX['DEVIATION_EMBEDDING_START'] + self.mlp_embedding_size
        self.OBS_IDX['PATTERN_EMBEDDING_START'] = self.OBS_IDX['CYCLE_EMBEDDING_START'] + self.mlp_embedding_size
        
        self.summary_block_size = self.OBS_IDX['PATTERN_EMBEDDING_START'] + self.mlp_embedding_size
        
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
        self._action_space = spaces.Discrete(self.action_space_size)

        self.POS_IDX = {
            'IS_OCCUPIED': 0, 'TYPE_NORM': 1, 'DIRECTION_NORM': 2, 'STRIKE_DIST_NORM': 3,
            'DAYS_HELD_NORM': 4, 'PROB_OF_PROFIT': 5, 'MAX_PROFIT_NORM': 6, 'MAX_LOSS_NORM': 7,
            'DELTA': 8, 'GAMMA': 9, 'THETA': 10, 'VEGA': 11, 'IS_HEDGED': 12,
        }
        self.PORTFOLIO_STATE_SIZE_PER_POS = len(self.POS_IDX)
        self.positions_block_size = self._cfg.max_positions * self.PORTFOLIO_STATE_SIZE_PER_POS

        # The portfolio block starts after the summary block
        self.PORTFOLIO_START_IDX = self.summary_block_size

        # The model sees the complete state: the summary block + the positions block.
        self.model_observation_size = self.summary_block_size + self.positions_block_size
       
        # The full vector passed in the timestep includes the model's observation + the action mask.
        self.framework_vec_size = self.model_observation_size + self.action_space_size
        
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.framework_vec_size,), dtype=np.float32)
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

        # 0 = Normal, 1 = Symmetric Mirror, 2 = Strategy-Type Mirror
        self.mirror_mode = 0
        # <<< --- NEW: Add the state variable to the environment --- >>>
        self.fixed_profit_target_pnl: float = 0.0
        self.initial_net_premium: float = 0.0

        # Pre-calculate the daily opportunity cost penalty once at initialization.
        daily_risk_free_rate = self._cfg.risk_free_rate / 365.25
        self.daily_opportunity_cost_penalty = self._cfg.initial_cash * daily_risk_free_rate

    def seed(self, seed: int, dynamic_seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)

        # We must also seed the global NumPy random number generator, as it is used
        # by the IV regime selection logic (np.random.choice).
        np.random.seed(seed)

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

        # The scaling factor is now proportional to a "standard" unit of P&L:
        # the max profit from a 1-strike-wide vertical spread.
        self.pnl_scaling_factor = self._cfg.strike_distance * self._cfg.lot_size
        # Add a failsafe for weird configs
        if self.pnl_scaling_factor <= 0:
            self.pnl_scaling_factor = 1000 # Fallback to the old default
 
        # --- 2. Determine Episode Length & Market Conditions (Must happen before portfolio setup) ---
        forced_length = self._cfg.get('forced_episode_length', 0)
        if forced_length > 0:
            self.episode_time_to_expiry = forced_length
        else:
            # Use the seeded NumPy random generator for a deterministic episode length.
            # np.integers is exclusive of the high value, so we add 1.
            self.episode_time_to_expiry = self.np_random.integers(
                self._cfg.min_time_to_expiry_days, 
                self._cfg.time_to_expiry_days + 1
            )

        self.total_steps = self.episode_time_to_expiry * self._cfg.steps_per_day

        # The logic for setting up the initial state and market is now fully integrated.
        
        forced_atm_iv = self._cfg.get('forced_atm_iv')
        chosen_regime_for_episode = None
        self.is_custom_iv_episode = False # Reset the flag for each new episode

        # PRIORITY 1: A forced IV setup from the evaluator file. This overrides everything.
        if forced_atm_iv:
            self.is_custom_iv_episode = True # Set the flag for this episode
            if not self._has_printed_setup_info:
                 print(f"(INFO) Using forced ATM IV of {forced_atm_iv}% for this episode.")
            
            chosen_regime_for_episode = {
                'name': f"Custom IV @ {forced_atm_iv}",
                'mu': 0.00001, 'omega': 0.000002, 'alpha': 0.05, 'beta': 0.92, 'overnight_vol_multiplier': 1.1,
                'atm_iv': forced_atm_iv,
                'far_otm_put_iv': forced_atm_iv * 1.5,
                'far_otm_call_iv': forced_atm_iv * 0.9,
            }
        
        # PRIORITY 2: If no forced IV, check if we are in historical mode.
        elif self._cfg.price_source == 'historical':
            # Use the 'Normal' regime as a consistent, sensible baseline for all historical runs.
            # This prevents random, high-volatility IVs from polluting historical tests.
            chosen_regime_for_episode = next((reg for reg in self.regimes if reg['name'] == 'Normal (Medium Vol, Bullish)'), self.regimes[0])

        # PRIORITY 3: If not forced and not historical, it must be a GARCH run. Select randomly.
        else:
            if self.iv_stationary_dist is not None:
                self.current_iv_regime_index = self.np_random.choice(len(self.regimes), p=self.iv_stationary_dist)
            else:
                self.current_iv_regime_index = self.np_random.integers(0, len(self.regimes))
            chosen_regime_for_episode = self.regimes[self.current_iv_regime_index]

        # After determining the regime, we MUST synchronize the environment's state index.
        # We find the index of the chosen regime in our master list.
        # This handles all cases: forced, historical, and random GARCH.
        try:
            # Find the index by matching the unique 'name' of the regime.
            self.current_iv_regime_index = [i for i, r in enumerate(self.regimes) if r['name'] == chosen_regime_for_episode['name']][0]
        except IndexError:
            # This is a failsafe for the custom_regime case, where the name won't be in the list.
            # In this case, the index doesn't matter as transitions are disabled anyway.
            if not self.is_custom_iv_episode:
                print(f"WARNING: Could not find chosen regime '{chosen_regime_for_episode['name']}' in master list. Defaulting to index 0.")
                self.current_iv_regime_index = 0
        
        # --- Now that state is consistent, update all managers ---
        self._update_market_rules_for_regime(override_regime=chosen_regime_for_episode)
        self.price_manager.reset(self.total_steps, chosen_regime=chosen_regime_for_episode)
        
        self.iv_bin_index = self.np_random.integers(0, len(self.market_rules_manager.iv_bins['call']['0']))
      
        # --- 3. THE DEFINITIVE FIX: Prioritized Initial State Setup ---
        
        # First, ensure the PortfolioManager is initialized, as it's needed for all paths.
        self.portfolio_manager = PortfolioManager(
            cfg=self._cfg, bs_manager=self.bs_manager, market_rules_manager=self.market_rules_manager,
            iv_calculator_func=self._get_dynamic_iv, strategy_name_to_id=self.strategy_name_to_id
        )
        
        self.fixed_profit_target_pnl = 0.0
        self.initial_net_premium = 0.0
        self.portfolio_to_setup_on_step_0 = self._cfg.get('forced_initial_portfolio')

        # PRIORITY 1: A forced portfolio setup from the evaluator via a JSON file.
        if self.portfolio_to_setup_on_step_0:
            if not self._has_printed_setup_info:
                print("(INFO) Setting up forced initial portfolio in environment...")
                self._has_printed_setup_info = True # Set the flag so it doesn't print again.

            self.forced_opening_strategy_name = None

        # PRIORITY 2: A forced opening strategy from the curriculum or evaluator.
        elif self.is_training_mode and self.training_curriculum:
            self._episode_count += 1
            approx_current_step = self._episode_count * self.total_steps
            active_strategy = 'ALL'
            sorted_phases = sorted(self.training_curriculum.keys())
            for start_step in sorted_phases:
                if approx_current_step >= start_step: active_strategy = self.training_curriculum[start_step]
                else: break
            self.forced_opening_strategy_name = active_strategy if active_strategy != 'ALL' else None
        else:
            self.forced_opening_strategy_name = self._cfg.get('forced_opening_strategy_name')
        
        # --- 4. Final State Reset ---
        self.high_water_mark = self._cfg.initial_cash
        self.realized_vol_series = np.zeros(self.total_steps + 1, dtype=np.float32)

        obs = self._get_observation()
        action_mask = self._get_true_action_mask() if not self._cfg.ignore_legal_actions else np.ones(self.action_space_size, dtype=np.int8)

        return {'observation': obs, 'action_mask': action_mask, 'to_play': -1}

    def _handle_day_change(self):
        """Called when a day passes. Evolves the IV regime using the Markov chain."""
        # If this is a custom run, do NOT change the regime.
        if self.is_custom_iv_episode:
            return

        if self.iv_transition_matrix is not None:
            # Get the probability distribution for the next state
            transition_probs = self.iv_transition_matrix[self.current_iv_regime_index]
            # Choose the next regime based on these probabilities
            self.current_iv_regime_index = self.np_random.choice(len(self.regimes), p=transition_probs)
            
            # Now that the regime has changed, we must rebuild the market rules
            self._update_market_rules_for_regime()
            # We also must re-initialize the PortfolioManager's IV calculator
            self.portfolio_manager.iv_calculator = self._get_dynamic_iv

    # <<< NEW: Create a helper to generate the skew table on the fly >>>
    def _update_market_rules_for_regime(self, override_regime: dict = None):
        """
        Generates and applies the skew table for the current IV regime.
        MODIFIED: Can now accept an 'override_regime' for deterministic evaluations.
        """
        if override_regime:
            chosen_regime = override_regime
        else:
            chosen_regime = self.regimes[self.current_iv_regime_index]
        
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

        # Keep track of portfolio state before the action
        portfolio_was_empty = self.portfolio_manager.portfolio.empty

        # --- MODIFIED (Corrected): The new 3-way augmentation logic on Step 0 ---
        action_name = self.indices_to_actions[action]
        original_action_name = action_name # Keep a copy for the log

        if self.current_step == 0 and self.mirror_mode != 0:
            mirrored_action_name = None
            
            if self.mirror_mode == 1: # Symmetric Mirror
                mirrored_action_name = self.portfolio_manager.SYMMETRIC_ACTION_MAP.get(action_name)
            
            elif self.mirror_mode == 2: # Strategy-Type Mirror
                mirrored_action_name = self.portfolio_manager.STRATEGY_TYPE_MAP.get(action_name)
            
            if mirrored_action_name:
                mirrored_action_index = self.actions_to_indices.get(mirrored_action_name)
                
                # First, get the true, up-to-the-moment action mask.
                true_action_mask = self._get_true_action_mask()

                # Only use the mirror if the resulting action is both valid AND LEGAL.
                if mirrored_action_index is not None and true_action_mask[mirrored_action_index] == 1:
                    action = mirrored_action_index # Override the agent's choice
                # If the mirror is illegal, we do nothing and proceed with the agent's original action.

        # 2. Execute the agent's action (or an override) on the current state.
        self._take_action_on_state(action)
        
        # 3. Check if a new portfolio was just created.
        # This is true if the portfolio WAS empty and is NOW NOT empty.
        if portfolio_was_empty and not self.portfolio_manager.portfolio.empty:
            # This block will now run for FORCED openings, SETUP openings, and normal agent OPEN_* actions.
            self._update_initial_premium_and_targets()
        
        # 4. Advance the market and get the final outcome.
        return self._advance_market_and_get_outcome(equity_before, original_action_name)

    def _get_dynamic_iv(self, offset: int, option_type: str) -> float:
        """
        Calculates the dynamic IV.
        MODIFIED: This version now uses the embedding from the new, full
        Council of Experts pipeline for a more robust prediction.
        """
        # 1. Get the floor IV from the static skew table. This remains the same.
        iv_from_table = self.market_rules_manager.get_implied_volatility(
            offset, option_type, self.iv_bin_index
        )
        
        # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
        # 2. Get the expert's prediction from the new architecture.
        vol_embedding = self.price_manager.volatility_embedding
        
        predicted_vol = 0.20 # A safe, neutral default

        # If the embedding has been calculated, derive a prediction from it.
        if vol_embedding is not None:
            # The magnitude (L2 norm) of the embedding is a good proxy for volatility intensity.
            magnitude = np.linalg.norm(vol_embedding)
            
            # Scale the magnitude to a realistic IV range (e.g., 5% to 150%).
            # We use tanh to squash the magnitude into a predictable (0, 1) range first.
            # Then we scale it to our desired IV range.
            scaled_magnitude = math.tanh(magnitude)
            predicted_vol = 0.05 + scaled_magnitude * 1.45 # Maps [0, 1] -> [0.05, 1.50]

        # 3. Add the market's risk premium.
        base_iv = predicted_vol + self.volatility_premium_abs
        
        # 4. Return the greater of the dynamic base and the static floor.
        return max(base_iv, iv_from_table)

    def _update_initial_premium_and_targets(self):
        """
        A centralized helper to be called WHENEVER a new portfolio is created.
        It correctly sets the initial premium and calculates profit/loss targets.
        """
        if self.portfolio_manager.portfolio.empty:
            self.initial_net_premium = 0.0
            self.fixed_profit_target_pnl = 0.0
            return

        # 1. Get the authoritative initial premium from the manager.
        self.initial_net_premium = self.portfolio_manager.initial_net_premium
        
        # 2. Calculate profit/loss targets based on this premium.
        self.fixed_profit_target_pnl = 0.0
        stats = self.portfolio_manager.get_raw_portfolio_stats(self.price_manager.current_price, self.iv_bin_index)
        if self.initial_net_premium < 0: # Credit
            credit_tp_pct = self._cfg.get('credit_strategy_take_profit_pct', 0)
            if credit_tp_pct > 0: self.fixed_profit_target_pnl = stats.get('max_profit', 0.0) * (credit_tp_pct / 100)
        elif self.initial_net_premium > 0: # Debit
            debit_tp_mult = self._cfg.get('debit_strategy_take_profit_multiple', 0)
            if debit_tp_mult > 0: self.fixed_profit_target_pnl = self.initial_net_premium * debit_tp_mult

    # <<< --- NEW: A dedicated helper method for setting up the portfolio --- >>>
    def _setup_initial_portfolio(self, portfolio_def: list):
        """
        Prices and executes the creation of a portfolio from a definition list.
        This is called only on Step 0 when a setup file is provided.
        """
        days_to_expiry_float = self.episode_time_to_expiry * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)
        for leg_def in portfolio_def:
            leg_def['days_to_expiry'] = days_to_expiry_float
            leg_def['entry_step'] = 0

        priced_legs = self.portfolio_manager._price_legs(portfolio_def, self.price_manager.start_price, self.iv_bin_index)
        
        if priced_legs:
            num_legs = len(priced_legs)
            strategy_key = f'CUSTOM_{num_legs}_LEGS' if num_legs > 1 else 'CUSTOM_HEDGED'
            strategy_id = self.strategy_name_to_id.get(strategy_key, -1)

            pnl_profile = self.portfolio_manager._calculate_universal_risk_profile(priced_legs, 0.0)
            pnl_profile['strategy_id'] = strategy_id
            
            self.portfolio_manager._execute_trades(priced_legs, pnl_profile)
            # Call the new centralized helper.
            self._update_initial_premium_and_targets()

    def _take_action_on_state(self, action: int):
        """
        PART 1 of the step process. It takes the agent's action, enforces all rules,
        updates the portfolio,
        It does NOT advance time or the market price.
        """

        # Intercept the call at Step 0 if a portfolio setup is pending.
        if self.current_step == 0 and self.portfolio_to_setup_on_step_0:
            self._setup_initial_portfolio(self.portfolio_to_setup_on_step_0)
            
            # Set the action info for a clean log file, then exit this method.
            self.last_action_info = {
                'final_action': -1, # Use a dummy action index
                'final_action_name': 'SETUP_PORTFOLIO_FROM_FILE',
                'total_steps_in_episode': self.total_steps,
                'was_illegal_action': False
            }
            self.portfolio_manager.sort_portfolio()
            return # Skip all other action handling

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
            # Call the new centralized helper AFTER any opening action.
            self._update_initial_premium_and_targets()
        elif final_action_name.startswith('CLOSE_POSITION_'):
            self.portfolio_manager.close_position(int(final_action_name.split('_')[-1]), self.price_manager.current_price, self.iv_bin_index, self.current_step)
        elif final_action_name == 'CLOSE_ALL':
            self.portfolio_manager.close_all_positions(self.price_manager.current_price, self.iv_bin_index, self.current_step)
            # After closing all, we must reset the profit target so a new one can be set.
            self.fixed_profit_target_pnl = 0.0
            self.initial_net_premium = 0.0
        elif final_action_name.startswith('SHIFT_'):
            if 'ATM' in final_action_name: self.portfolio_manager.shift_to_atm(final_action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step)
            else: self.portfolio_manager.shift_position(final_action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step)
        elif final_action_name.startswith('HEDGE_NAKED_POS_'):
            self.portfolio_manager.add_hedge(int(final_action_name.split('_')[-1]),
                self.price_manager.current_price, self.iv_bin_index, self.current_step
            )
        elif final_action_name.startswith('CONVERT_TO_'):
            self._route_convert_action(final_action_name)
        elif final_action_name == 'RECENTER_VOLATILITY_POSITION':
            self.portfolio_manager.recenter_volatility_position(self.price_manager.current_price, self.iv_bin_index, self.current_step)
        # <<< --- NEW: Routing for Advanced Delta Management --- >>>
        elif final_action_name == 'HEDGE_DELTA_WITH_ATM_OPTION':
            current_day = self.current_step // self._cfg.steps_per_day
            days_to_expiry_float = (self.episode_time_to_expiry - current_day) * (self.TOTAL_DAYS_IN_WEEK / self.TRADING_DAYS_IN_WEEK)
            self.portfolio_manager.hedge_delta_with_atm_option(self.price_manager.current_price, self.iv_bin_index, self.current_step, days_to_expiry_float)
        
        elif 'DELTA_BY_SHIFTING_LEG' in final_action_name:
            parts = final_action_name.split('_')
            change_direction = 'increase' if parts[0] == 'INCREASE' else 'decrease'
            leg_index = int(parts[-1])
            
            if not (0 <= leg_index < len(self.portfolio_manager.get_portfolio())): return

            leg = self.portfolio_manager.get_portfolio().iloc[leg_index]
            leg_type = leg['type']
            leg_dir = leg['direction']
            shift_dir = ""

            # a) Resolve the meta-action into a concrete SHIFT_UP or SHIFT_DOWN
            if change_direction == 'increase':
                if leg_type == 'call' and leg_dir == 'long': shift_dir = 'DOWN'
                elif leg_type == 'call' and leg_dir == 'short': shift_dir = 'UP'
                elif leg_type == 'put' and leg_dir == 'long': shift_dir = 'UP'
                elif leg_type == 'put' and leg_dir == 'short': shift_dir = 'UP' # <<< --- CORRECTED LINE
            else: # 'decrease'
                if leg_type == 'call' and leg_dir == 'long': shift_dir = 'UP'
                elif leg_type == 'call' and leg_dir == 'short': shift_dir = 'DOWN'
                elif leg_type == 'put' and leg_dir == 'long': shift_dir = 'DOWN'
                elif leg_type == 'put' and leg_dir == 'short': shift_dir = 'DOWN' # <<< --- CORRECTED LINE
            
            if shift_dir:
                resolved_action_name = f"SHIFT_{shift_dir}_POS_{leg_index}"
                
                # b) Re-validate the resolved action against the true action mask.
                resolved_action_index = self.actions_to_indices.get(resolved_action_name)
                true_action_mask = self._get_true_action_mask()

                if resolved_action_index is not None and true_action_mask[resolved_action_index] == 1:
                    # c) Only execute if the resolved action is valid AND legal.
                    self.portfolio_manager.shift_position(resolved_action_name, self.price_manager.current_price, self.iv_bin_index, self.current_step)
                # d) If the resolved action is illegal (due to a conflict), do nothing.
                #    This effectively turns the agent's choice into a HOLD.

        elif final_action_name.startswith('HEDGE_PORTFOLIO_BY_ROLLING_LEG_'):
            leg_index = int(final_action_name.split('_')[-1])
            
            # a) Resolve the meta-action: Find the optimal new strike first.
            new_strike = self.portfolio_manager._find_portfolio_neutral_strike(
                leg_index, self.price_manager.current_price, self.iv_bin_index
            )

            if new_strike is not None:
                # b) Determine what the equivalent SHIFT action would be.
                # This requires checking the original strike and the new strike.
                original_strike = self.portfolio_manager.get_portfolio().iloc[leg_index]['strike_price']
                shift_dir = "UP" if new_strike > original_strike else "DOWN"
                
                resolved_action_name = f"SHIFT_{shift_dir}_POS_{leg_index}"
                resolved_action_index = self.actions_to_indices.get(resolved_action_name)
                true_action_mask = self._get_true_action_mask()
                
                # c) Re-validate: Check if this equivalent SHIFT would have been legal.
                # The _get_true_action_mask already contains the conflict check logic for shifts.
                if resolved_action_index is not None and true_action_mask[resolved_action_index] == 1:
                    # d) Only execute the roll if the equivalent shift is legal.
                    self.portfolio_manager.hedge_portfolio_by_rolling_leg(leg_index, self.price_manager.current_price, self.iv_bin_index, self.current_step)
                # e) If the roll would cause a conflict, do nothing (effectively a HOLD).

        # 4. Sort the portfolio and take the crucial snapshot.
        self.portfolio_manager.sort_portfolio()

    def _route_convert_action(self, action_name):
        """Helper to route all CONVERT_TO_* actions."""
        # (This is just a cleaner way to organize the many elif statements)
        pm = self.portfolio_manager
        price, iv_idx, step = self.price_manager.current_price, self.iv_bin_index, self.current_step
        if action_name == 'CONVERT_TO_IRON_CONDOR': pm.convert_to_iron_condor(price, iv_idx, step)
        elif action_name == 'CONVERT_TO_IRON_FLY': pm.convert_to_iron_fly(price, iv_idx, step)
        elif action_name == 'CONVERT_TO_STRANGLE': pm.convert_to_strangle(price, iv_idx, step)
        elif action_name == 'CONVERT_TO_STRADDLE': pm.convert_to_straddle(price, iv_idx, step)
        elif action_name == 'CONVERT_TO_BULL_CALL_SPREAD': pm.convert_to_bull_call_spread(price, iv_idx, step)
        elif action_name == 'CONVERT_TO_BULL_PUT_SPREAD': pm.convert_to_bull_put_spread(price, iv_idx, step)
        elif action_name == 'CONVERT_TO_BEAR_CALL_SPREAD': pm.convert_to_bear_call_spread(price, iv_idx, step)
        elif action_name == 'CONVERT_TO_BEAR_PUT_SPREAD': pm.convert_to_bear_put_spread(price, iv_idx, step)
        elif action_name == 'CONVERT_TO_CALL_CONDOR': pm.convert_to_condor('call', price, iv_idx, step)
        elif action_name == 'CONVERT_TO_PUT_CONDOR': pm.convert_to_condor('put', price, iv_idx, step)

    def _advance_market_and_get_outcome(self, equity_before: float, action_taken: str) -> BaseEnvTimestep:
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
        self.portfolio_manager.update_positions_after_time_step(time_decay_days, self.price_manager.current_price, self.iv_bin_index, self.current_step)

        self.portfolio_manager.debug_print_portfolio(
            current_price=self.price_manager.current_price,
            step=self.current_step,
            day=self.current_day_index,
            action_taken=action_taken
        )

        # --- FINAL, THREE-TIERED TERMINATION LOGIC ---
        terminated_by_rule = False
        final_shaped_reward_override = None
        termination_reason = "RUNNING" # Default state

        current_pnl = self.portfolio_manager.get_total_pnl(self.price_manager.current_price, self.iv_bin_index)

        # The stop-loss check is now DISABLED on the very first step of the episode.
        # It's not possible to be stopped out before the market has moved.
        if self._cfg.use_stop_loss and not self.portfolio_manager.portfolio.empty and self.current_step > 1:
            # This logic is now guaranteed to have the correct self.initial_net_premium.
            initial_cost_or_credit = abs(self.initial_net_premium)
            
            if initial_cost_or_credit > 1e-6:
                stop_loss_level = initial_cost_or_credit * self._cfg.stop_loss_multiple_of_cost
                unrealized_pnl = current_pnl - self.portfolio_manager.realized_pnl

                if unrealized_pnl <= -stop_loss_level:
                    terminated_by_rule = True
                    final_shaped_reward_override = -1.0
                    termination_reason = "STOP_LOSS"

        # 2. Take-Profit Rules (Only check if not already stopped out)
        if not terminated_by_rule and not self.portfolio_manager.portfolio.empty:

            # --- Rule 2a: Portfolio-Level "Home Run" Target ---
            if not terminated_by_rule:
                fixed_target_pct = self._cfg.profit_target_pct
                if fixed_target_pct > 0:
                    pnl_pct = (current_pnl / self.portfolio_manager.initial_cash) * 100
                    if pnl_pct >= fixed_target_pct:
                        terminated_by_rule = True
                        final_shaped_reward_override = self._cfg.jackpot_reward
                        
                        # <<< --- THE DEFINITIVE FIX: Descriptive Reason --- >>>
                        termination_reason = f"PORTFOLIO TARGET ({fixed_target_pct}%) MET"

            # --- Rule 2b: Dynamic Strategy-Level Target ---
            profit_target_pnl = self.fixed_profit_target_pnl
            if profit_target_pnl > 0 and current_pnl >= profit_target_pnl:
                terminated_by_rule = True
                final_shaped_reward_override = self._cfg.jackpot_reward
                
                # <<< --- THE DEFINITIVE FIX: Descriptive Reason --- >>>
                # Determine if it was a credit or debit target that was hit.
                net_premium = self.initial_net_premium
                if net_premium < 0: # Credit strategy
                    termination_reason = f"CREDIT TARGET ({self._cfg.credit_strategy_take_profit_pct}%) MET"
                else: # Debit strategy
                    termination_reason = f"DEBIT TARGET ({self._cfg.debit_strategy_take_profit_multiple}x) MET"

        # Final termination condition
        terminated_by_time = self.current_step >= self.total_steps
        terminated = terminated_by_rule or terminated_by_time

        if terminated and termination_reason == "RUNNING":
            termination_reason = "TIME_LIMIT"

        if terminated_by_rule: self.portfolio_manager.close_all_positions(self.price_manager.current_price, self.iv_bin_index, self.current_step)

        # --- Calculate Reward ---
        equity_after = self.portfolio_manager.get_current_equity(self.price_manager.current_price, self.iv_bin_index)
        
        # <<< --- Calculate the Daily Opportunity Cost --- >>>
        opportunity_cost_penalty = 0.0
        # This penalty is applied only at the end of each trading day.
        is_end_of_day = (self.current_step % self.steps_per_day) == 0
        if is_end_of_day:
            # 1. Start with a default penalty for a single calendar day.
            penalty_multiplier = 1.0
            
            # 2. Check if the day that just ended was a Friday.
            # We use (current_day_index - 1) because current_day_index has already ticked over.
            day_of_week_that_ended = (self.current_day_index - 1) % self.TRADING_DAYS_IN_WEEK
            if day_of_week_that_ended == 4: # 4 corresponds to Friday
                # If it's Friday, the cost of holding is for 3 calendar days (Fri, Sat, Sun).
                penalty_multiplier = 3.0

        # Pass this cost to the reward function.
        shaped_reward, raw_reward = self._calculate_shaped_reward(equity_before, equity_after, opportunity_cost_penalty)
        
        self.final_eval_reward += raw_reward

        # Use the stored action info to determine the final reward
        was_illegal_action = self.last_action_info['was_illegal_action']
        if final_shaped_reward_override is not None: final_reward = final_shaped_reward_override
        elif was_illegal_action: final_reward = self._cfg.illegal_action_penalty
        else: final_reward = shaped_reward

        # --- Prepare and Return Timestep ---
        obs = self._get_observation()
        action_mask = self._get_true_action_mask() if not self._cfg.ignore_legal_actions else np.ones(self.action_space_size, dtype=np.int8)

        # --- Create the base info dictionary ---
        info = {
            'price': self.price_manager.current_price,
            'eval_episode_return': self.final_eval_reward,
            'illegal_actions_in_episode': self.illegal_action_count,
            'was_illegal_action': bool(was_illegal_action),
            'executed_action_name': self.last_action_info['final_action_name'],
            'portfolio_stats': self.portfolio_manager.get_raw_portfolio_stats(self.price_manager.current_price, self.iv_bin_index),
            'total_steps_in_episode': self.total_steps,
            'market_regime': self.current_iv_regime_name,
            'termination_reason': termination_reason
        }

        # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
        # Only calculate and add the BiasMeter diagnostics if we are in evaluation mode.
        if self.is_eval_mode:
            meter = BiasMeter(obs[:self.model_observation_size], self.OBS_IDX)
            info['directional_bias'] = meter.directional_bias
            info['volatility_bias'] = meter.volatility_bias
        else:
            # During training, we provide default placeholder values.
            info['directional_bias'] = "N/A (Training)"
            info['volatility_bias'] = "N/A (Training)"

        if terminated:
            # `self.current_step` has already been incremented, so it reflects the total number of steps.
            num_steps_in_episode = self.current_step
            info['episode_duration'] = num_steps_in_episode
            
            if num_steps_in_episode <= 1:
                error_message = (
                    f"\n\n==================== CRITICAL ERROR: Invalid Episode Generated ====================\n"
                    f"Episode finished after only {num_steps_in_episode} step(s).\n"
                    f"This is a bug in the environment's termination logic.\n"
                    f"\n--- FINAL STEP INFO (BLACK BOX RECORDER) ---\n"
                    f"{info}\n" # The 'info' dict from this step contains the crucial termination_reason
                    f"====================================================================================\n\n"
                )
                raise RuntimeError(error_message)

        # --- FINAL STATE VALIDATION (The Definitive Safety Net) ---
        
        # 1. Check for portfolio size violations
        portfolio_size = len(self.portfolio_manager.portfolio)
        max_size = self._cfg.max_positions
        assert portfolio_size <= max_size, \
            f"FATAL INVARIANT VIOLATION: Portfolio size ({portfolio_size}) has exceeded max_positions ({max_size}). Action taken: '{action_taken}'"

        # 2. Check for invalid or temporary Strategy IDs
        if not self.portfolio_manager.portfolio.empty:
            # Check if any leg in the portfolio has a negative strategy_id
            invalid_ids_exist = (self.portfolio_manager.portfolio['strategy_id'] < 0).any()
            assert not invalid_ids_exist, \
                f"FATAL: Portfolio contains legs with invalid Strategy IDs (< 0) after action: '{action_taken}'.\n{self.portfolio_manager.portfolio.to_string()}"

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
            return self.np_random.choice(legal_open_indices), True

        # Override Rule 4: Default fallback for all other mid-episode illegal moves
        return self.actions_to_indices['HOLD'], True

    def _get_observation(self) -> np.ndarray:
        # Market State
        vec = np.zeros(self.model_observation_size, dtype=np.float32)
        vec[self.OBS_IDX['PRICE_NORM']] = (self.price_manager.current_price / self.price_manager.start_price) - 1.0
        vec[self.OBS_IDX['TIME_NORM']] = (self.total_steps - self.current_step) / self.total_steps
        vec[self.OBS_IDX['PNL_NORM']] = math.tanh(self.portfolio_manager.get_total_pnl(self.price_manager.current_price, self.iv_bin_index) / self._cfg.initial_cash)
        log_return = math.log(self.price_manager.current_price / (self.price_manager.price_path[self.current_step - 1] + 1e-8)) if self.current_step > 0 else 0.0
        vec[self.OBS_IDX['LOG_RETURN']] = np.clip(log_return, -0.1, 0.1) * 10

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

        # 1. Add the final probabilities from the Master Directional Expert
        dir_probs = self.price_manager.directional_prediction_probs
        if dir_probs is not None and len(dir_probs) == 3:
            vec[self.OBS_IDX['DIR_EXPERT_PROB_DOWN']] = dir_probs[0]
            vec[self.OBS_IDX['DIR_EXPERT_PROB_NEUTRAL']] = dir_probs[1]
            vec[self.OBS_IDX['DIR_EXPERT_PROB_UP']] = dir_probs[2]
        else: # Failsafe
            vec[self.OBS_IDX['DIR_EXPERT_PROB_DOWN']] = 0.33
            vec[self.OBS_IDX['DIR_EXPERT_PROB_NEUTRAL']] = 0.34
            vec[self.OBS_IDX['DIR_EXPERT_PROB_UP']] = 0.33

        # 2. Add the embedding from EACH expert in the council
        expert_embeddings = {
            'volatility': self.price_manager.volatility_embedding,
            'trend': self.price_manager.trend_embedding,
            'oscillator': self.price_manager.oscillator_embedding,
            'deviation': self.price_manager.deviation_embedding,
            'cycle': self.price_manager.cycle_embedding,
            'pattern': self.price_manager.pattern_embedding,
        }

        for name, embedding in expert_embeddings.items():
            start_idx_key = f'{name.upper()}_EMBEDDING_START'
            embedding_size = self.vol_embedding_size if name == 'volatility' else self.mlp_embedding_size
            
            if embedding is not None and len(embedding) == embedding_size:
                start_idx = self.OBS_IDX[start_idx_key]
                end_idx = start_idx + embedding_size
                vec[start_idx:end_idx] = embedding           

        # Per-Position State
        self.portfolio_manager.get_positions_state(vec, self.PORTFOLIO_START_IDX, self.PORTFOLIO_STATE_SIZE_PER_POS, self.POS_IDX, self.price_manager.current_price, self.iv_bin_index, self.current_step, self.total_steps)

        # 1. First, concatenate the state vector and the action mask
        true_action_mask = self._get_true_action_mask()
        final_obs_vec = np.concatenate((vec, true_action_mask.astype(np.float32)))

        # --- 3. THE FIX: FINAL, UNBREAKABLE SANITIZATION PASS ---
        # This is the most robust way to prevent CUDA errors. It checks the
        # entire completed observation vector for any bad numbers and replaces them.
        #if not np.all(np.isfinite(final_obs_vec)):
        #    # Find the indices of any inf or NaN values
        #    bad_indices = np.where(~np.isfinite(final_obs_vec))
        #    # Replace them with a safe default (0.0)
        #    final_obs_vec[bad_indices] = 0.0
        #    
        #    # This print statement is invaluable for debugging. It will tell you
        #    # exactly which part of your observation vector is producing bad data.
        #    print(f"WARNING: Found and corrected non-finite values in observation vector at indices: {bad_indices}")

        # 2. Then, run assertions on the FINAL object
        assert final_obs_vec.shape == (self.framework_vec_size,), f"Observation shape mismatch. Expected {self.framework_vec_size}, but got {final_obs_vec.shape}"
        assert np.all(np.isfinite(final_obs_vec)), "Observation vector contains NaN or Inf."

        # 3. Return the fully validated final vector
        return final_obs_vec

    def _calculate_shaped_reward(self, equity_before: float, equity_after: float, opportunity_cost_penalty: float) -> Tuple[float, float]:
        """
        Calculates the final shaped reward for the agent.
        MODIFIED: Now includes a penalty for the opportunity cost of capital.
        """
        # --- 1. Calculate Raw P&L Change ---
        raw_reward = equity_after - equity_before
        
        # --- 2. Update Water Marks and Calculate Drawdown ---
        self.portfolio_manager.update_mtm_water_marks(equity_after)
        drawdown = self.portfolio_manager.high_water_mark - equity_after
        
        # --- 3. Calculate the Drawdown Penalty ---
        drawdown_penalty = self._cfg.drawdown_penalty_weight * drawdown
        
        # --- 4. Calculate the Capital Preservation Bonus ---
        capital_preservation_bonus = 0.0
        current_pnl = equity_after - self._cfg.initial_cash
        if self.capital_preservation_bonus_pct > 0 and current_pnl > 0:
            capital_preservation_bonus = current_pnl * self.capital_preservation_bonus_pct
            
        # --- 5. Combine Components into the Final Reward ---
        # The agent's reward is now its P&L, adjusted for risk, and "taxed" by the risk-free rate.
        risk_adjusted_reward = raw_reward - drawdown_penalty + capital_preservation_bonus - opportunity_cost_penalty
        
        # --- 6. Scale and Squash the Reward ---
        scaled_reward = risk_adjusted_reward / self.pnl_scaling_factor
        if not math.isfinite(scaled_reward):
            print(f"WARNING: Calculated a non-finite reward. Raw: {raw_reward}, Drawdown: {drawdown_penalty}, Bonus: {capital_preservation_bonus}, OppCost: {opportunity_cost_penalty}")
            scaled_reward = 0.0

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

        # <<< --- ADD THE NEW CONDOR CONVERSION ACTIONS HERE --- >>>
        actions['CONVERT_TO_CALL_CONDOR'] = i; i+=1
        actions['CONVERT_TO_PUT_CONDOR'] = i; i+=1

        # --- NEW: Strategy Morphing Actions ---
        actions['CONVERT_TO_IRON_CONDOR'] = i; i+=1
        actions['CONVERT_TO_IRON_FLY'] = i; i+=1
        actions['CONVERT_TO_STRANGLE'] = i; i+=1
        actions['CONVERT_TO_STRADDLE'] = i; i+=1
        actions['CONVERT_TO_BULL_CALL_SPREAD'] = i; i+=1
        actions['CONVERT_TO_BEAR_CALL_SPREAD'] = i; i+=1
        actions['CONVERT_TO_BULL_PUT_SPREAD'] = i; i+=1
        actions['CONVERT_TO_BEAR_PUT_SPREAD'] = i; i+=1

        actions['RECENTER_VOLATILITY_POSITION'] = i; i+=1

        actions['HEDGE_DELTA_WITH_ATM_OPTION'] = i; i+=1

        for d in ['LONG', 'SHORT']:
            actions[f'OPEN_{d}_STRADDLE'] = i; i+=1

        # Generate a range of delta-based strangle actions.
        for delta in range(15, 31, 5): # This creates [15, 20, 25, 30]
            actions[f'OPEN_LONG_STRANGLE_DELTA_{delta}'] = i; i+=1
            actions[f'OPEN_SHORT_STRANGLE_DELTA_{delta}'] = i; i+=1

        for d in ['LONG', 'SHORT']:
            actions[f'OPEN_{d}_IRON_FLY'] = i; i+=1
            actions[f'OPEN_{d}_IRON_CONDOR'] = i; i+=1

            for t in ['CALL', 'PUT']:
                actions[f'OPEN_{d}_{t}_CONDOR'] = i; i+=1

        # The loop for 'w' (width) is removed entirely.
        for t in ['CALL', 'PUT']:
            for d in ['LONG', 'SHORT']:
                actions[f'OPEN_{d}_{t}_FLY'] = i; i+=1 # <-- No more width number

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

        # <<< --- NEW PER-LEG DELTA SHIFT ACTIONS --- >>>
        for j in range(self._cfg.max_positions):
            actions[f'INCREASE_DELTA_BY_SHIFTING_LEG_{j}'] = i; i+=1
            actions[f'DECREASE_DELTA_BY_SHIFTING_LEG_{j}'] = i; i+=1

        # <<< --- NEW: Add the powerful risk-shedding roll action --- >>>
        for j in range(self._cfg.max_positions):
            actions[f'HEDGE_PORTFOLIO_BY_ROLLING_LEG_{j}'] = i; i+=1

        # <<< --- NEW: Add the six advanced strategies --- >>>
        actions['OPEN_JADE_LIZARD'] = i; i += 1
        actions['OPEN_REVERSE_JADE_LIZARD'] = i; i += 1
        actions['OPEN_BIG_LIZARD'] = i; i += 1
        actions['OPEN_REVERSE_BIG_LIZARD'] = i; i += 1
        actions['OPEN_PUT_RATIO_SPREAD'] = i; i += 1
        actions['OPEN_CALL_RATIO_SPREAD'] = i; i += 1

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
        """
        Computes the action mask for when the portfolio has one or more positions.
        This version uses independent checks for each action type to prevent
        logic errors from chained if/elif statements.
        """
        action_mask = np.zeros(self.action_space_size, dtype=np.int8)
        portfolio_df = self.portfolio_manager.get_portfolio()
        
        # --- 1. Universal & Per-Leg Actions ---
        action_mask[self.actions_to_indices['HOLD']] = 1
        action_mask[self.actions_to_indices['CLOSE_ALL']] = 1
        atm_price = self.market_rules_manager.get_atm_price(self.price_manager.current_price)
        for i, pos in portfolio_df.iterrows():
            self._set_if_exists(action_mask, f'CLOSE_POSITION_{i}')
            if not pos['is_hedged']:
                self._set_if_exists(action_mask, f'HEDGE_NAKED_POS_{i}')
                if pos['strike_price'] != atm_price:
                    # The flawed SHIFT_TO_ATM action is now correctly restricted.
                    self._set_if_exists(action_mask, f'SHIFT_TO_ATM_{i}')
            self._set_shift_if_no_conflict(action_mask, i, pos, direction="UP")
            self._set_shift_if_no_conflict(action_mask, i, pos, direction="DOWN")
            if self._is_delta_shift_possible(pos, 'increase'): self._set_if_exists(action_mask, f'INCREASE_DELTA_BY_SHIFTING_LEG_{i}')
            if self._is_delta_shift_possible(pos, 'decrease'): self._set_if_exists(action_mask, f'DECREASE_DELTA_BY_SHIFTING_LEG_{i}')

        # --- 2. Whole-Portfolio Transformation Actions ---
        if not portfolio_df.empty:
            current_strategy_id = portfolio_df.iloc[0]['strategy_id']
            option_type = portfolio_df.iloc[0]['type']
            s_map = self.strategy_name_to_id

            # All sets are now built using the correct internal strategy names.
            short_strangle_ids = {s_map.get(f'SHORT_STRANGLE_DELTA_{delta}') for delta in [15, 20, 25, 30]}
            short_straddle_id = s_map.get('SHORT_STRADDLE')
            iron_condor_ids = {s_map.get(f'{d}_IRON_CONDOR') for d in ['LONG', 'SHORT']}
            call_condor_ids = {s_map.get(f'{d}_CALL_CONDOR') for d in ['LONG', 'SHORT']}
            put_condor_ids = {s_map.get(f'{d}_PUT_CONDOR') for d in ['LONG', 'SHORT']}
            fly_id = s_map.get('SHORT_IRON_FLY')
            call_fly_ids = {s_map.get(f'{d}_CALL_FLY') for d in ['LONG', 'SHORT']}
            put_fly_ids = {s_map.get(f'{d}_PUT_FLY') for d in ['LONG', 'SHORT']}
            spread_ids = {s_map.get(f'{d}_{t}_SPREAD') for d in ['BULL', 'BEAR'] for t in ['CALL', 'PUT']}

            
            # --- Rules for 2-Leg Positions ---
            if len(portfolio_df) == 2:
                if current_strategy_id in short_strangle_ids and len(portfolio_df) <= self._cfg.max_positions - 2:
                    self._set_if_exists(action_mask, 'CONVERT_TO_IRON_CONDOR')
                if current_strategy_id == short_straddle_id and len(portfolio_df) <= self._cfg.max_positions - 2:
                    self._set_if_exists(action_mask, 'CONVERT_TO_IRON_FLY')
                if current_strategy_id in spread_ids and len(portfolio_df) <= self._cfg.max_positions - 2:
                    if option_type == 'call': self._set_if_exists(action_mask, 'CONVERT_TO_CALL_CONDOR')
                    else: self._set_if_exists(action_mask, 'CONVERT_TO_PUT_CONDOR')
                is_short_vol_position = current_strategy_id in short_strangle_ids or current_strategy_id == short_straddle_id
                if is_short_vol_position:
                    greeks = self.portfolio_manager.get_portfolio_greeks(self.price_manager.current_price, self.iv_bin_index)
                    if abs(greeks['delta_norm']) > self.delta_neutral_threshold:
                        self._set_if_exists(action_mask, 'RECENTER_VOLATILITY_POSITION')

            # --- Rules for 4-Leg Positions ---
            if len(portfolio_df) == 4:
                if current_strategy_id in iron_condor_ids:
                    self._set_if_exists(action_mask, 'CONVERT_TO_STRANGLE')
                    self._set_if_exists(action_mask, 'CONVERT_TO_BULL_PUT_SPREAD')
                    self._set_if_exists(action_mask, 'CONVERT_TO_BEAR_CALL_SPREAD')
                if current_strategy_id in call_condor_ids or current_strategy_id in call_fly_ids:
                    self._set_if_exists(action_mask, 'CONVERT_TO_BULL_CALL_SPREAD')
                    self._set_if_exists(action_mask, 'CONVERT_TO_BEAR_CALL_SPREAD')
                if current_strategy_id in put_condor_ids or current_strategy_id in put_fly_ids:
                    self._set_if_exists(action_mask, 'CONVERT_TO_BULL_PUT_SPREAD')
                    self._set_if_exists(action_mask, 'CONVERT_TO_BEAR_PUT_SPREAD')
                if current_strategy_id == fly_id:
                    self._set_if_exists(action_mask, 'CONVERT_TO_STRADDLE')

        # --- 3. Portfolio-Level Delta Management ---
        greeks = self.portfolio_manager.get_portfolio_greeks(self.price_manager.current_price, self.iv_bin_index)
        if abs(greeks['delta_norm']) > self.delta_neutral_threshold:
            if len(portfolio_df) < self._cfg.max_positions:
                # a) Determine what the hedge leg WOULD be
                hedge_leg_type = 'put' if greeks['delta_norm'] > 0 else 'call'
                
                # b) Check if this proposed leg would create a conflict
                is_conflict = any(
                    (leg['type'] == hedge_leg_type and leg['strike_price'] == atm_price)
                    for _, leg in portfolio_df.iterrows()
                )
                
                # c) Only make the action legal if there is NO conflict.
                if not is_conflict:
                    self._set_if_exists(action_mask, 'HEDGE_DELTA_WITH_ATM_OPTION')
            for i in range(len(portfolio_df)):
                self._set_if_exists(action_mask, f'HEDGE_PORTFOLIO_BY_ROLLING_LEG_{i}')
        
        return action_mask

    def _is_delta_shift_possible(self, leg_data: pd.Series, direction: str) -> bool:
        """
        Determines if a leg can be shifted to achieve a desired delta change.
        """
        leg_type = leg_data['type']
        leg_dir = leg_data['direction']
        
        if direction == 'increase': # To increase delta, you want a more positive (or less negative) value
            # Move towards ITM for longs, away from ITM for shorts
            if (leg_type == 'call' and leg_dir == 'long') or (leg_type == 'put' and leg_dir == 'short'):
                return True # SHIFT_DOWN is possible
            elif (leg_type == 'put' and leg_dir == 'long') or (leg_type == 'call' and leg_dir == 'short'):
                return True # SHIFT_UP is possible
        
        elif direction == 'decrease': # To decrease delta, you want a more negative (or less positive) value
            # Move away from ITM for longs, towards ITM for shorts
            if (leg_type == 'call' and leg_dir == 'long') or (leg_type == 'put' and leg_dir == 'short'):
                return True # SHIFT_UP is possible
            elif (leg_type == 'put' and leg_dir == 'long') or (leg_type == 'call' and leg_dir == 'short'):
                return True # SHIFT_DOWN is possible
        
        return False

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

    def _is_legal_opening_action(self, action_name: str, atm_price: float, days_to_expiry: float) -> bool:
        """
        Check if a given opening action is legal. This definitive version uses
        more permissive and robust rules.
        """
        # --- Rule 1: Multi-leg strategies are always considered legal at this stage ---
        # Their legality is determined by slot constraints later.
        is_single_leg = 'ATM' in action_name and 'STRADDLE' not in action_name and 'STRANGLE' not in action_name
        if not is_single_leg:
            return True

        # --- Rule 2: Single-leg specific rules ---
        try:
            parts = action_name.split('_')
            direction, option_type, offset_str = parts[1], parts[2].lower(), parts[3].replace('ATM', '')
            offset = int(offset_str)
        except (ValueError, IndexError):
            return False # Invalid action name format

        # --- Rule 2a: Prevent shorting deep in-the-money options ---
        # This is a critical risk management rule.
        if direction == 'SHORT':
            strike_price = atm_price + (offset * self._cfg.strike_distance)
            if strike_price <= 0: return False

            # Use the dynamic IV calculator for an accurate delta
            vol = self._get_dynamic_iv(offset, option_type)
            greeks = self.bs_manager.get_all_greeks_and_price(
                self.price_manager.current_price, strike_price, days_to_expiry, vol, option_type == 'call'
            )
            abs_delta = abs(greeks['delta'])

            # Disallow shorting any option that has more than an 80% chance of expiring ITM.
            if abs_delta > 0.80:
                return False
        
        # If none of the above rules failed, the action is legal.
        # We no longer restrict long options by delta at this stage.
        return True

    def _apply_slot_constraints(self, base_mask: np.ndarray) -> np.ndarray:
        """Apply max position slot constraints to legal opening actions."""
        final_mask = np.zeros_like(base_mask)
        available_slots = self.portfolio_manager.max_positions

        for index, is_legal in enumerate(base_mask):
            if not is_legal:
                continue
            name = self.indices_to_actions[index]

            # 1. We add a new, more specific check for our 3-leg strategies.
            #    This must come BEFORE the generic 'SPREAD' check.
            if 'JADE_LIZARD' in name or 'BIG_LIZARD' in name or 'RATIO_SPREAD' in name:
                if available_slots >= 3:
                    final_mask[index] = 1
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
            # <<< --- THE CORRECTED LOGIC IS HERE --- >>>
            # This is a more robust way to identify single legs. It positively
            # checks for the unique substrings of single leg actions.
            "SINGLE_LEG": lambda name: '_CALL_ATM' in name or '_PUT_ATM' in name,
            
            "STRADDLE": lambda name: 'STRADDLE' in name,
            "STRANGLE": lambda name: 'STRANGLE' in name,
            "SPREAD": lambda name: 'SPREAD' in name,
            "IRON_FLY_AND_CONDORS": lambda name: 'IRON' in name or 'CONDOR' in name,
            "BUTTERFLY": lambda name: 'FLY' in name and 'IRON' not in name,
        }

        chosen_family_name = self.np_random.choice(list(strategy_families.keys()))
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
        current_obs_vector = self._get_observation()[:self.model_observation_size]
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
