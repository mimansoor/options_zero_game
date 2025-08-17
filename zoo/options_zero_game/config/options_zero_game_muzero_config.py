# zoo/options_zero_game/config/options_zero_game_muzero_config.py
import numpy as np # Make sure numpy is imported
import copy
from easydict import EasyDict

# Import the environment to get its version and default parameters
from zoo.options_zero_game.envs.options_zero_game_env import OptionsZeroGameEnv
from zoo.options_zero_game.envs.utils import generate_dynamic_iv_skew_table

# ==============================================================
#                 Static Parameters
# ==============================================================
# ==============================================================
#                 Optimized Static Parameters
# ==============================================================
# Utilize the strong CPU, but avoid over-subscription. Matched to n_episode.
collector_env_num = 20
# Collect one episode per environment worker before sending data.
n_episode = 20
# Evaluation can also be parallelized, but is less critical for training speed.
evaluator_env_num = 10
# Kept at 256, a good balance for 8GB VRAM. Reduce to 128 if you see CUDA memory errors.
batch_size = 256
# Increased for higher quality moves. The RTX 2070 can handle this.
num_simulations = 50
# Significantly increased to keep the GPU busy and improve sample efficiency.
update_per_collect = 2000
# This results in a much healthier replay_ratio. (2000 / (20 episodes * ~40 steps)) = ~2.5
replay_ratio = 2.5
max_env_step = int(7e8)
reanalyze_ratio = 0.

# <<< NEW: Add this powerful helper function at the top of your config file >>>
def generate_dynamic_iv_skew_table(max_offset: int, atm_iv: float, far_otm_put_iv: float, far_otm_call_iv: float) -> dict:
    """
    Generates a realistic, dynamic IV skew table using a quadratic curve.
    This creates a "volatility smirk" where OTM puts have the highest IV.

    Args:
        max_offset: The max number of strikes from ATM (e.g., 30).
        atm_iv: The IV at the money (e.g., 20.0 for 20%).
        far_otm_put_iv: The IV at the furthest OTM put strike (e.g., 45.0).
        far_otm_call_iv: The IV at the furthest OTM call strike (e.g., 18.0).

    Returns:
        A dictionary in the format required by the MarketRulesManager.
    """
    # We solve a system of equations for a quadratic: y = ax^2 + bx + c
    # The points are (-max_offset, far_otm_put_iv), (0, atm_iv), (max_offset, far_otm_call_iv)

    A = np.array([
        [max_offset**2, -max_offset, 1],
        [0, 0, 1],
        [max_offset**2, max_offset, 1]
    ])
    B = np.array([far_otm_put_iv, atm_iv, far_otm_call_iv])

    a, b, c = np.linalg.solve(A, B)

    skew_table = {'call': {}, 'put': {}}
    for offset in range(-max_offset, max_offset + 1):
        # Calculate the IV on the quadratic curve
        iv = a * offset**2 + b * offset + c
        # The table expects a [min_iv, max_iv] range, so we create a small range around the point.
        iv_range = [max(5.0, iv - 1.0), iv + 1.0] # Ensure IV doesn't drop below 5%

        # The same skew curve applies to both puts and calls
        skew_table['call'][str(offset)] = iv_range
        skew_table['put'][str(offset)] = iv_range

    return skew_table

# ==============================================================
#           UNIFIED REGIME DEFINITIONS (The Single Source of Truth)
# ==============================================================
# Each dictionary now contains BOTH GARCH parameters for price movement
# AND IV parameters for option pricing.
UNIFIED_REGIMES = [
    {
        'name': 'Crisis (High Vol, Bearish)',
        # GARCH Params
        'mu': -0.0005, 'omega': 0.0001, 'alpha': 0.15, 'beta': 0.82, 'overnight_vol_multiplier': 2.2,
        # IV Params
        'atm_iv': 50.0, 'far_otm_put_iv': 90.0, 'far_otm_call_iv': 40.0,
    },
    {
        'name': 'Normal (Medium Vol, Bullish)',
        # GARCH Params (Using SPY_Real as a baseline)
        'mu': 0.000905, 'omega': 0.00000397, 'alpha': 0.1280, 'beta': 0.8404, 'overnight_vol_multiplier': 5.99,
        # IV Params
        'atm_iv': 25.0, 'far_otm_put_iv': 50.0, 'far_otm_call_iv': 20.0,
    },
    {
        'name': 'Complacent (Low Vol, Neutral)',
        # GARCH Params (Using Bond_Markets as a baseline)
        'mu': 0.00001, 'omega': 0.000002, 'alpha': 0.05, 'beta': 0.92, 'overnight_vol_multiplier': 1.1,
        # IV Params
        'atm_iv': 12.0, 'far_otm_put_iv': 20.0, 'far_otm_call_iv': 10.0,
    },
    {
        'name': 'Crypto (High Vol, High Growth)',
        # GARCH Params
        'mu': 0.0015, 'omega': 0.0001, 'alpha': 0.10, 'beta': 0.88, 'overnight_vol_multiplier': 1.85,
        # IV Params
        'atm_iv': 75.0, 'far_otm_put_iv': 110.0, 'far_otm_call_iv': 70.0,
    },
]

# ==============================================================
#                 Curriculum Schedule
# ==============================================================
# A structured training plan that teaches the agent concepts in phases.
# Each phase focuses on a core strategy for 2 million steps before moving to the next.

TRAINING_CURRICULUM = {
    # === Phase 1: Foundational Premium Selling (Directional) ===
    # Goal: Learn to sell premium with a bullish assumption (theta decay).
    0: 'OPEN_SHORT_PUT_ATM-5',

    # === Phase 2: Counterpart Directional Premium Selling ===
    # Goal: Learn to sell premium with a bearish assumption.
    int(2e6): 'OPEN_SHORT_CALL_ATM+5',

    # === Phase 2: Counterpart Directional Premium Selling ===
    # Goal: Learn to sell premium with a bearish assumption.
    int(3e6): 'OPEN_SHORT_PUT_ATM+5',

    # === Phase 5: Selling Volatility (Undefined Risk) ===
    # Goal: Learn a non-directional strategy by selling ATM volatility (Straddles).
    int(4e6): 'OPEN_SHORT_STRADDLE',

    # === Phase 5: Selling Volatility (Undefined Risk) ===
    # Goal: Learn a non-directional strategy by selling ATM volatility (Straddles).
    int(5e6): 'OPEN_LONG_STRADDLE',

    # === Phase 6: Selling Volatility (Risk Defined) ===
    # Goal: Learn to create a range-bound, positive theta position with defined risk.
    int(6e6): 'OPEN_SHORT_IRON_CONDOR',

    # === Phase 6: Selling Volatility (Risk Defined) ===
    # Goal: Learn to create a range-bound, positive theta position with defined risk.
    int(7e6): 'OPEN_LONG_IRON_CONDOR',

    # === Phase 7: Advanced Range-Bound Strategy ===
    # Goal: Learn a different range-bound profile with a sharper profit peak (Butterflies).
    int(8e6): 'OPEN_SHORT_CALL_FLY_2',

    # === Phase 7: Advanced Range-Bound Strategy ===
    # Goal: Learn a different range-bound profile with a sharper profit peak (Butterflies).
    int(9e6): 'OPEN_LONG_CALL_FLY_2',

    # === Phase 8: Advanced Volatility Selling (Delta-based) ===
    # Goal: Learn the professional method of entering strangles based on delta, not fixed widths.
    int(10e6): 'OPEN_SHORT_STRANGLE_DELTA_15',

    # === Phase 8: Advanced Volatility Selling (Delta-based) ===
    # Goal: Learn the professional method of entering strangles based on delta, not fixed widths.
    int(11e6): 'OPEN_LONG_STRANGLE_DELTA_15',
    
    # === Phase 9: Learning to BUY Options (Debit Strategies) ===
    # Goal: Change mindset. Learn to pay theta for a large directional move (Long Calls).
    int(12e6): 'OPEN_LONG_CALL_ATM+5',

    # === Phase 9: Learning to BUY Options (Debit Strategies) ===
    # Goal: Change mindset. Learn to pay theta for a large directional move (Long Calls).
    int(13e6): 'OPEN_LONG_CALL_ATM-5',

    # === Phase 10: Learning to BUY Spreads ===
    # Goal: Learn to make a risk-defined directional bet by buying a Bull Call Spread.
    int(14e6): 'OPEN_BULL_CALL_SPREAD',

    # === Phase 11: Learning to BUY Spreads ===
    # Goal: Learn to make a risk-defined directional bet by buying a Bear Call Spread.
    int(16e6): 'OPEN_BEAR_CALL_SPREAD',

    # === Phase 12: Learning to BUY Spreads ===
    # Goal: Learn to make a risk-defined directional bet by buying a Bull Put Spread.
    int(18e6): 'OPEN_BULL_PUT_SPREAD',

    # === Phase 13: Learning to BUY Spreads ===
    # Goal: Learn to make a risk-defined directional bet by buying a Bear Put Spread.
    int(20e6): 'OPEN_BEAR_PUT_SPREAD',

    # === Final Phase: Integration and Agent Autonomy ===
    # Goal: Allow the agent to use any of its learned strategies to maximize reward.
    int(22e6): 'ALL'
}

# This class will "hide" the integer-keyed dictionary from EasyDict.
class CurriculumHolder:
    def __init__(self, schedule):
        self.schedule = schedule

    # This tells Python how to "print" this object as valid, parsable code.
    def __repr__(self):
        # Return the string representation of the dictionary itself.
        return repr(self.schedule)

# ==============================================================
#           Main Config (The Parameters)
# ==============================================================
# This makes the script runnable from anywhere.
# <<< NEW: Define high-level volatility parameters >>>
MAX_STRIKE_OFFSET = 40
AGENT_MAX_OPEN_OFFSET = 2
ATM_IV = 25.0  # Volatility at the money is 25%
FAR_OTM_PUT_IV = 50.0 # Volatility for the -30 strike put is 50%
FAR_OTM_CALL_IV = 20.0 # Volatility for the +30 strike call is 20% (creating a "smirk")

# <<< THE FIX: Generate the skew table dynamically >>>
dynamic_iv_skew_table = generate_dynamic_iv_skew_table(
    max_offset=MAX_STRIKE_OFFSET,
    atm_iv=ATM_IV,
    far_otm_put_iv=FAR_OTM_PUT_IV,
    far_otm_call_iv=FAR_OTM_CALL_IV
)

# ==============================================================================
#                 Strategy ID Mapping (Definitive, Final Version)
# ==============================================================================
# This block programmatically generates all strategy IDs to ensure consistency
# and completeness, making it the single source of truth.

strategy_name_to_id = {}
next_id = 0

# --- 1. Simple Naked Legs (Internal Use for Re-profiling) ---
for direction in ['LONG', 'SHORT']:
    for opt_type in ['CALL', 'PUT']:
        strategy_name_to_id[f'{direction}_{opt_type}'] = next_id; next_id += 1

# Add the full action names for all single leg actions
agent_max_open_offset = AGENT_MAX_OPEN_OFFSET
for offset in range(-agent_max_open_offset, agent_max_open_offset + 1):
    strike_str = f"ATM{offset:+d}"
    for direction in ['LONG', 'SHORT']:
        for opt_type in ['CALL', 'PUT']:
            action_name = f'OPEN_{direction}_{opt_type}_{strike_str}'
            internal_name = f'{direction}_{opt_type}'
            strategy_name_to_id[action_name] = strategy_name_to_id[internal_name]

# --- 2. Core Volatility Strategies ---
for direction in ['LONG', 'SHORT']:
    internal_name = f'{direction}_STRADDLE'
    strategy_name_to_id[internal_name] = next_id; next_id += 1
    # The action name includes "_ATM"
    strategy_name_to_id[f'OPEN_{internal_name}'] = strategy_name_to_id[internal_name]

# The other core strategies do not have the suffix
for direction in ['LONG', 'SHORT']:
    for name in ['IRON_CONDOR', 'IRON_FLY']:
        internal_name = f'{direction}_{name}'
        strategy_name_to_id[internal_name] = next_id; next_id += 1
        strategy_name_to_id[f'OPEN_{internal_name}'] = strategy_name_to_id[internal_name]

# --- 3. Spreads ---
for direction in ['BULL', 'BEAR']:
    for opt_type in ['CALL', 'PUT']:
        internal_name = f'{direction}_{opt_type}_SPREAD'
        strategy_name_to_id[internal_name] = next_id; next_id += 1
        strategy_name_to_id[f'OPEN_{internal_name}'] = strategy_name_to_id[internal_name]

# --- 4. Strategies with Variations (Strangles, Butterflies) ---
for direction in ['LONG', 'SHORT']:
    # Delta Strangles
    for delta in [15, 20, 25, 30]:
        internal_name = f'{direction}_STRANGLE_DELTA_{delta}'
        strategy_name_to_id[internal_name] = next_id; next_id += 1
        strategy_name_to_id[f'OPEN_{internal_name}'] = strategy_name_to_id[internal_name]
    # Fixed-Width Butterflies
    for width in [1, 2]:
        for opt_type in ['CALL', 'PUT']:
            internal_name = f'{direction}_{opt_type}_FLY_{width}'
            strategy_name_to_id[internal_name] = next_id; next_id += 1
            strategy_name_to_id[f'OPEN_{internal_name}'] = strategy_name_to_id[internal_name]

# --- 5. Custom / Post-Adjustment States ---
strategy_name_to_id['CUSTOM_HEDGED'] = next_id; next_id += 1
strategy_name_to_id['CUSTOM_2_LEGS'] = next_id; next_id += 1
strategy_name_to_id['CUSTOM_3_LEGS'] = next_id; next_id += 1
strategy_name_to_id['CUSTOM_4_LEGS'] = next_id; next_id += 1
strategy_name_to_id['LONG_RATIO_SPREAD'] = next_id; next_id += 1
strategy_name_to_id['SHORT_RATIO_SPREAD'] = next_id; next_id += 1

options_zero_game_muzero_config = dict(
    # Define the main output directory for all experiments.
    exp_name=f'experiments/options_zero_game_muzero_agent_v{OptionsZeroGameEnv.VERSION}_ns{num_simulations}_upc{update_per_collect}_bs{batch_size}',
    env=dict(
        env_id='OptionsZeroGame-v0',
        env_version=OptionsZeroGameEnv.VERSION,
        # --- Collector-Specific Settings ---
        collector_env_num=collector_env_num,
        # The collector uses the opening curriculum.
        disable_opening_curriculum=False,
        
        # --- Evaluator-Specific Settings ---
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        is_eval_mode=False,

        # EasyDict will see this as a regular object, not a dict to convert.
        training_curriculum=CurriculumHolder(TRAINING_CURRICULUM),

        # The evaluator lets the agent choose its own move.
        # The framework will automatically use these settings for the evaluator envs.
        evaluator_env_cfg=dict(
            is_eval_mode=True,
            disable_opening_curriculum=True,

            # The evaluator ALWAYS uses a fixed, consistent episode length.
            # This ensures evaluations are always run on the full, max-length episode.
            forced_episode_length=40, 
        ),

        manager=dict(shared_memory=False, ),
        # Pass all necessary parameters to the environment
        price_source='mixed',
        historical_data_path='zoo/options_zero_game/data/market_data_cache',
        drawdown_penalty_weight=0.1,
        unified_regimes=UNIFIED_REGIMES,
        strategy_name_to_id=strategy_name_to_id,
        illegal_action_penalty=-1.0,
        rolling_vol_window=5,
        # The absolute IV percentage points to add to the expert's prediction.
        # This represents the market's inherent "volatility risk premium".
        # A value of 0.05 means we add 5% to the predicted IV.
        volatility_premium_abs=0.05,

        # The normalized delta threshold at which the ADJUST_TO_DELTA_NEUTRAL action becomes available.
        # A value of 0.1 means the action is legal if the portfolio's net delta
        # is more than 10% of the maximum possible delta.
        delta_neutral_threshold=0.1,

        # The target net debit for a butterfly as a percentage of the ATM strike.
        # 0.01 means we are trying to pay ~1% of the stock price for the butterfly.
        butterfly_target_cost_pct=0.01, 

        max_strike_offset=MAX_STRIKE_OFFSET,
        
        # <<< NEW: Add a parameter specifically for the agent's naked opening actions >>>
        # The agent can only OPEN naked positions within this narrower range.
        agent_max_open_offset=AGENT_MAX_OPEN_OFFSET,

        # This MUST match the 'sequence_length' used in the expert trainer CONFIG.
        expert_sequence_length=60,
        # <<< NEW: Parameter for the Capital Preservation Bonus >>>
        # A small percentage of the current P&L to add to the reward each step.
        capital_preservation_bonus_pct=0.001, # 0.1% of P&L per step
    ),
    policy=dict(
        model=dict(
            # These values will be dynamically set below
            observation_shape=0,
            action_space_size=0,
            model_type='mlp',
            lstm_hidden_size=512,
            latent_state_dim=512,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            # <<< THE FIX: Add these mandatory parameters >>>
            # This explicitly tells the model the range of rewards and values to expect.
            # This MUST be wide enough to handle the large P&L swings in your financial env.
            # A range of -50k to +50k with a step/bucket size of 250 is a robust choice.
            reward_support_size=401,  # Corresponds to a range of (-50000, 50000) with step 250
            value_support_size=401,   # Must match reward_support_size
        ),
        model_path='./best_ckpt/ckpt_best.pth.tar',
        cuda=True,
        game_segment_length=2 * OptionsZeroGameEnv.config['time_to_expiry_days'] * OptionsZeroGameEnv.config['steps_per_day'],
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        td_steps=10,
        discount_factor=0.999,
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=int(1e5),
        piecewise_decay_lr_scheduler=False,
        optim_type='Adam',
        learning_rate=3e-4,  # This is 0.0003
        weight_decay=1e-4,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,

        #learn=dict(
        #    learner=dict(
        #        hook=dict(
        #            # Provide the full path to the checkpoint you want to resume from
        #            load_ckpt_before_run='./best_ckpt/ckpt_best.pth.tar'
        #        )
        #    )
        #),
    ),
)
main_config = EasyDict(options_zero_game_muzero_config)

# ==============================================================
#             Dynamic Configuration (The Correct Way)
# ==============================================================
# Create a temporary environment to dynamically get the observation and action space sizes.
# This ensures the config is always in sync with the environment.
temp_env_cfg = copy.deepcopy(main_config.env)
temp_env = OptionsZeroGameEnv(temp_env_cfg)
action_space_size = temp_env.action_space.n
# CRITICAL FIX: The environment's observation_space.shape is a tuple (e.g., (99,)).
# The model requires an integer for the MLP input dimension. We extract it here.
observation_shape = temp_env.observation_space.shape[0]
del temp_env  # Clean up

# --- Update the main_config with the correct, dynamically-determined values ---
main_config.policy.model.observation_shape = observation_shape
main_config.policy.model.action_space_size = action_space_size

# ==============================================================
#                  Create-Config (The Blueprint)
# ==============================================================
create_config = dict(
    env=dict(
        type='options_zero_game',
        import_names=['zoo.options_zero_game.envs.options_zero_game_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
create_config = EasyDict(create_config)

if __name__ == "__main__":
    # ==============================================================
    #           Automatic Training Resumption Logic
    # ==============================================================
    import os
    from easydict import EasyDict

    # --- Define the path to the checkpoint you want to resume from ---
    # This is now the single source of truth for resuming.
    # Let's point it to the 'ckpt_latest.pth.tar' in a specific experiment for robustness.
    # NOTE: You might want to update this path to a more general one like './best_ckpt/ckpt_best.pth.tar'
    # if that is your intended workflow.
    resume_path = './best_ckpt/ckpt_best.pth.tar'

    # Check if the checkpoint file actually exists.
    if os.path.exists(resume_path):
        print(f"\n--- Checkpoint found at '{resume_path}'. ---")
        print("--- Configuring to RESUME training. ---\n")
        
        # If it exists, dynamically create and populate the 'learn' hook.
        # We create empty EasyDicts to ensure the nested structure exists.
        main_config.policy.learn = EasyDict({})
        main_config.policy.learn.learner = EasyDict({})
        main_config.policy.learn.learner.hook = EasyDict({})
        main_config.policy.learn.learner.hook.load_ckpt_before_run = resume_path
    else:
        print(f"\n--- No checkpoint found at '{resume_path}'. ---")
        print("--- Starting a FRESH training run. ---\n")
        # If the file does not exist, we do nothing. The 'learn' hook will not be
        # added to the config, and the framework will start training from scratch.

    import argparse
    import time
    from lzero.entry import train_muzero

    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Train the Options-Zero-Game MuZero Agent.")
    parser.add_argument(
        '--seed', 
        type=int, 
        default=0, 
        help="Master seed for the experiment. Defaults to 0 for reproducibility. "
             "Use a negative value (e.g., -1) to generate a random seed based on the current time."
    )
    args = parser.parse_args()

    # 2. Determine the final seed value
    if args.seed < 0:
        # User requested a random seed
        final_seed = int(time.time())
        print(f"--- Running with a RANDOM seed: {final_seed} ---")
    else:
        # User provided a specific seed or is using the default
        final_seed = args.seed
        print(f"--- Running with a FIXED seed: {final_seed} ---")

    # 3. Pass the final seed to the training function
    train_muzero(
        [main_config, create_config], 
        seed=final_seed, 
        max_env_step=max_env_step
    )
