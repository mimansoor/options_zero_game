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
#           UNIFIED REGIME DEFINITIONS (More Realistic Version)
# ==============================================================
# This revised set of regimes is tuned to more common VIX levels for
# a standard equity index, providing a more realistic training ground.
UNIFIED_REGIMES = [
    {
        'name': 'Crisis (High Vol, Bearish)',
        # Represents a VIX in the 40-50 range. A significant market downturn.
        'mu': -0.0005, 'omega': 0.0001, 'alpha': 0.15, 'beta': 0.82, 'overnight_vol_multiplier': 2.2,
        'atm_iv': 45.0, 'far_otm_put_iv': 75.0, 'far_otm_call_iv': 38.0,
    },
    {
        'name': 'Elevated Vol (Bullish)',
        # Represents a VIX in the 20-30 range. The market is trending up but with some uncertainty.
        'mu': 0.000905, 'omega': 0.00000397, 'alpha': 0.1280, 'beta': 0.8404, 'overnight_vol_multiplier': 5.99,
        'atm_iv': 22.0, 'far_otm_put_iv': 40.0, 'far_otm_call_iv': 18.0,
    },
    {
        'name': 'Complacent (Low Vol, Neutral)',
        # Represents a VIX in the low teens. A quiet, range-bound market.
        'mu': 0.00001, 'omega': 0.000002, 'alpha': 0.05, 'beta': 0.92, 'overnight_vol_multiplier': 1.1,
        'atm_iv': 14.0, 'far_otm_put_iv': 22.0, 'far_otm_call_iv': 12.0,
    },
    {
        'name': 'Volatile (Choppy, Neutral)',
        # Represents a VIX in the 30s. High uncertainty, but no clear market direction (choppy).
        # This replaces the extreme "Crypto" regime with a more common scenario.
        'mu': 0.0001, 'omega': 0.00005, 'alpha': 0.13, 'beta': 0.85, 'overnight_vol_multiplier': 1.85,
        'atm_iv': 30.0, 'far_otm_put_iv': 55.0, 'far_otm_call_iv': 25.0,
    },
]

# ==============================================================
#                 Curriculum Schedule
# ==============================================================
# A structured training plan that teaches the agent concepts in phases.
# Each phase focuses on a core strategy for 2 million steps before moving to the next.
TRAINING_CURRICULUM = {
    # Phase 1: Learn how to be Bullish (4M steps)
    0: 'OPEN_BULLISH_POSITION',

    # Phase 2: Learn how to be Bearish (4M steps)
    int(4e6): 'OPEN_BEARISH_POSITION',

    # Phase 3: Learn how to be Neutral / Sell Volatility (4M steps)
    int(8e6): 'OPEN_NEUTRAL_POSITION',

    # Final Phase: Full Autonomy - Agent can choose any intent.
    int(12e6): 'ALL'
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
MAX_STRIKE_OFFSET = 50
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

# Modify the loop to only map the OPEN names for SHORT naked legs.
# for direction in ['LONG', 'SHORT']:
#     for opt_type in ['CALL', 'PUT']:
for opt_type in ['CALL', 'PUT']: # Only loop through option types
    direction = 'SHORT' # Hardcode the direction
    action_name = f'OPEN_{direction}_{opt_type}_ATM'
    internal_name = f'{direction}_{opt_type}'
    strategy_name_to_id[action_name] = strategy_name_to_id[internal_name]

# --- 2. Core Volatility Strategies ---
internal_name = 'SHORT_STRADDLE' # Only create SHORT
strategy_name_to_id[internal_name] = next_id; next_id += 1
strategy_name_to_id[f'OPEN_{internal_name}'] = strategy_name_to_id[internal_name]

# The other core strategies do not have the suffix
for name in ['IRON_CONDOR', 'IRON_FLY']:
    internal_name = f'SHORT_{name}' # Only create SHORT
    strategy_name_to_id[internal_name] = next_id; next_id += 1
    strategy_name_to_id[f'OPEN_{internal_name}'] = strategy_name_to_id[internal_name]

# --- 3. Spreads ---
for direction in ['BULL', 'BEAR']:
    for opt_type in ['CALL', 'PUT']:
        internal_name = f'{direction}_{opt_type}_SPREAD'
        strategy_name_to_id[internal_name] = next_id; next_id += 1
        strategy_name_to_id[f'OPEN_{internal_name}'] = strategy_name_to_id[internal_name]

# --- 4. Strategies with Variations (Strangles, Butterflies) ---
for delta in [25, 30]:
    internal_name = f'SHORT_STRANGLE_DELTA_{delta}' # Only create SHORT
    strategy_name_to_id[internal_name] = next_id; next_id += 1
    strategy_name_to_id[f'OPEN_{internal_name}'] = strategy_name_to_id[internal_name]

# <<< --- NEW: Section for Advanced Multi-Leg Strategies --- >>>
strategy_name_to_id['JADE_LIZARD'] = next_id; next_id += 1
strategy_name_to_id['OPEN_JADE_LIZARD'] = strategy_name_to_id['JADE_LIZARD']

strategy_name_to_id['REVERSE_JADE_LIZARD'] = next_id; next_id += 1
strategy_name_to_id['OPEN_REVERSE_JADE_LIZARD'] = strategy_name_to_id['REVERSE_JADE_LIZARD']

strategy_name_to_id['BIG_LIZARD'] = next_id; next_id += 1
strategy_name_to_id['OPEN_BIG_LIZARD'] = strategy_name_to_id['BIG_LIZARD']

strategy_name_to_id['REVERSE_BIG_LIZARD'] = next_id; next_id += 1
strategy_name_to_id['OPEN_REVERSE_BIG_LIZARD'] = strategy_name_to_id['REVERSE_BIG_LIZARD']

strategy_name_to_id['PUT_RATIO_SPREAD'] = next_id; next_id += 1
strategy_name_to_id['OPEN_PUT_RATIO_SPREAD'] = strategy_name_to_id['PUT_RATIO_SPREAD']

strategy_name_to_id['CALL_RATIO_SPREAD'] = next_id; next_id += 1
strategy_name_to_id['OPEN_CALL_RATIO_SPREAD'] = strategy_name_to_id['CALL_RATIO_SPREAD']

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
        disable_opening_curriculum=True,
        
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
        price_source='historical',
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

        # The target net debit for a butterfly as a percentage of the ATM strike.
        # 0.01 means we are trying to pay ~1% of the stock price for the butterfly.
        butterfly_target_cost_pct=0.01, 

        max_strike_offset=MAX_STRIKE_OFFSET,
        
        # This MUST match the 'sequence_length' used in the expert trainer CONFIG.
        expert_sequence_length=60,

        # <<< --- NEW: Parameter to control the inverse price/IV correlation --- >>>
        # A negative value means IV goes up when price goes down.
        # This value controls the strength of the effect.
        iv_price_correlation_strength=-0.10,

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
