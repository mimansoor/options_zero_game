import copy
from easydict import EasyDict

# Import the environment to get its version and default parameters
from zoo.options_zero_game.envs.options_zero_game_env import OptionsZeroGameEnv

# ==============================================================
#                 Static Parameters
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 4
batch_size = 256
num_simulations = 25
update_per_collect = 1000
replay_ratio = 0.25
max_env_step = int(5e7)
reanalyze_ratio = 0.

market_regimes = [
    # Name, mu, omega, alpha, beta, overnight_vol_multiplier
    {'name': 'Bond_Markets', 'mu': 0.00001, 'omega': 0.000002, 'alpha': 0.05, 'beta': 0.92, 'overnight_vol_multiplier': 1.1},
    {'name': 'Utilities_Sector', 'mu': 0.00004, 'omega': 0.00001, 'alpha': 0.08, 'beta': 0.90, 'overnight_vol_multiplier': 1.2},
    {'name': 'Developed_Markets', 'mu': 0.00006, 'omega': 0.000005, 'alpha': 0.08, 'beta': 0.90, 'overnight_vol_multiplier': 1.3},
    {'name': 'Foreign_Exchange_FX', 'mu': 0.0000, 'omega': 0.000003, 'alpha': 0.08, 'beta': 0.91, 'overnight_vol_multiplier': 1.4},
    {'name': 'Tech_Sector', 'mu': 0.0001, 'omega': 0.00004, 'alpha': 0.11, 'beta': 0.87, 'overnight_vol_multiplier': 1.5},
    {'name': 'Individual_Stocks', 'mu': 0.00008, 'omega': 0.00007, 'alpha': 0.10, 'beta': 0.88, 'overnight_vol_multiplier': 1.6},
    {'name': 'Emerging_Markets', 'mu': 0.0001, 'omega': 0.00005, 'alpha': 0.12, 'beta': 0.86, 'overnight_vol_multiplier': 1.7},
    {'name': 'Bull_LowVol', 'mu': 0.0005, 'omega': 0.000002, 'alpha': 0.05, 'beta': 0.94, 'overnight_vol_multiplier': 1.4},
    {'name': 'Crisis_HighVol', 'mu': -0.0005, 'omega': 0.0001, 'alpha': 0.15, 'beta': 0.82, 'overnight_vol_multiplier': 2.2},
    {'name': 'Frontier_Markets', 'mu': 0.0002, 'omega': 0.0001, 'alpha': 0.20, 'beta': 0.70, 'overnight_vol_multiplier': 1.8},
    {'name': 'Commodities_Oil', 'mu': 0.0000, 'omega': 0.0002, 'alpha': 0.28, 'beta': 0.70, 'overnight_vol_multiplier': 1.9},
    {'name': 'Volatility_VIX', 'mu': 0.0, 'omega': 0.0005, 'alpha': 0.25, 'beta': 0.65, 'overnight_vol_multiplier': 2.0},
    {'name': 'Cryptocurrencies', 'mu': 0.001, 'omega': 0.001, 'alpha': 0.20, 'beta': 0.75, 'overnight_vol_multiplier': 2.5},
    {'name': 'TSLA_Real', 'mu': 0.001070, 'omega': 0.00003585, 'alpha': 0.0350, 'beta': 0.9418, 'overnight_vol_multiplier': 6.58},
    {'name': 'SPY_Real', 'mu': 0.000905, 'omega': 0.00000397, 'alpha': 0.1280, 'beta': 0.8404, 'overnight_vol_multiplier': 5.99},
    {'name': 'RELIANCE.NS_Real', 'mu': 0.000352, 'omega': 0.00000267, 'alpha': 0.0207, 'beta': 0.9666, 'overnight_vol_multiplier': 5.14},
    {'name': '^NSEI_Real', 'mu': 0.000703, 'omega': 0.00000308, 'alpha': 0.0928, 'beta': 0.8733, 'overnight_vol_multiplier': 5.40},
    {'name': 'TCS.NS_Real', 'mu': 0.000261, 'omega': 0.00002354, 'alpha': 0.0378, 'beta': 0.8327, 'overnight_vol_multiplier': 5.53},
    {'name': 'ABB.NS_Real', 'mu': 0.001221, 'omega': 0.00025891, 'alpha': 0.1361, 'beta': 0.2717, 'overnight_vol_multiplier': 3.37},
    {'name': 'BTC-USD_Real', 'mu': 0.001557, 'omega': 0.00002610, 'alpha': 0.0662, 'beta': 0.9083, 'overnight_vol_multiplier': 1.85},
    {'name': 'INTC_Real', 'mu': -0.000764, 'omega': 0.00033938, 'alpha': 0.3683, 'beta': 0.2445, 'overnight_vol_multiplier': 6.33},
    {'name': '^DJI_Real', 'mu': 0.000561, 'omega': 0.00000490, 'alpha': 0.1261, 'beta': 0.8208, 'overnight_vol_multiplier': 6.04},
    {'name': '^IXIC_Real', 'mu': 0.000983, 'omega': 0.00000426, 'alpha': 0.1053, 'beta': 0.8768, 'overnight_vol_multiplier': 6.14},
    {'name': '^RUT_Real', 'mu': 0.000419, 'omega': 0.00001442, 'alpha': 0.0707, 'beta': 0.8612, 'overnight_vol_multiplier': 6.70},
    {'name': 'GC=F_Real', 'mu': 0.000309, 'omega': 0.00000454, 'alpha': 0.0361, 'beta': 0.9160, 'overnight_vol_multiplier': 2.02},
    {'name': 'TLT_Real', 'mu': -0.000399, 'omega': 0.00000113, 'alpha': 0.0301, 'beta': 0.9592, 'overnight_vol_multiplier': 6.58},
    {'name': 'BND_Real', 'mu': -0.000033, 'omega': 0.00000004, 'alpha': 0.0292, 'beta': 0.9687, 'overnight_vol_multiplier': 6.87},
    {'name': 'ETH-USD_Real', 'mu': 0.001368, 'omega': 0.00002192, 'alpha': 0.0539, 'beta': 0.9345, 'overnight_vol_multiplier': 1.40},
    {'name': 'SOL-USD_Real', 'mu': 0.001677, 'omega': 0.00013563, 'alpha': 0.1267, 'beta': 0.8455, 'overnight_vol_multiplier': 1.10},
]

# ==============================================================
#                 Strategy ID Mapping
# ==============================================================
# The single, complete source of truth for all strategy names and their IDs.
strategy_name_to_id = {
    # Base strategies (for single legs)
    'LONG_CALL': 0, 'SHORT_CALL': 1, 'LONG_PUT': 2, 'SHORT_PUT': 3,

    # Complex Strategies
    'LONG_STRADDLE': 4, 'SHORT_STRADDLE': 5,
    'LONG_IRON_FLY': 10, 'SHORT_IRON_FLY': 11,
    'LONG_IRON_CONDOR': 12, 'SHORT_IRON_CONDOR': 13,

    # Strategies with Width Variations
    'LONG_STRANGLE_1': 6, 'SHORT_STRANGLE_1': 7,
    'LONG_STRANGLE_2': 8, 'SHORT_STRANGLE_2': 9,

    'LONG_VERTICAL_CALL_1': 14, 'SHORT_VERTICAL_CALL_1': 15,
    'LONG_VERTICAL_CALL_2': 16, 'SHORT_VERTICAL_CALL_2': 17,
    'LONG_VERTICAL_PUT_1': 18, 'SHORT_VERTICAL_PUT_1': 19,
    'LONG_VERTICAL_PUT_2': 20, 'SHORT_VERTICAL_PUT_2': 21,

    'LONG_CALL_FLY_1': 22, 'SHORT_CALL_FLY_1': 23,
    'LONG_PUT_FLY_1': 24, 'SHORT_PUT_FLY_1': 25,
    'LONG_CALL_FLY_2': 26, 'SHORT_CALL_FLY_2': 27,
    'LONG_PUT_FLY_2': 28, 'SHORT_PUT_FLY_2': 29,
}

# Dynamically add the new delta-based strangle strategy IDs
next_id = max(strategy_name_to_id.values()) + 1
for delta in range(15, 31, 5):
    strategy_name_to_id[f'LONG_STRANGLE_DELTA_{delta}'] = next_id
    next_id += 1
    strategy_name_to_id[f'SHORT_STRANGLE_DELTA_{delta}'] = next_id
    next_id += 1

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

    # === Phase 3: Introduction to Risk Management (Credit Spreads) ===
    # Goal: Learn to hedge a short put by converting it into a Bull Put Spread.
    int(4e6): 'OPEN_SHORT_VERTICAL_PUT_1',

    # === Phase 4: Counterpart Risk Management ===
    # Goal: Learn to hedge a short call by converting it into a Bear Call Spread.
    int(6e6): 'OPEN_SHORT_VERTICAL_CALL_1',

    # === Phase 5: Selling Volatility (Undefined Risk) ===
    # Goal: Learn a non-directional strategy by selling ATM volatility (Straddles).
    int(8e6): 'OPEN_SHORT_STRADDLE_ATM',

    # === Phase 6: Selling Volatility (Risk Defined) ===
    # Goal: Learn to create a range-bound, positive theta position with defined risk.
    int(10e6): 'OPEN_SHORT_IRON_CONDOR',

    # === Phase 7: Advanced Range-Bound Strategy ===
    # Goal: Learn a different range-bound profile with a sharper profit peak (Butterflies).
    int(12e6): 'OPEN_SHORT_CALL_FLY_1',

    # === Phase 8: Advanced Volatility Selling (Delta-based) ===
    # Goal: Learn the professional method of entering strangles based on delta, not fixed widths.
    int(14e6): 'OPEN_SHORT_STRANGLE_DELTA_20',
    
    # === Phase 9: Learning to BUY Options (Debit Strategies) ===
    # Goal: Change mindset. Learn to pay theta for a large directional move (Long Calls).
    int(16e6): 'OPEN_LONG_CALL_ATM+0',

    # === Phase 10: Learning to BUY Spreads ===
    # Goal: Learn to make a risk-defined directional bet by buying a Bear Put Spread.
    int(18e6): 'OPEN_LONG_VERTICAL_PUT_1',

    # === Final Phase: Integration and Agent Autonomy ===
    # Goal: Allow the agent to use any of its learned strategies to maximize reward.
    int(20e6): 'ALL'
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
        market_regimes=market_regimes,
        strategy_name_to_id=strategy_name_to_id,
        illegal_action_penalty=-1.0,
        rolling_vol_window=5,
        # The absolute IV percentage points to add to the expert's prediction.
        # This represents the market's inherent "volatility risk premium".
        # A value of 0.05 means we add 5% to the predicted IV.
        volatility_premium_abs=0.05,
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
        learning_rate=3e-3,
        weight_decay=1e-4,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e6),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,

        learn=dict(
            learner=dict(
                hook=dict(
                    # Provide the full path to the checkpoint you want to resume from
                    load_ckpt_before_run='./best_ckpt/ckpt_best.pth.tar'
                )
            )
        ),
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
