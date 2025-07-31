import copy
from easydict import EasyDict

# IMPORTANT: Import your environment to access its class and version
from zoo.options_zero_game.envs.options_zero_game_env import OptionsZeroGameEnv

# ==============================================================
#                 Hyperparameters for Tuning
# ==============================================================
collector_env_num = 20
evaluator_env_num = 20
batch_size = 256
num_simulations = 100
update_per_collect = 1000
max_env_step = int(5e6)
reanalyze_ratio = 0.
n_episode = 20

# A comprehensive list of GARCH parameters for different market types.
# The 'mixed' price_source setting will randomly choose between these and historical data.
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
]

# ==============================================================
#                    Main Config (The Parameters)
# ==============================================================
options_zero_game_muzero_config = dict(
    exp_name=f'options_zero_game_muzero_v{OptionsZeroGameEnv.VERSION}_ns{num_simulations}_upc{update_per_collect}_bs{batch_size}',
    env=dict(
        env_id='OptionsZeroGame-v0',
        env_version=OptionsZeroGameEnv.VERSION,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        
        # --- Environment-Specific Parameters ---
        # Select the price data source: 'garch', 'historical', or 'mixed'
        price_source='mixed',
        historical_data_path='zoo/options_zero_game/data/market_data_cache',
        market_regimes=market_regimes,

        # Set ignore_legal_actions to True. This allows MuZero's MCTS to explore all actions,
        # while the environment itself will enforce the rules and apply penalties for illegal moves.
        ignore_legal_actions=True, 
        
        # Other gameplay parameters
        drawdown_penalty_weight=0.1,
        illegal_action_penalty=-1.0,
        rolling_vol_window=5,
    ),
    policy=dict(
        # The model, observation_shape, and action_space_size will be dynamically
        # configured in the __main__ block below.
        model=dict(
            model_type='mlp',
            lstm_hidden_size=512,
            latent_state_dim=512,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        # Path to a checkpoint file to resume training or for evaluation.
        # Leave commented out to start a fresh training run.
        # model_path='./path/to/your/ckpt_best.pth.tar',
        cuda=True,
        game_segment_length=OptionsZeroGameEnv.config['time_to_expiry_days'] * OptionsZeroGameEnv.config['steps_per_day'],
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        td_steps=10,
        discount_factor=0.999,
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=int(1e5),
        optim_type='AdamW',
        learning_rate=3e-4,
        weight_decay=1e-4,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
main_config = EasyDict(options_zero_game_muzero_config)

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

# =================================================================
#  Dynamic Configuration and Training Launcher
# =================================================================
if __name__ == '__main__':
    # This entry point is used when you run the script directly:
    # `python -u zoo/options_zero_game/config/options_zero_game_muzero_config.py`
    
    from lzero.entry import train_muzero
    
    # 1. Create a temporary environment instance to get the true observation and action shapes.
    #    This makes the config robust to any changes in the environment.
    temp_env_cfg = main_config.env 
    temp_env = OptionsZeroGameEnv(temp_env_cfg)
    obs_shape = temp_env.observation_space.shape
    act_size = temp_env.action_space.n
    
    # 2. Update the main configuration dictionary with the correct, dynamic values.
    main_config.policy.model.observation_shape = obs_shape
    main_config.policy.model.action_space_size = act_size
    
    # 3. Print a sanity check to the console to confirm the shapes before starting.
    print("="*50)
    print(">>> DYNAMICALLY CONFIGURED MODEL SHAPES <<<")
    print(f"    Observation Shape: {main_config.policy.model.observation_shape}")
    print(f"    Action Space Size: {main_config.policy.model.action_space_size}")
    print("="*50)
    
    # 4. Launch the training process.
    #    Ensure any old experiment directories are removed if you have changed the
    #    environment in a way that alters the model architecture.
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
