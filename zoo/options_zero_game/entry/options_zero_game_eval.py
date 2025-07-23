import numpy as np

# Import the official evaluation entry point
from lzero.entry import eval_muzero

# Import our game's configuration files
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config, create_config

# <<< THE FIX: We must also import our environment files so the registry can find them.
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

if __name__ == "__main__":
    # ==============================================================
    # 1. Set the path to your best trained model checkpoint
    # ==============================================================
    model_path = './options_zero_game_muzero_global_markets_ns50_upc1000_bs128/ckpt/ckpt_best.pth.tar' # UPDATE THIS PATH

    # ==============================================================
    # 2. Configure the environment for logging
    # ==============================================================
    
    # Tell the create_config to build our 'log_replay' environment instead of the default one
    create_config.env.type = 'log_replay'
    
    # Set the path for the output log file
    main_config.env.log_file_path = 'visualizer-ui/public/replay_log.json'

    # We can choose a specific market regime to evaluate for consistency
    main_config.env.market_regimes = [
        {'name': 'Developed_Markets', 'mu': 0.00006, 'omega': 0.000005, 'alpha': 0.08, 'beta': 0.90}
    ]

    # ==============================================================
    # 3. Configure the evaluation run for a single episode
    # ==============================================================
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    create_config.env_manager.type = 'base'

    print("Starting evaluation run to generate replay log...")
    
    # Call the official eval_muzero function
    returns_mean, returns = eval_muzero(
        [main_config, create_config],
        seed=0,
        num_episodes_each_seed=1,
        print_seed_details=True,
        model_path=model_path
    )
    
    print(f"\nEvaluation run complete. Final Episode PnL: {returns[0][0]:.2f}")
    print(f"The replay_log.json should now be in the visualizer-ui/public/ directory.")
