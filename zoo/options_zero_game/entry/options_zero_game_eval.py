# zoo/options_zero_game/entry/options_zero_game_eval.py
# <<< UPGRADED VERSION with --symbol and --seed arguments >>>

import numpy as np
import argparse
import time
import copy

from lzero.entry import eval_muzero
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config, create_config
# Import environment files so the framework can find them in its registry.
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

if __name__ == "__main__":
    # 1. Set up the argument parser to accept --seed and --symbol
    parser = argparse.ArgumentParser(description="Evaluate the Options-Zero-Game MuZero Agent and generate a replay log.")
    parser.add_argument(
        '--seed', 
        type=int, 
        default=0, 
        help="Master seed for the experiment. Defaults to 0 for reproducibility. Use a negative value for a random seed."
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help="Force evaluation on a specific historical symbol (e.g., 'SPY', 'TSLA'). Defaults to random."
    )
    args = parser.parse_args()

    # 2. Determine the final seed value
    if args.seed < 0:
        final_seed = int(time.time())
        print(f"--- Running with a RANDOM seed: {final_seed} ---")
    else:
        final_seed = args.seed
        print(f"--- Running with a FIXED seed: {final_seed} ---")

    # Create deep copies of the configs to avoid modifying the originals
    eval_main_config = copy.deepcopy(main_config)
    eval_create_config = copy.deepcopy(create_config)
    
    # ==============================================================
    # 3. Set the path to your trained model checkpoint
    # ==============================================================
    model_path = './best_ckpt/ckpt_best.pth.tar'

    # ==============================================================
    # 4. Configure the environment for this specific evaluation run
    # ==============================================================
    eval_create_config.env.type = 'log_replay'
    eval_main_config.env.log_file_path = 'zoo/options_zero_game/visualizer-ui/build/replay_log.json'
    
    eval_main_config.env.evaluator_env_num = 1
    eval_main_config.env.n_evaluator_episode = 1
    eval_create_config.env_manager.type = 'base'

    # Apply the forced symbol from the command line, if provided
    if args.symbol:
        print(f"--- Forcing evaluation on historical symbol: {args.symbol} ---")
        eval_main_config.env.forced_historical_symbol = args.symbol
    else:
        print("--- Using a random historical symbol for evaluation ---")

    print("--- Starting Final Evaluation Run to Generate Replay Log ---")
    
    # ==============================================================
    # 5. Call the standard evaluator function with the modified configs
    # ==============================================================
    returns_mean, returns = eval_muzero(
        [eval_main_config, eval_create_config],
        seed=final_seed,
        num_episodes_each_seed=1,
        print_seed_details=True,
        model_path=model_path
    )
    
    print("\n--- Evaluation Complete ---")
    if returns and returns[0]:
        print(f"Final Episode PnL: {returns[0][0]:.2f}")
        print(f"The replay_log.json has been successfully saved in the visualizer-ui/build/ directory.")
    else:
        print("Evaluation finished, but no return values were recorded.")
