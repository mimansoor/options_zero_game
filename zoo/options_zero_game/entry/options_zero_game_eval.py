# zoo/options_zero_game/entry/options_zero_game_eval.py

import numpy as np
from lzero.entry import eval_muzero
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config, create_config
# Import environment files so the framework can find them in its registry.
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

if __name__ == "__main__":
    import argparse
    import time

    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Evaluate the Options-Zero-Game MuZero Agent.")
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

    # ==============================================================
    # 1. Set the path to your trained model checkpoint
    # ==============================================================
    # Make sure this path points to your actual trained model
    model_path = './best_ckpt/ckpt_best.pth.tar'

    # ==============================================================
    # 2. Configure the environment to use the LogReplayEnv wrapper
    # ==============================================================
    create_config.env.type = 'log_replay'
    main_config.env.log_file_path = 'zoo/options_zero_game/visualizer-ui/build/replay_log.json'

    # ==============================================================
    # 3. Configure the run for a single evaluation episode
    # ==============================================================
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    
    # CRITICAL: For evaluation, we use a 'base' environment manager, not the subprocess one.
    # This prevents hanging and is more efficient for single-threaded evaluation.
    create_config.env_manager.type = 'base'

    print("--- Starting Final Evaluation Run to Generate Replay Log ---")
    
    # ==============================================================
    # 4. Call the standard evaluator function
    # ==============================================================
    # This function will now succeed because the underlying environment bugs are fixed.
    returns_mean, returns = eval_muzero(
        [main_config, create_config],
        seed=final_seed,  # Use a fixed seed for reproducible evaluations
        num_episodes_each_seed=1,
        print_seed_details=True,
        model_path=model_path
    )
    
    print("\n--- Evaluation Complete ---")
    if returns:
        print(f"Final Episode PnL: {returns[0][0]:.2f}")
        print(f"The replay_log.json has been successfully saved in the visualizer-ui/build/ directory.")
    else:
        print("Evaluation finished, but no return values were recorded.")
