import numpy as np
import argparse
import time
import copy
import json
import os
import torch

from lzero.entry import eval_muzero
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config, create_config
# Import environment files so the framework can find them in its registry.
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Options-Zero-Game MuZero Agent and generate a replay log.")
    parser.add_argument('--seed', type=int, default=-1, help="Master seed for the experiment. Use -1 for a random time-based seed.")
    parser.add_argument('--symbol', type=str, default=None, help="Force evaluation on a specific historical symbol (e.g., 'SPY').")
    parser.add_argument('--strategy', type=str, default=None, help="Force a specific opening strategy for the replay.")
    parser.add_argument('--days', type=int, default=0, help="Force a specific episode length in days for the evaluation.")
    parser.add_argument('--agents_choice', action='store_true', help="Let the agent choose its own opening move (disables curriculum).")
    parser.add_argument(
        '--portfolio_setup_file', 
        type=str, 
        default=None, 
        help="Path to a JSON file defining the portfolio to set up at Step 0."
    )

    # <<< --- NEW: Add the override arguments --- >>>
    parser.add_argument('--profit_target_pct', type=float, default=None, help="Override the global profit target percentage (e.g., 3 for 3%).")
    parser.add_argument('--credit_tp_pct', type=float, default=None, help="Override the take-profit percentage for credit strategies (e.g., 70 for 70%).")
    parser.add_argument('--debit_tp_mult', type=float, default=None, help="Override the take-profit multiple for debit strategies (e.g., 2 for 2x).")
    parser.add_argument(
        '--days_back', 
        type=int, 
        default=None, 
        help="Force the historical evaluation to start a specific number of days from the LATEST possible date. "
             "E.g., --days_back 1 starts on the last possible day, --days_back 25 starts 24 days before that."
    )
    
    args = parser.parse_args()

    final_seed = int(time.time()) if args.seed < 0 else args.seed
    print(f"--- Running with seed: {final_seed} ---")

    # Set all seeds and determinism flags for full reproducibility.
    #if args.seed >= 0:
        #print("--- ENABLING FULL DETERMINISM FOR PYTORCH ---")
        #torch.manual_seed(final_seed)
        #if torch.cuda.is_available():
            #torch.cuda.manual_seed_all(final_seed)
        ## These two lines are crucial for forcing cuDNN to use deterministic algorithms.
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

    # Create a deep copy of the configs to avoid modifying the global objects
    eval_main_config = copy.deepcopy(main_config)
    eval_create_config = copy.deepcopy(create_config)
    
    model_path = './best_ckpt/ckpt_best.pth.tar'

    eval_main_config.exp_name = 'eval/evaluator'

    # Force the evaluator to use the pure Python MCTS implementation, which
    # respects the Python-level seeds, rather than the non-deterministic C++ version.
    # We must check if the policy key exists before trying to modify it.
    if 'policy' not in eval_main_config:
        eval_main_config.policy = {}
    eval_main_config.policy.mcts_ctree = False
    print("--- MCTS OVERRIDE: Forcing Python MCTS for deterministic evaluation. ---")

    # --- Standard Evaluation Setup ---
    eval_create_config.env.type = 'log_replay'
    eval_main_config.env.log_file_path = 'zoo/options_zero_game/visualizer-ui/build/replay_log.json'
    eval_main_config.env.evaluator_env_num = 1
    eval_main_config.env.n_evaluator_episode = 1
    eval_create_config.env_manager.type = 'base'
    eval_main_config.env.is_eval_mode = True

    if args.days_back is not None:
        if args.days_back > 0:
            eval_main_config.env.forced_days_back = args.days_back
            print(f"--- OVERRIDE: Forcing historical start date to {args.days_back} day(s) from the end of the dataset. ---")
        else:
            print("(WARNING) --days_back must be a positive integer. Ignoring.")

    # <<< --- NEW: Apply the overrides to the config --- >>>
    if args.profit_target_pct is not None:
        eval_main_config.env.profit_target_pct = args.profit_target_pct
        print(f"--- OVERRIDE: Setting Profit Target to {args.profit_target_pct}% ---")
    if args.credit_tp_pct is not None:
        eval_main_config.env.credit_strategy_take_profit_pct = args.credit_tp_pct
        print(f"--- OVERRIDE: Setting Credit Take-Profit to {args.credit_tp_pct}% ---")
    if args.debit_tp_mult is not None:
        eval_main_config.env.debit_strategy_take_profit_multiple = args.debit_tp_mult
        print(f"--- OVERRIDE: Setting Debit Take-Profit to {args.debit_tp_mult}x ---")

    if args.portfolio_setup_file:
        print(f"--- Custom Portfolio Setup Detected from file: {args.portfolio_setup_file} ---")
        try:
            if not os.path.exists(args.portfolio_setup_file):
                raise FileNotFoundError(f"The specified file was not found: {args.portfolio_setup_file}")

            with open(args.portfolio_setup_file, 'r') as f:
                setup_data = json.load(f)
            
            # 1. Validate the new object-based format.
            if not isinstance(setup_data, dict) or 'legs' not in setup_data:
                raise ValueError("JSON file must be an object with a 'legs' key.")
            if not isinstance(setup_data['legs'], list):
                raise ValueError("The 'legs' key must contain a list of leg dictionaries.")
            
            # 2. Inject the portfolio definition from the 'legs' key.
            eval_main_config.env.forced_initial_portfolio = setup_data['legs']
            print(f"Successfully parsed and will set up {len(setup_data['legs'])} leg(s) at Step 0.")
            
            # 3. Inject the optional ATM IV override if it exists.
            if 'atm_iv' in setup_data:
                forced_iv = float(setup_data['atm_iv'])
                if forced_iv > 0:
                    eval_main_config.env.forced_atm_iv = forced_iv
                    print(f"Successfully parsed and will force ATM IV to {forced_iv}%.")

            if args.strategy:
                print("(INFO) --portfolio_setup_file overrides --strategy.")
                eval_main_config.env.forced_opening_strategy_name = None

        except (json.JSONDecodeError, FileNotFoundError, ValueError, TypeError) as e:
            print(f"FATAL: Could not process portfolio setup file. Error: {e}")
            exit(1)

    # --- Logic to control the opening move ---
    # This logic now correctly respects that a forced portfolio might already be set.
    if args.strategy and not args.portfolio_setup_file:
        print(f"--- Forcing opening strategy: {args.strategy} ---")
        eval_main_config.env.forced_opening_strategy_name = args.strategy
    elif args.agents_choice:
        print("--- Letting agent choose its opening move (curriculum disabled) ---")
        eval_main_config.env.disable_opening_curriculum = True
    else:
        # If no strategy is forced and it's not agent's choice, we don't need to print anything.
        # The environment will use its default curriculum logic.
        print("--- Using random curriculum for opening move ---")
        pass

    if args.symbol:
        print(f"--- Forcing evaluation on historical symbol: {args.symbol} ---")
        eval_main_config.env.forced_historical_symbol = args.symbol
        print(f"--- Setting price_source to 'historical' due to --symbol flag ---")
        eval_main_config.env.price_source = 'historical'

    if args.days > 0:
        print(f"--- Forcing episode length to {args.days} days ---")
        eval_main_config.env.forced_episode_length = args.days

    print("\n--- Starting Final Evaluation Run to Generate Replay Log ---")
   
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
