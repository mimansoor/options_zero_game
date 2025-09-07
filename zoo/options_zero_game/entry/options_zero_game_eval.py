import argparse
import copy
import json
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from lzero.entry import eval_muzero
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config, create_config
# Import environment files so the framework can find them in its registry.
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Options-Zero-Game MuZero Agent.")
    
    # --- Core Evaluation Arguments ---
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='./best_ckpt/ckpt_best.pth.tar', 
        help="Path to the model checkpoint (.pth.tar) to evaluate."
    )
    parser.add_argument(
        '-n', '--episodes', 
        type=int, 
        default=1, 
        help="Number of episodes to run for evaluation."
    )
    
    # --- Override Arguments ---
    parser.add_argument('--seed', type=int, default=-1, help="Master seed for the experiment. Use a negative value for a random time-based seed.")
    parser.add_argument('--symbol', type=str, default=None, help="Force evaluation on a specific historical symbol (e.g., 'SPY').")
    parser.add_argument('--strategy', type=str, default=None, help="Force a specific opening strategy for the replay (bypasses agent's choice).")
    parser.add_argument('--days', type=int, default=0, help="Force a specific episode length in days for the evaluation.")
    parser.add_argument('--agents_choice', action='store_true', help="Let the agent choose its own opening intent (curriculum disabled).")
    parser.add_argument('--portfolio_setup_file', type=str, default=None, help="Path to a JSON file defining the portfolio to set up at Step 0.")
    parser.add_argument('--profit_target_pct', type=float, default=None, help="Override the global profit target percentage (e.g., 3 for 3%).")
    parser.add_argument('--credit_tp_pct', type=float, default=None, help="Override the take-profit percentage for credit strategies (e.g., 70 for 70%).")
    parser.add_argument('--debit_tp_mult', type=float, default=None, help="Override the take-profit multiple for debit strategies (e.g., 2 for 2x).")
    parser.add_argument('--days_back', type=int, default=None, help="Force historical evaluation to start a specific number of days from the LATEST possible date.")
    parser.add_argument( '--deterministic', action='store_true', help="Run in a fully deterministic mode (slower, but reproducible).")
    
    args = parser.parse_args()

    master_seed = int(time.time()) if args.seed < 0 else args.seed
    
    print(f"--- Starting Evaluation Run ---")
    print(f"  - Model Path: {args.model_path}")
    print(f"  - Episodes:   {args.episodes}")
    print(f"  - Master Seed: {master_seed}")
    
    if not os.path.exists(args.model_path):
        print(f"\nFATAL ERROR: Model checkpoint not found at '{args.model_path}'.")
        exit(1)

    # This list will store the simple, clean PnL from each individual run.
    all_episode_pnls = []

    # Create a master tqdm progress bar for our manual loop.
    episode_iterator = tqdm(range(args.episodes), desc="Evaluating Episodes", ncols=120)
    print(f"episode_iterator: {episode_iterator}")

    for i in episode_iterator:
        # Create a fresh config for each run to ensure perfect isolation.
        eval_main_config = copy.deepcopy(main_config)
        eval_create_config = copy.deepcopy(create_config)
        
        # Use a unique, deterministic seed for each episode.
        current_seed = master_seed + i

        # Configure the evaluator to run for ONLY ONE episode.
        eval_main_config.env.n_evaluator_episode = 1
        eval_main_config.env.evaluator_env_num = 1
        eval_create_config.env_manager.type = 'base' # Use the simple, non-parallel manager

        # Only generate the detailed replay log for the very first episode of a single-episode run.
        if i == 0 and args.episodes == 1:
            eval_create_config.env.type = 'log_replay'
            eval_main_config.env.log_file_path = 'zoo/options_zero_game/visualizer-ui/build/replay_log.json'
        else:
            eval_create_config.env.type = 'options_zero_game'
        
        # --- Apply all command-line overrides for this specific run ---
        eval_main_config.exp_name = 'eval/evaluator'

        if args.deterministic:
            print("--- RUNNING IN DETERMINISTIC MODE ---")
            # 1. Set all PyTorch and CUDA seeds and flags for full reproducibility.
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # 2. Force the evaluator to use the pure Python MCTS implementation.
            if 'policy' not in eval_main_config:
                eval_main_config.policy = {}
            eval_main_config.policy.mcts_ctree = False

        eval_main_config.env.is_eval_mode = True

        if args.days_back is not None and args.days_back > 0:
            eval_main_config.env.forced_days_back = args.days_back
        if args.profit_target_pct is not None:
            eval_main_config.env.profit_target_pct = args.profit_target_pct
        if args.credit_tp_pct is not None:
            eval_main_config.env.credit_strategy_take_profit_pct = args.credit_tp_pct
        if args.debit_tp_mult is not None:
            eval_main_config.env.debit_strategy_take_profit_multiple = args.debit_tp_mult
        if args.portfolio_setup_file:
            # Full logic for portfolio setup file is preserved
            try:
                if not os.path.exists(args.portfolio_setup_file): raise FileNotFoundError(f"File not found: {args.portfolio_setup_file}")
                with open(args.portfolio_setup_file, 'r') as f: setup_data = json.load(f)
                if not isinstance(setup_data, dict) or 'legs' not in setup_data: raise ValueError("JSON must be an object with a 'legs' key.")
                eval_main_config.env.forced_initial_portfolio = setup_data['legs']
                if 'atm_iv' in setup_data: eval_main_config.env.forced_atm_iv = float(setup_data['atm_iv'])
            except Exception as e:
                print(f"FATAL: Could not process portfolio setup file. Error: {e}")
                exit(1)
        elif args.strategy:
            eval_main_config.env.forced_opening_strategy_name = args.strategy
        elif args.agents_choice:
            eval_main_config.env.disable_opening_curriculum = True
        if args.symbol:
            eval_main_config.env.forced_historical_symbol = args.symbol
            eval_main_config.env.price_source = 'historical'
        if args.days > 0:
            eval_main_config.env.forced_episode_length = args.days

        # Call the evaluator for a single, clean run.
        _, returns = eval_muzero(
            [eval_main_config, eval_create_config],
            seed=current_seed,
            num_episodes_each_seed=1,
            print_seed_details=False,
            model_path=args.model_path
        )
        
        # Append the clean, simple result to our list.
        if returns and returns[0] is not None and len(returns[0]) > 0:
            pnl = returns[0][0]
            all_episode_pnls.append(pnl)
            # Update the progress bar with the latest result
            episode_iterator.set_description(f"Last PnL: ${pnl:,.2f}")

    # Perform final calculations on our simple, reliable list of PnLs.
    print("\n--- Evaluation Complete ---")
    if all_episode_pnls:
        print(f"  - Episodes Run:  {len(all_episode_pnls)}")
        print(f"  - Average PnL:   ${np.mean(all_episode_pnls):,.2f}")
        print(f"  - Std Dev of PnL: ${np.std(all_episode_pnls):,.2f}")
        print(f"  - Max PnL:       ${np.max(all_episode_pnls):,.2f}")
        print(f"  - Min PnL:       ${np.min(all_episode_pnls):,.2f}")
        
        if args.episodes == 1:
            print(f"\nThe replay_log.json has been successfully saved.")

        # We only generate a histogram if there are enough data points for it to be meaningful.
        if len(all_episode_pnls) > 1:
            try:
                print("\n--- Generating PnL Distribution Histogram ---")
                output_dir = "zoo/options_zero_game/visualizer-ui/build"
                output_filename = os.path.join(output_dir, 'pnl_histogram.png')

                # Ensure the directory exists
                os.makedirs(output_dir, exist_ok=True)

                plt.figure(figsize=(12, 7))
                plt.hist(all_episode_pnls, bins=20, edgecolor='black', alpha=0.8, color='#2a9d8f')
                mean_pnl = np.mean(all_episode_pnls)
                plt.axvline(mean_pnl, color='#e76f51', linestyle='--', linewidth=2, label=f'Mean PnL: ${mean_pnl:,.2f}')
                plt.title(f'PnL Distribution over {len(all_episode_pnls)} Episodes', fontsize=16)
                plt.xlabel('Final PnL per Episode ($)', fontsize=12)
                plt.ylabel('Frequency (Number of Episodes)', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.legend()
                
                plt.savefig(output_filename)
                plt.close()
                
                print(f"Successfully saved histogram to '{os.path.abspath(output_filename)}'")
            except Exception as e:
                print(f"\nWarning: Could not generate histogram. Error: {e}")

    else:
        print("Evaluation finished, but no valid return values were recorded.")

