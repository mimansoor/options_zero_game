# zoo/options_zero_game/entry/strategy_analyzer.py
# <<< FINAL VERSION WITH AUTOMATED JSON REPORTING >>>

import argparse
import copy
import json
import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from lzero.entry import eval_muzero
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config, create_config
# Import environment files so the framework can find them
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

def get_valid_strategies() -> list:
    """
    Helper function to get a list of all valid opening strategies.
    This corrected version properly initializes the environment with a config.
    """
    temp_env_cfg = copy.deepcopy(main_config.env)
    temp_env = zoo.options_zero_game.envs.options_zero_game_env.OptionsZeroGameEnv(cfg=temp_env_cfg)
    return [name for name in temp_env.actions_to_indices if name.startswith('OPEN_')]

def calculate_statistics(results: list, strategy_name: str) -> dict:
    """
    Calculates a full statistical summary and returns it as a dictionary.
    """
    if not results: return {}

    pnls = [r['pnl'] for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = (num_wins / len(pnls)) * 100 if pnls else 0.0
    avg_profit = sum(wins) / num_wins if num_wins > 0 else 0.0
    avg_loss = sum(losses) / num_losses if num_losses > 0 else 0.0
    
    expectancy = ((win_rate / 100) * avg_profit) + ((1 - win_rate / 100) * abs(avg_loss))
    profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float('inf')

    max_win_streak, max_loss_streak, current_win_streak, current_loss_streak = 0, 0, 0, 0
    for pnl in pnls:
        if pnl > 0: current_win_streak += 1; current_loss_streak = 0
        else: current_loss_streak += 1; current_win_streak = 0
        max_win_streak = max(max_win_streak, current_win_streak)
        max_loss_streak = max(max_loss_streak, current_loss_streak)

    return {
        "Strategy": strategy_name,
        "Total_Trades": len(pnls),
        "Win_Rate_%": win_rate,
        "Expectancy_$": expectancy,
        "Profit_Factor": profit_factor,
        "Avg_Win_$": avg_profit,
        "Avg_Loss_$": avg_loss,
        "Max_Win_$": max(wins) if wins else 0.0,
        "Max_Loss_$": min(losses) if losses else 0.0,
        "Win_Streak": max_win_streak,
        "Loss_Streak": max_loss_streak,
    }

if __name__ == "__main__":
    valid_strategies = get_valid_strategies()
    
    parser = argparse.ArgumentParser(description="Advanced Strategy Analyzer for Options-Zero-Game.")
    parser.add_argument(
        '--strategy',
        type=str,
        default='ALL',
        choices=valid_strategies + ['ALL'],
        help="The specific opening strategy to test, or 'ALL' to run a comparative analysis."
    )
    parser.add_argument('-n', '--episodes', type=int, default=10, help="The number of episodes to run per strategy.")
    parser.add_argument('--model_path', type=str, default='./best_ckpt/ckpt_best.pth.tar', help="Path to the trained model checkpoint.")
    parser.add_argument(
        '--symbol',
        type=str,
        default=None,
        help="Force evaluation on a specific historical symbol (e.g., 'SPY', 'TSLA'). Defaults to random."
    )
    parser.add_argument(
        '--days',
        type=int,
        default=20,
        help="Force a specific episode length in days for the analysis."
    )
    # --- NEW: Argument to accept a timestamp from the automation script ---
    parser.add_argument('--timestamp', type=str, default=None, help="A specific timestamp (YYYYMMDD_HHMMSS) to use for the output filename.")
    
    args = parser.parse_args()

    # Determine which strategies to run
    if args.strategy == 'ALL':
        strategies_to_run = valid_strategies
        print(f"--- Starting Comparative Analysis for ALL {len(strategies_to_run)} Strategies ---")
    else:
        strategies_to_run = [args.strategy]
        print(f"--- Starting Analysis for Strategy: {args.strategy} ---")
    
    print(f"--- Running for {args.episodes} episodes per strategy ---")

    all_stats = []
    
    # --- The Main Analysis Loop ---
    for strategy_name in tqdm(strategies_to_run, desc="Overall Progress"):
        
        all_episode_results = []
        
        # Inner loop for running episodes for the current strategy
        for i in tqdm(range(args.episodes), desc=f"Testing {strategy_name}", leave=False):
            current_main_config = copy.deepcopy(main_config)
            current_create_config = copy.deepcopy(create_config)
            
            current_main_config.env.forced_opening_strategy_name = strategy_name
            current_main_config.env.is_eval_mode = True
            current_main_config.env.forced_historical_symbol = args.symbol
            current_main_config.env.n_evaluator_episode = 1
            current_main_config.env.evaluator_env_num = 1
            current_main_config.env.forced_episode_length = args.days
            
            _, returns = eval_muzero(
                [current_main_config, current_create_config],
                seed=i,
                num_episodes_each_seed=1,
                print_seed_details=False,
                model_path=args.model_path
            )
            
            if returns and len(returns[0]) > 0:
                pnl = returns[0][0]
                all_episode_results.append({'pnl': pnl, 'duration': -1})
        
        # Calculate stats for the completed strategy and store them
        strategy_stats = calculate_statistics(all_episode_results, strategy_name)
        if strategy_stats:
            all_stats.append(strategy_stats)

    # --- MODIFIED: Display and Save the Final Comparative Table ---
    if not all_stats:
        print("\nNo data was collected to generate a report.")
    else:
        print("\n" + "="*80)
        print("--- Comparative Strategy Analysis Results ---")
        print("="*80)
        
        df = pd.DataFrame(all_stats)
        df.set_index('Strategy', inplace=True)
        pd.options.display.float_format = '{:,.2f}'.format
        print(df)
        print("\n" + "="*80)

        # <<< --- NEW: Save the results to a timestamped JSON file --- >>>
        
        # 1. Define the output directory
        output_dir = "zoo/options_zero_game/visualizer-ui/build/reports"
        os.makedirs(output_dir, exist_ok=True)

        # 2. Use the provided timestamp if available, otherwise generate a new one.
        timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"strategy_report_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)

        # 3. Convert the list of dictionaries to a JSON string and save
        try:
            with open(output_path, 'w') as f:
                json.dump(all_stats, f, indent=2)
            print(f"✅ Successfully saved detailed report to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save report. Error: {e}")
