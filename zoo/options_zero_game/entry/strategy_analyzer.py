# zoo/options_zero_game/entry/strategy_analyzer.py
# <<< FINAL VERSION WITH COMPARATIVE ANALYSIS >>>

import argparse
import math
import copy
from tqdm import tqdm
import pandas as pd

from lzero.entry import eval_muzero
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config, create_config
# Import environment files so the framework can find them
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

def get_valid_strategies() -> list:
    """Helper function to get a list of all valid opening strategies."""
    temp_env = zoo.options_zero_game.envs.options_zero_game_env.OptionsZeroGameEnv()
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
    
    expectancy = ((win_rate / 100) * avg_profit) + ((1 - win_rate / 100) * avg_loss)
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
        default='ALL', # Default to running all strategies
        choices=valid_strategies + ['ALL'], # Add 'ALL' as a valid choice
        help="The specific opening strategy to test, or 'ALL' to run a comparative analysis."
    )
    parser.add_argument('-n', '--episodes', type=int, default=10, help="The number of episodes to run per strategy.")
    parser.add_argument('--model_path', type=str, default='./best_ckpt/ckpt_best.pth.tar', help="Path to the trained model checkpoint.")
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
            current_main_config.env.n_evaluator_episode = 1
            current_main_config.env.evaluator_env_num = 1
            
            _, returns = eval_muzero(
                [current_main_config, current_create_config],
                seed=i,
                num_episodes_each_seed=1,
                print_seed_details=False,
                model_path=args.model_path
            )
            
            if returns and len(returns[0]) > 0:
                pnl = returns[0][0]
                # Duration is still a known limitation
                all_episode_results.append({'pnl': pnl, 'duration': -1})
        
        # Calculate stats for the completed strategy and store them
        strategy_stats = calculate_statistics(all_episode_results, strategy_name)
        if strategy_stats:
            all_stats.append(strategy_stats)

    # --- Display the Final Comparative Table ---
    if not all_stats:
        print("\nNo data was collected to generate a report.")
    else:
        print("\n" + "="*80)
        print("--- Comparative Strategy Analysis Results ---")
        print("="*80)
        
        # Use pandas for a clean, professional-looking table
        df = pd.DataFrame(all_stats)
        df.set_index('Strategy', inplace=True)
        
        # Format the dataframe for better readability
        pd.options.display.float_format = '{:,.2f}'.format
        
        print(df)
        print("\n" + "="*80)
