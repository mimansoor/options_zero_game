import argparse
import math
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import shutil
import glob

from lzero.entry import eval_muzero
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config, create_config
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

# --- All helper functions (get_valid_strategies, calculate_statistics, calculate_trader_score) are unchanged ---

def get_valid_strategies() -> list:
    temp_env_cfg = copy.deepcopy(main_config.env)
    temp_env = zoo.options_zero_game.envs.options_zero_game_env.OptionsZeroGameEnv(cfg=temp_env_cfg)
    return [name for name in temp_env.actions_to_indices if name.startswith('OPEN_')]

def calculate_statistics(results: list, strategy_name: str) -> dict:
    """Calculates a full statistical summary and returns it as a dictionary."""
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
    
    # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
    # If there are no losses, the profit factor is infinite. We represent this
    # with None, which correctly serializes to the valid JSON value 'null'.
    profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else None

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
        "Profit_Factor": profit_factor, # This will now be a number or None
        "Avg_Win_$": avg_profit,
        "Avg_Loss_$": avg_loss,
        "Max_Win_$": max(wins) if wins else 0.0,
        "Max_Loss_$": min(losses) if losses else 0.0,
        "Win_Streak": max_win_streak,
        "Loss_Streak": max_loss_streak,
    }

def calculate_trader_score(strategy_data):
    """
    Calculates a unified 'Trader's Score' using a robust multi-factor model.
    This version correctly separates the logic for profitable and unprofitable
    strategies to ensure a meaningful and correctly signed score.
    """
    expectancy = strategy_data.get("Expectancy_$", 0)
    
    # --- The Fundamental Check: Is the strategy profitable? ---
    if expectancy <= 0:
        # For losing strategies, the score is a direct function of its negative expectancy.
        # A larger loss results in a score closer to -1.
        # We use a scaling factor to prevent tanh from saturating too quickly.
        # A max loss of 500k is a reasonable scale.
        return np.tanh(expectancy / 50000) # Returns a value between -1 and 0.

    # --- If we reach here, the strategy is PROFITABLE. ---
    
    # "Good" Factors (Guaranteed Positive)
    profit_factor = strategy_data.get("Profit_Factor", 0)
    win_rate = strategy_data.get("Win_Rate_%", 0)
    
    # "Bad" Factors (Measures of Risk/Drawdown)
    # Add 1 to avoid division by zero for a perfect strategy with zero loss.
    avg_loss = 1 + abs(strategy_data.get("Avg_Loss_$", 0))
    max_loss = 1 + abs(strategy_data.get("Max_Loss_$", 0))

    # --- Calculate the Score for Profitable Strategies ---
    # The raw score is a ratio of profitability and consistency vs. risk.
    # Expectancy is the core driver.
    raw_score = (expectancy * win_rate * profit_factor) / (avg_loss + max_loss)

    # Use log to scale the raw score gracefully.
    # Since expectancy > 0, raw_score will be positive. log will work correctly.
    log_scaled_score = np.log(1 + raw_score)

    # Tanh normalizes the final score to a clean [0, 1] range.
    final_score = np.tanh(log_scaled_score)

    return final_score if np.isfinite(final_score) else 0.0 # Default profitable but non-finite score to 0

if __name__ == "__main__":
    valid_strategies = get_valid_strategies()
    
    parser = argparse.ArgumentParser(description="Advanced Strategy Analyzer for Options-Zero-Game.")
    parser.add_argument('--strategy', type=str, default='ALL', choices=valid_strategies + ['ALL'], help="The specific opening strategy to test, or 'ALL' to run all.")
    parser.add_argument('--tail', type=int, default=0, help="Run only the last N opening strategies from the master list. A non-zero value overrides --strategy ALL.")
    parser.add_argument('-n', '--episodes', type=int, default=10, help="The number of episodes to run per strategy.")
    parser.add_argument('--model_path', type=str, default='./best_ckpt/ckpt_best.pth.tar', help="Path to the trained model checkpoint.")
    parser.add_argument('--symbol', type=str, default=None, help="Force evaluation on a specific historical symbol (e.g., 'SPY'), or use 'ANY' to force random historical data.")
    parser.add_argument('--start_seed', type=int, default=0, help="The starting seed for the episode sequence.")
    parser.add_argument('--timestamp', type=str, default=None, help="A specific timestamp (YYYYMMDD_HHMMSS) to use for the output filename.")
    parser.add_argument('--days', type=int, default=20, help="Force a specific episode length in days for the analysis.")

    # <<< --- NEW: Add the override arguments --- >>>
    parser.add_argument('--profit_target_pct', type=float, default=None, help="Override the global profit target percentage (e.g., 3 for 3%).")
    parser.add_argument('--credit_tp_pct', type=float, default=None, help="Override the take-profit percentage for credit strategies (e.g., 70 for 70%).")
    parser.add_argument('--debit_tp_mult', type=float, default=None, help="Override the take-profit multiple for debit strategies (e.g., 2 for 2x).")
    
    args = parser.parse_args()

    # Print any overrides being used so it's clear in the logs
    if args.profit_target_pct is not None:
        print(f"--- OVERRIDE: Setting Profit Target to {args.profit_target_pct}% ---")
    if args.credit_tp_pct is not None:
        print(f"--- OVERRIDE: Setting Credit Take-Profit to {args.credit_tp_pct}% ---")
    if args.debit_tp_mult is not None:
        print(f"--- OVERRIDE: Setting Debit Take-Profit to {args.debit_tp_mult}x ---")

    # ... (logic to determine strategies_to_run is unchanged) ...
    strategies_to_run = valid_strategies[-args.tail:] if args.tail > 0 else (valid_strategies if args.strategy == 'ALL' else [args.strategy])

    # --- The Main Analysis Loop ---
    all_stats = []
    ANALYZER_RUN_DIR_PREFIX = 'strat_eval/strategy_analyzer_runs'

    # Initial cleanup
    for dir_path in glob.glob(f"{ANALYZER_RUN_DIR_PREFIX}*"):
        if os.path.isdir(dir_path): shutil.rmtree(dir_path)

    for strategy_name in tqdm(strategies_to_run, desc="Overall Progress"):
        all_episode_results = []
        for i in tqdm(range(args.episodes), desc=f"Testing {strategy_name}", leave=False):
            current_main_config = copy.deepcopy(main_config)
            current_create_config = copy.deepcopy(create_config)
            
            current_main_config.exp_name = ANALYZER_RUN_DIR_PREFIX
            
            # <<< --- NEW: Apply the overrides inside the loop --- >>>
            if args.profit_target_pct is not None:
                current_main_config.env.profit_target_pct = args.profit_target_pct
            if args.credit_tp_pct is not None:
                current_main_config.env.credit_strategy_take_profit_pct = args.credit_tp_pct
            if args.debit_tp_mult is not None:
                current_main_config.env.debit_strategy_take_profit_multiple = args.debit_tp_mult
            
            # ... (rest of the config setup is unchanged) ...
            if args.symbol:
                current_main_config.env.price_source = 'historical'
                current_main_config.env.forced_historical_symbol = None if args.symbol.upper() == 'ANY' else args.symbol
            
            current_main_config.env.forced_opening_strategy_name = strategy_name
            current_main_config.env.is_eval_mode = True
            current_main_config.env.n_evaluator_episode = 1
            current_main_config.env.evaluator_env_num = 1
            current_main_config.env.forced_episode_length = args.days
            
            current_seed = args.start_seed + i
            
            _, returns = eval_muzero(
                [current_main_config, current_create_config],
                seed=current_seed,
                num_episodes_each_seed=1,
                print_seed_details=False,
                model_path=args.model_path
            )
            
            if returns and len(returns[0]) > 0:
                pnl = returns[0][0]
                all_episode_results.append({'pnl': pnl, 'duration': -1})
        
        strategy_stats = calculate_statistics(all_episode_results, strategy_name)
        if strategy_stats:
            all_stats.append(strategy_stats)

        # Per-strategy cleanup
        try:
            for dir_path in glob.glob(f"{ANALYZER_RUN_DIR_PREFIX}*"):
                if os.path.isdir(dir_path): shutil.rmtree(dir_path)
        except OSError as e:
            print(f"Warning: Could not clean up directories. Error: {e}")

    # ... (The rest of the script for calculating Trader_Score and saving the report is unchanged) ...
    if all_stats:
        for strategy_row in all_stats:
            strategy_row['Trader_Score'] = calculate_trader_score(strategy_row)
        
        df = pd.DataFrame(all_stats).sort_values(by="Trader_Score", ascending=False).set_index('Strategy')
        print("\n" + "="*80)
        print("--- Comparative Strategy Analysis Results ---")
        print("="*80)
        print(df)
        
        output_dir = "zoo/options_zero_game/visualizer-ui/build/reports"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_report_{timestamp}.json"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"✅ Successfully saved detailed report to: {output_path}")
