import os
import sys
import subprocess
import json
from datetime import datetime
from tqdm import tqdm
import shutil
import glob
import pandas as pd
import numpy as np
import argparse

# Add the project root to the Python path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import all necessary calculation functions from the worker script
from zoo.options_zero_game.entry.strategy_analyzer import (
    get_valid_strategies, calculate_statistics, calculate_trader_score, calculate_elo_ratings
)

def run_full_analysis():
    """
    The main orchestrator for running a full strategy analysis.
    This script handles cleanup, checkpoint copying, calls the worker
    script for each strategy, and performs all post-processing, including
    maintaining a persistent Elo rating file.
    """
    # --- Configuration ---
    REPORTS_DIR = "zoo/options_zero_game/visualizer-ui/build/reports"
    ANALYZER_SCRIPT_PATH = "zoo/options_zero_game/entry/strategy_analyzer.py"
    ANALYZER_RUN_DIR = 'eval/strategy_analyzer'
    BEST_CKPT_DEST = "best_ckpt/ckpt_best.pth.tar"
    EXPERIMENT_DIR = "experiments/options_zero_game_muzero_agent_v2.2-Final_ns50_upc2000_bs256"
    ELO_STATE_FILE = "zoo/options_zero_game/experts/elo_ratings.json"

    print(f"--- [ {datetime.now()} ] Starting Full Analysis Cycle ---")

    # <<< --- NEW: Add a command-line argument for determinism --- >>>
    parser = argparse.ArgumentParser(description="Run a full, deterministic strategy analysis with Elo ratings.")
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help="Run the analysis in a fully deterministic mode by disabling the C++ MCTS. Slower but reproducible."
    )
    args, remaining_argv = parser.parse_known_args()

    print(f"--- [ {datetime.now()} ] Starting Full Analysis Cycle ---")
    if args.deterministic:
        print("--- RUNNING IN DETERMINISTIC MODE ---")

    try:
        # 1. Initial Cleanup
        print(f"Performing initial cleanup of old '{ANALYZER_RUN_DIR}*' directories...")
        for dir_path in glob.glob(f"{ANALYZER_RUN_DIR}*"):
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
        print("Initial cleanup complete.")

        # 2. Copy the Latest Checkpoint
        latest_ckpt_path = os.path.join(EXPERIMENT_DIR, "ckpt", "ckpt_best.pth.tar")
        if not os.path.exists(latest_ckpt_path):
            raise FileNotFoundError(f"FATAL: Could not find latest checkpoint at '{latest_ckpt_path}'.")
        os.makedirs(os.path.dirname(BEST_CKPT_DEST), exist_ok=True)
        shutil.copy(latest_ckpt_path, BEST_CKPT_DEST)
        print(f"Successfully copied latest checkpoint to '{BEST_CKPT_DEST}'.")

        # 3. Load existing Elo ratings
        existing_ratings = {}
        if os.path.exists(ELO_STATE_FILE):
            try:
                with open(ELO_STATE_FILE, 'r') as f:
                    existing_ratings = json.load(f)
                print(f"Loaded {len(existing_ratings)} existing Elo ratings from {ELO_STATE_FILE}")
            except (FileNotFoundError, json.JSONDecodeError):
                print(f"Warning: Elo file at {ELO_STATE_FILE} is corrupted or empty. Starting fresh.")
        else:
            print("No existing Elo file found. Starting all strategies at default rating.")

        # 4. Prepare the intermediate report file for raw PNL data
        os.makedirs(REPORTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = os.path.join(REPORTS_DIR, f"temp_pnl_data_{timestamp}.json")
        print(f"Initializing intermediate PNL file: {intermediate_file}")
        with open(intermediate_file, 'w') as f:
            json.dump([], f)

        # 5. Get the clean list of strategies
        strategy_list = get_valid_strategies()
        print(f"Found {len(strategy_list)} strategies to analyze.")

        # 6. The Main Loop: Run one analysis per strategy
        for strategy in tqdm(strategy_list, desc="Overall Progress"):
            print(f"\n--- Analyzing strategy: {strategy} ---")
            command = [
                "python3", ANALYZER_SCRIPT_PATH,
                "--strategy", strategy,
                "--report_file", intermediate_file,
                "--exp_name", ANALYZER_RUN_DIR
            ]

            # <<< --- NEW: Pass the deterministic flag to the worker script --- >>>
            if args.deterministic:
                command.append("--deterministic")

            command.extend(remaining_argv)

            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print("\n" + "="*80)
                print(f"❌ WARNING: Analysis for strategy '{strategy}' FAILED.")
                print(f"Exit Code: {e.returncode}")
                print("--- ERROR TRACEBACK FROM FAILED PROCESS ---")
                print(e.stderr)
                print("="*80 + "\n")
                print("Continuing to next strategy...")
            except KeyboardInterrupt:
                print("\nAnalysis interrupted by user. Exiting.")
                return

        # 7. Post-Processing Step
        print("\n--- All strategies analyzed. Starting post-processing and Elo calculation... ---")
        
        with open(intermediate_file, 'r') as f:
            raw_pnl_data = json.load(f)

        all_stats, all_pnl_by_strategy = [], {}
        for item in raw_pnl_data:
            stats = calculate_statistics([{'pnl': pnl} for pnl in item['pnl_results']], item['strategy'])
            if stats: all_stats.append(stats)
            all_pnl_by_strategy[item['strategy']] = item['pnl_results']

        # a) Calculate both standard and PnL-weighted Elo ratings.
        standard_elo = calculate_elo_ratings(all_pnl_by_strategy, existing_ratings)
        pnl_weighted_elo = calculate_pnl_weighted_elo(all_pnl_by_strategy, existing_ratings)

        # b) Add all metrics to the final stats dictionary.
        for strategy_row in all_stats:
            strategy_name = strategy_row['Strategy']
            strategy_row['Trader_Score'] = calculate_trader_score(strategy_row)
            strategy_row['ELO_Rank'] = standard_elo.get(strategy_name, 1200)
            strategy_row['PnL_ELO_Rank'] = pnl_weighted_elo.get(strategy_name, 1200) # <-- Add the new rank

        # c) Update the persistent Elo file with the STANDARD Elo ratings for next time.
        with open(ELO_STATE_FILE, 'w') as f:
            json.dump(standard_elo, f, indent=2)
        print(f"Successfully updated and saved standard Elo ratings to {ELO_STATE_FILE}")
        
        # d) Sort the final DataFrame by the new PnL-Weighted Elo Rank for display.
        df = pd.DataFrame(all_stats).sort_values(by="PnL_ELO_Rank", ascending=False).set_index('Strategy')
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 200)
        print("\n" + "="*120)
        print("--- FINAL RANKED STRATEGY REPORT (SORTED BY PNL-WEIGHTED ELO) ---")
        print("="*120)
        print(df)
        print("="*120)

        # 8. Save the final, complete report
        final_report_file = os.path.join(REPORTS_DIR, f"strategy_report_{timestamp}.json")
        # Sanitize the final data one last time before saving to JSON.
        sanitized_stats = []
        for strategy_row in all_stats:
            sanitized_row = {}
            for key, value in strategy_row.items():
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    # Replace NaN/inf with None, which serializes to 'null' in JSON.
                    sanitized_row[key] = None
                else:
                    sanitized_row[key] = value
            sanitized_stats.append(sanitized_row)
        with open(final_report_file, 'w') as f:
            json.dump(sanitized_stats, f, indent=2)
        
        os.remove(intermediate_file)

        # 9. Final Cleanup
        print(f"Performing final cleanup of old '{ANALYZER_RUN_DIR}*' directories...")
        for dir_path in glob.glob(f"{ANALYZER_RUN_DIR}*"):
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
        print("Final cleanup complete.")

        # 10. Archive the checkpoint
        archive_ckpt_path = os.path.join(REPORTS_DIR, f"ckpt_best_{timestamp}.pth.tar")
        shutil.copy(BEST_CKPT_DEST, archive_ckpt_path)
        print(f"Final report and checkpoint for this cycle saved to: {REPORTS_DIR}")

    except Exception as e:
        print("\n" + "#"*80)
        print(f"## FATAL ERROR in orchestrator script: {e}")
        print("#"*80 + "\n")
        sys.exit(1)

    print(f"\n--- [ {datetime.now()} ] Full Analysis Finished Successfully ---")

if __name__ == "__main__":
    run_full_analysis()
