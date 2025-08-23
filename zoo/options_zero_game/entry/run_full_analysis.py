import os
import sys
import subprocess
import json
from datetime import datetime
from tqdm import tqdm
import shutil
import glob
import pandas as pd

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
    script for each strategy, and performs all post-processing.
    """
    # --- Configuration ---
    REPORTS_DIR = "zoo/options_zero_game/visualizer-ui/build/reports"
    ANALYZER_SCRIPT_PATH = "zoo/options_zero_game/entry/strategy_analyzer.py"
    ANALYZER_RUN_DIR = 'eval/strategy_analyzer'
    BEST_CKPT_DEST = "best_ckpt/ckpt_best.pth.tar"
    EXPERIMENT_DIR = "experiments/options_zero_game_muzero_agent_v2.2-Final_ns50_upc2000_bs256"
    
    print(f"--- [ {datetime.now()} ] Starting Full Analysis Cycle ---")

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

        # 3. Prepare the intermediate report file for raw PNL data
        os.makedirs(REPORTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = os.path.join(REPORTS_DIR, f"temp_pnl_data_{timestamp}.json")
        print(f"Initializing intermediate PNL file: {intermediate_file}")
        with open(intermediate_file, 'w') as f:
            json.dump([], f)

        # 4. Get the clean list of strategies
        strategy_list = get_valid_strategies()
        print(f"Found {len(strategy_list)} strategies to analyze.")

        # 5. The Main Loop: Run one analysis per strategy
        for strategy in tqdm(strategy_list, desc="Overall Progress"):
            print(f"\n--- Analyzing strategy: {strategy} ---")
            command = [
                "python3", ANALYZER_SCRIPT_PATH,
                "--strategy", strategy,
                "--report_file", intermediate_file,
                "--exp_name", ANALYZER_RUN_DIR
            ]
            command.extend(sys.argv[1:])

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

        # --- Post-Processing Step ---
        print("\n--- All strategies analyzed. Starting post-processing and Elo calculation... ---")
        
        with open(intermediate_file, 'r') as f:
            raw_pnl_data = json.load(f)

        all_stats = []
        all_pnl_by_strategy = {}
        
        for item in raw_pnl_data:
            strategy_name = item['strategy']
            pnl_results_for_stats = [{'pnl': pnl} for pnl in item['pnl_results']]
            stats = calculate_statistics(pnl_results_for_stats, strategy_name)
            if stats:
                all_stats.append(stats)
            all_pnl_by_strategy[strategy_name] = item['pnl_results']

        elo_ratings = calculate_elo_ratings(all_pnl_by_strategy)

        for strategy_row in all_stats:
            strategy_row['Trader_Score'] = calculate_trader_score(strategy_row)
            strategy_row['ELO_Rank'] = elo_ratings.get(strategy_row['Strategy'], 1200)

        # Print the final, ranked table to the console
        df = pd.DataFrame(all_stats).sort_values(by="ELO_Rank", ascending=False).set_index('Strategy')
        print("\n" + "="*120)
        print("--- FINAL RANKED STRATEGY REPORT (SORTED BY ELO) ---")
        print("="*120)
        print(df)
        print("="*120)
        
        # Save the final, complete report
        final_report_file = os.path.join(REPORTS_DIR, f"strategy_report_{timestamp}.json")
        with open(final_report_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        
        print(f"\nFinal report with Elo ratings saved to: {final_report_file}")
        
        os.remove(intermediate_file)

        archive_ckpt_path = os.path.join(REPORTS_DIR, f"ckpt_best_{timestamp}.pth.tar")
        shutil.copy(BEST_CKPT_DEST, archive_ckpt_path)
        print(f"Successfully archived checkpoint to '{archive_ckpt_path}'")

    except Exception as e:
        print("\n" + "#"*80)
        print(f"## FATAL ERROR in orchestrator script: {e}")
        print("#"*80 + "\n")
        sys.exit(1)

    print(f"\n--- [ {datetime.now()} ] Full Analysis Finished Successfully ---")

if __name__ == "__main__":
    run_full_analysis()
