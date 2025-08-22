import os
import sys
import subprocess
import json
from datetime import datetime
from tqdm import tqdm
import shutil
import glob

# Add the project root to the Python path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from zoo.options_zero_game.entry.strategy_analyzer import get_valid_strategies

def run_full_analysis():
    """
    The main orchestrator for running a full strategy analysis.
    This script handles cleanup, checkpoint copying, and calls the worker
    script for each strategy in a separate, isolated process.
    """
    # --- Configuration ---
    REPORTS_DIR = "zoo/options_zero_game/visualizer-ui/build/reports"
    ANALYZER_SCRIPT_PATH = "zoo/options_zero_game/entry/strategy_analyzer.py"
    ANALYZER_RUN_DIR_PREFIX = 'strat_eval/strategy_analyzer_runs'
    ANALYZER_RUN_DIR = 'strat_eval/strategy_analyzer' # This will be the prefix
    BEST_CKPT_DEST = "best_ckpt/ckpt_best.pth.tar"
    EXPERIMENT_DIR = "experiments/options_zero_game_muzero_agent_v2.2-Final_ns50_upc2000_bs256"
    
    print(f"--- [ {datetime.now()} ] Starting Full Analysis Cycle ---")

    try:
        # <<< --- NEW: 1. Initial Cleanup --- >>>
        # Clean up temporary directories from any previous runs.
        print(f"Performing initial cleanup of old '{ANALYZER_RUN_DIR_PREFIX}*' directories...")
        for dir_path in glob.glob(f"{ANALYZER_RUN_DIR_PREFIX}*"):
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
        print("Initial cleanup complete.")

        # <<< --- NEW: 2. Copy the Latest Checkpoint --- >>>
        latest_ckpt_path = os.path.join(EXPERIMENT_DIR, "ckpt", "ckpt_best.pth.tar")
        if not os.path.exists(latest_ckpt_path):
            print(f"FATAL: Could not find latest checkpoint at '{latest_ckpt_path}'. Aborting.")
            return
        
        os.makedirs(os.path.dirname(BEST_CKPT_DEST), exist_ok=True)
        shutil.copy(latest_ckpt_path, BEST_CKPT_DEST)
        print(f"Successfully copied latest checkpoint to '{BEST_CKPT_DEST}' for analysis.")

        # 3. Prepare the new report file
        os.makedirs(REPORTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(REPORTS_DIR, f"strategy_report_{timestamp}.json")
        
        print(f"Initializing new report file: {report_file}")
        with open(report_file, 'w') as f:
            json.dump([], f)

        # 4. Get the clean list of strategies
        try:
            strategy_list = get_valid_strategies()
            print(f"Found {len(strategy_list)} strategies to analyze.")
        except Exception as e:
            print(f"FATAL: Could not get strategy list. Error: {e}")
            return

        # 5. The Main Loop
        for strategy in tqdm(strategy_list, desc="Overall Progress"):
            print(f"\n--- Analyzing strategy: {strategy} ---")
            
            command = [
                "python3", ANALYZER_SCRIPT_PATH,
                "--strategy", strategy,
                "--report_file", report_file,
                "--exp_name", ANALYZER_RUN_DIR
            ]
            command.extend(sys.argv[1:])

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                # If the subprocess crashes, log its detailed error message (stderr)
                print("\n" + "="*80)
                print(f"❌ WARNING: Analysis for strategy '{strategy}' FAILED.")
                print(f"Exit Code: {e.returncode}")
                print("--- ERROR TRACEBACK FROM FAILED PROCESS ---")
                print(e.stderr) # This prints the full traceback from the crashed script
                print("="*80 + "\n")
                print("Continuing to next strategy...")
            except KeyboardInterrupt:
                print("\nAnalysis interrupted by user. Exiting.")
                return

        archive_ckpt_path = os.path.join(REPORTS_DIR, f"ckpt_best_{timestamp}.pth.tar")
        shutil.copy(BEST_CKPT_DEST, archive_ckpt_path)
        print(f"Successfully archived checkpoint to '{archive_ckpt_path}'")

    except Exception as e:
        # This catches errors in the orchestrator itself (e.g., file not found)
        print("\n" + "#"*80)
        print(f"## FATAL ERROR in orchestrator script: {e}")
        print("#"*80 + "\n")
        # We exit with a non-zero code to signal failure to the main loop script
        sys.exit(1)

    print(f"\n--- [ {datetime.now()} ] Full Analysis Finished Successfully ---")
    print(f"Final report saved to: {report_file}")

if __name__ == "__main__":
    run_full_analysis()
