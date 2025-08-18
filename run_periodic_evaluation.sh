#!/bin/bash

# ===================================================================
#          Options-Zero-Game Periodic Evaluation Script
# ===================================================================

# --- Configuration ---
EXPERIMENT_DIR="experiments/options_zero_game_muzero_agent_v2.2-Final_ns50_upc2000_bs256"
BEST_CKPT_DEST="best_ckpt/ckpt_best.pth.tar"
REPORTS_DIR="zoo/options_zero_game/visualizer-ui/build/reports"

# <<< --- NEW: Define the temporary directory used by the analyzer --- >>>
ANALYZER_RUN_DIR="strategy_eval/strategy_analyzer_runs*"

# --- Script Logic ---

echo "--- [$(date)] Starting Periodic Evaluation ---"

# <<< --- NEW: Clean up the previous analysis run directory --- >>>
echo "Cleaning up previous analysis run directories: $ANALYZER_RUN_DIR"
# The -f flag ensures this command doesn't fail if the directory doesn't exist yet
rm -rf "$ANALYZER_RUN_DIR"
echo "Cleanup complete."

# 1. Find and copy the latest, best checkpoint to the location the evaluator uses
LATEST_CKPT="${EXPERIMENT_DIR}/ckpt/ckpt_best.pth.tar"

if [ -f "$LATEST_CKPT" ]; then
    echo "Found latest checkpoint: $LATEST_CKPT"
    cp "$LATEST_CKPT" "$BEST_CKPT_DEST"
    echo "Successfully copied to $BEST_CKPT_DEST for evaluation run."
else
    echo "ERROR: Could not find the latest checkpoint at $LATEST_CKPT. Aborting."
    exit 1
fi

# 2. Create the reports directory if it doesn't exist
mkdir -p "$REPORTS_DIR"

# 3. Generate a single, consistent timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 4. Run the strategy analyzer, passing the timestamp and symbol flag
echo "Running strategy analyzer for all strategies using historical data only..."
python3 zoo/options_zero_game/entry/strategy_analyzer.py --strategy ALL -n 10 --timestamp "$TIMESTAMP" --symbol ANY

echo "Analysis complete. Report and model will be saved with timestamp: $TIMESTAMP"

# 5. Archive the checkpoint used for this run
ARCHIVE_CKPT_PATH="${REPORTS_DIR}/ckpt_best_${TIMESTAMP}.pth.tar"
if [ -f "$BEST_CKPT_DEST" ]; then
    cp "$BEST_CKPT_DEST" "$ARCHIVE_CKPT_PATH"
    echo "Successfully archived checkpoint to: $ARCHIVE_CKPT_PATH"
else
    echo "WARNING: Could not find checkpoint at $BEST_CKPT_DEST to archive."
fi

# <<< --- NEW: Clean up the temporary directory after a SUCCESSFUL run --- >>>
# 7. Clean up the analysis run directory for housekeeping.
echo "Cleaning up temporary analysis directory: $ANALYZER_RUN_DIR"
rm -rf "$ANALYZER_RUN_DIR"
echo "Housekeeping complete."

echo "--- [$(date)] Evaluation Finished ---"
