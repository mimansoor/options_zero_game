#!/bin/bash

# ===================================================================
#          Options-Zero-Game Periodic Evaluation Script
# This script can now accept command-line arguments to override
# the environment's profit-taking parameters.
#
# Usage:
#   ./run_periodic_evaluation.sh
#   ./run_periodic_evaluation.sh --profit_target_pct 3 --credit_tp_pct 70 --debit_tp_mult 2
# ===================================================================

# --- Configuration ---
EXPERIMENT_DIR="experiments/options_zero_game_muzero_agent_v2.2-Final_ns50_upc2000_bs256"
BEST_CKPT_DEST="best_ckpt/ckpt_best.pth.tar"
REPORTS_DIR="zoo/options_zero_game/visualizer-ui/build/reports"
ANALYZER_RUN_DIR="strategy_analyzer_runs"

# --- Script Logic ---

echo "--- [$(date)] Starting Periodic Evaluation ---"

# 1. Clean up the previous analysis run directory
echo "Cleaning up previous analysis run directory: $ANALYZER_RUN_DIR"
rm -rf "$ANALYZER_RUN_DIR"
echo "Cleanup complete."

# 2. Find and copy the latest, best checkpoint
LATEST_CKPT="${EXPERIMENT_DIR}/ckpt/ckpt_best.pth.tar"

if [ -f "$LATEST_CKPT" ]; then
    echo "Found latest checkpoint: $LATEST_CKPT"
    cp "$LATEST_CKPT" "$BEST_CKPT_DEST"
    echo "Successfully copied to $BEST_CKPT_DEST for evaluation run."
else
    echo "ERROR: Could not find the latest checkpoint at $LATEST_CKPT. Aborting."
    exit 1
fi

# 3. Create the reports directory if it doesn't exist
mkdir -p "$REPORTS_DIR"

# 4. Generate a single, consistent timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 5. Run the strategy analyzer
echo "Running strategy analyzer for all strategies using historical data only..."

# <<< --- THE DEFINITIVE FIX --- >>>
# The "$@" variable in bash is a special variable that holds all of the
# command-line arguments passed to this script. By adding it here, we
# are transparently forwarding them to the Python script.
python3 zoo/options_zero_game/entry/strategy_analyzer.py \
    --strategy ALL \
    -n 15 \
    --timestamp "$TIMESTAMP" \
    --symbol ANY \
    "$@"  # <-- This is the key change

echo "Analysis complete. Report and model will be saved with timestamp: $TIMESTAMP"

# 6. Archive the checkpoint used for this run
ARCHIVE_CKPT_PATH="${REPORTS_DIR}/ckpt_best_${TIMESTAMP}.pth.tar"
if [ -f "$BEST_CKPT_DEST" ]; then
    cp "$BEST_CKPT_DEST" "$ARCHIVE_CKPT_PATH"
    echo "Successfully archived checkpoint to: $ARCHIVE_CKPT_PATH"
else
    echo "WARNING: Could not find checkpoint at $BEST_CKPT_DEST to archive."
fi

# 7. Clean up the temporary analysis directory
echo "Cleaning up temporary analysis directory: $ANALYZER_RUN_DIR"
rm -rf "$ANALYZER_RUN_DIR"
echo "Housekeeping complete."

echo "--- [$(date)] Evaluation Finished ---"
