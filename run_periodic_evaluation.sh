#!/bin/bash

# ===================================================================
#          Options-Zero-Game Periodic Evaluation Script
# ===================================================================

# --- Configuration ---
EXPERIMENT_DIR="experiments/options_zero_game_muzero_agent_v2.2-Final_ns50_upc2000_bs256"
BEST_CKPT_DEST="best_ckpt/ckpt_best.pth.tar"
REPORTS_DIR="zoo/options_zero_game/visualizer-ui/build/reports"

# --- Script Logic ---

echo "--- [$(date)] Starting Periodic Evaluation ---"

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

# <<< --- NEW: Generate a single, consistent timestamp --- >>>
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 3. Run the strategy analyzer, passing the timestamp to it
echo "Running strategy analyzer for all strategies using historical data only..."
python3 zoo/options_zero_game/entry/strategy_analyzer.py --strategy ALL -n 10 --timestamp "$TIMESTAMP" --symbol ANY

echo "Analysis complete. Report and model will be saved with timestamp: $TIMESTAMP"

# <<< --- NEW: Archive the checkpoint used for this run --- >>>
# 4. Copy the checkpoint that was just used for the analysis into the reports directory
ARCHIVE_CKPT_PATH="${REPORTS_DIR}/ckpt_best_${TIMESTAMP}.pth.tar"
if [ -f "$BEST_CKPT_DEST" ]; then
    cp "$BEST_CKPT_DEST" "$ARCHIVE_CKPT_PATH"
    echo "Successfully archived checkpoint to: $ARCHIVE_CKPT_PATH"
else
    echo "WARNING: Could not find checkpoint at $BEST_CKPT_DEST to archive."
fi

echo "--- [$(date)] Evaluation Finished ---"
