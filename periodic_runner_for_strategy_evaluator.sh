#!/bin/bash

# ===================================================================
#           Options-Zero-Game Automation Loop
# This script is designed to be run inside a persistent session
# like tmux or screen. It handles activating the Python environment
# and running the evaluation script every 6 hours.
# ===================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# The main loop
while true
do
    echo "========================================================"
    echo "--- [$(date)] Activating Python environment and starting new cycle ---"
    
    # Activate the Python virtual environment
    # This ensures all subsequent python commands use the correct packages.
    source ../venv/bin/activate

    # Run your main evaluation script
    ./run_periodic_evaluation.sh --profit_target_pct 3 --credit_tp_pct 70 --debit_tp_mult 2

    # Deactivate the environment (good practice)
    deactivate

    echo "--- [$(date)] Cycle complete. Sleeping for 6 hours... ---"
    echo "========================================================"
    
    # Sleep for 6 hours (21600 seconds)
    sleep 21600
done
