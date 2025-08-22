#!/bin/bash

# ===================================================================
#           Options-Zero-Game Automation Loop (Robust)
# This script now calls a dedicated Python orchestrator, which is
# more robust than complex shell logic.
# ===================================================================
set -e

# The main loop
while true
do
    echo "========================================================"
    echo "--- [$(date)] Activating Python environment and starting new cycle ---"

    source ../venv/bin/activate

    # --- THE FIX: Call the new Python orchestrator script ---
    # The shell script no longer needs to know about the inner workings.
    # It just forwards all arguments to the robust Python script.
    python3 zoo/options_zero_game/entry/run_full_analysis.py "$@"

    deactivate

    echo "--- [$(date)] Cycle complete. Calculating time until next 6-hour mark... ---"
    
    # Cadence-based sleep logic remains the same
    INTERVAL=21600
    CURRENT_TIME=$(date +%s)
    SLEEP_SECONDS=$(( ( ( ($CURRENT_TIME / $INTERVAL) + 1 ) * $INTERVAL ) - $CURRENT_TIME ))
    SLEEP_HUMAN=$(printf '%dh:%dm:%ds\n' $(($SLEEP_SECONDS/3600)) $(($SLEEP_SECONDS%3600/60)) $(($SLEEP_SECONDS%60)))
    echo "Next run scheduled in $SLEEP_HUMAN. Sleeping..."
    echo "========================================================"
    
    sleep $SLEEP_SECONDS
done
