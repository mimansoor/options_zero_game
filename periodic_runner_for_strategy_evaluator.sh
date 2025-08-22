#!/bin/bash

# ===================================================================
#           Options-Zero-Game Automation Loop (Cadence-Based)
# This script runs the evaluation script on a strict 6-hour cadence
# (e.g., at 00:00, 06:00, 12:00, 18:00), automatically adjusting
# the sleep time to account for the script's execution duration.
# ===================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# The main loop
while true
do
    echo "========================================================"
    echo "--- [$(date)] Activating Python environment and starting new cycle ---"

    # Activate the Python virtual environment
    source ../venv/bin/activate

    # Run your main evaluation script
    # The "$@" allows you to pass arguments to this loop script, which will be
    # forwarded to the evaluation script.
    # Example: ./run_loop.sh --profit_target_pct 3
    #./run_periodic_evaluation.sh "$@"
    ./run_periodic_evaluation.sh --profit_target_pct 3 --credit_tp_pct 70 --debit_tp_mult 2

    # Deactivate the environment (good practice)
    deactivate

    echo "--- [$(date)] Cycle complete. Calculating time until next 6-hour mark... ---"

    # <<< --- THE DEFINITIVE FIX: Cadence-Based Sleep Calculation --- >>>

    # 1. Define the interval in seconds (6 hours = 21600 seconds)
    INTERVAL=21600

    # 2. Get the current time in seconds since the Unix epoch
    CURRENT_TIME=$(date +%s)

    # 3. Calculate how long we need to sleep to wake up at the NEXT multiple of INTERVAL
    #    This is a standard shell arithmetic way to mimic cron's behavior.
    SLEEP_SECONDS=$(( ( ( ($CURRENT_TIME / $INTERVAL) + 1 ) * $INTERVAL ) - $CURRENT_TIME ))

    # Convert seconds to a human-readable format for the log
    SLEEP_HUMAN=$(printf '%dh:%dm:%ds\n' $(($SLEEP_SECONDS/3600)) $(($SLEEP_SECONDS%3600/60)) $(($SLEEP_SECONDS%60)))

    echo "Next run scheduled in $SLEEP_HUMAN. Sleeping..."
    echo "========================================================"

    sleep $SLEEP_SECONDS

done
