import numpy as np
import os

# <<< --- THE DEFINITIVE FIX IS HERE --- >>>
# Get the absolute path of the directory this script is in.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# The model path is now the same directory as the script.
MODEL_PATH = SCRIPT_DIR
STATIONARY_DIST_FILE = os.path.join(MODEL_PATH, "iv_stationary_distribution.npy")
TRANSITION_MATRIX_FILE = os.path.join(MODEL_PATH, "iv_transition_matrix.npy")

# ... (The rest of the file remains exactly the same) ...
# Define the order of your regimes EXACTLY as they appear in your config file
# 0: Crisis, 1: Elevated Vol, 2: Complacent, 3: Volatile/Choppy
REGIME_NAMES = [
    'Crisis (High Vol, Bearish)',
    'Elevated Vol (Bullish)',
    'Complacent (Low Vol, Neutral)',
    'Volatile (Choppy, Neutral)'
]

def tune_iv_model():
    """
    Loads the existing IV regime model and manually adjusts its probabilities
    to create a more balanced and realistic training environment.
    """
    print("--- Starting IV Regime Model Tuning ---")

    # --- 1. Tune the Stationary Distribution (Starting Probabilities) ---
    print("\nTuning Stationary Distribution (Starting Probabilities)...")
    try:
        stationary_dist = np.load(STATIONARY_DIST_FILE)
        print("Original Distribution:")
        for i, name in enumerate(REGIME_NAMES):
            print(f"  - {name}: {stationary_dist[i]:.2%}")

        # Define your desired long-term probabilities
        desired_dist = np.array([
            0.10,  # Crisis: Happen only 10% of the time
            0.40,  # Elevated Vol: Happen 40% of the time
            0.35,  # Complacent: Happen 35% of the time
            0.15,  # Volatile/Choppy: Happen 15% of the time
        ])

        # Normalize to ensure it sums to 1
        tuned_dist = desired_dist / desired_dist.sum()

        print("\nTuned Distribution:")
        for i, name in enumerate(REGIME_NAMES):
            print(f"  - {name}: {tuned_dist[i]:.2%}")

        np.save(STATIONARY_DIST_FILE, tuned_dist)
        print(f"\nSuccessfully saved tuned stationary distribution to '{STATIONARY_DIST_FILE}'")

    except FileNotFoundError:
        print(f"ERROR: Cannot find {STATIONARY_DIST_FILE}. Please run iv_regime_analyzer.py first.")
        return

    # --- 2. Tune the Transition Matrix (Reduce "Stickiness") ---
    print("\n\nTuning Transition Matrix (Reducing 'Stickiness')...")
    try:
        transition_matrix = np.load(TRANSITION_MATRIX_FILE)
        print("\nOriginal Transition Matrix (Partial - Diagonal Values):")
        for i, name in enumerate(REGIME_NAMES):
            print(f"  - Probability of staying in {name}: {transition_matrix[i, i]:.2%}")
        
        crisis_stay_prob = transition_matrix[0, 0]
        amount_to_redistribute = crisis_stay_prob * 0.30

        transition_matrix[0, 0] -= amount_to_redistribute
        transition_matrix[0, 1] += amount_to_redistribute

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        tuned_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)

        print("\nTuned Transition Matrix (Partial - Diagonal Values):")
        for i, name in enumerate(REGIME_NAMES):
            print(f"  - Probability of staying in {name}: {tuned_matrix[i, i]:.2%}")

        np.save(TRANSITION_MATRIX_FILE, tuned_matrix)
        print(f"\nSuccessfully saved tuned transition matrix to '{TRANSITION_MATRIX_FILE}'")

    except FileNotFoundError:
        print(f"ERROR: Cannot find {TRANSITION_MATRIX_FILE}. Please run iv_regime_analyzer.py first.")

    print("\n--- IV Regime Model Tuning Complete! ---")


if __name__ == "__main__":
    tune_iv_model()
