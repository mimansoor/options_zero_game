# zoo/options_zero_game/experts/iv_regime_analyzer.py
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import os

# --- Import your configuration from the main config file ---
from zoo.options_zero_game.config.options_zero_game_muzero_config import IV_REGIMES, MAX_STRIKE_OFFSET
from zoo.options_zero_game.envs.utils import generate_dynamic_iv_skew_table

def analyze_iv_regimes():
    """
    Analyzes historical volatility data to build a Markov Chain model
    for transitioning between user-defined IV regimes.
    """
    print("--- Starting IV Regime Analysis ---")
    
    # --- 1. Load Historical Data ---
    # We use VIX as a proxy for ATM IV, and ^SKEW for skew.
    # Note: ^SKEW data might be limited. We will use a fallback.
    print("Downloading historical volatility data (VIX)...")
    vix_data = yf.download('^VIX', period="10y", interval="1d")
    if vix_data.empty:
        raise ValueError("Could not download VIX data.")
    
    # We will use the VIX level to approximate the skew, as SKEW index data is less reliable.
    # High VIX -> High put skew. Low VIX -> Flat skew.
    # This is a reasonable and robust simplification.
    
    # --- 2. Create Anchor "Fingerprints" for Your Regimes ---
    print("Creating anchor vectors for each IV regime...")
    anchor_vectors = []
    for regime in IV_REGIMES:
        # The fingerprint is [atm_iv, put_skew, call_skew]
        # Skew is defined as (Far OTM IV - ATM IV)
        put_skew = regime['far_otm_put_iv'] - regime['atm_iv']
        call_skew = regime['far_otm_call_iv'] - regime['atm_iv']
        anchor_vectors.append([regime['atm_iv'], put_skew, call_skew])
    anchor_vectors = np.array(anchor_vectors)
    
    # --- 3. Label Each Day in History ---
    print("Labeling historical data points...")
    labels = []

    # <<< THE FIX: Force the results of .min() and .max() to be simple floats >>>
    # This prevents Pandas from passing a Series object into the list.
    min_vix = float(vix_data['Close'].min())
    max_vix = float(vix_data['Close'].max())
    
    min_skew, max_skew = 5, 40

    vix_values = vix_data['Close'].to_numpy()

    for vix in tqdm(vix_values, desc="Labeling Days"):
        if not np.isfinite(vix):
            continue

        daily_put_skew = np.interp(vix, [min_vix, max_vix], [min_skew, max_skew])
        daily_call_skew = -daily_put_skew / 4
        
        daily_vector_raw = np.array([vix, daily_put_skew, daily_call_skew])
        
        # <<< THE FIX: Ensure the daily_vector is a flat 1D array >>>
        # This guarantees its shape is (3,), which is broadcast-compatible with (6, 3).
        daily_vector = daily_vector_raw.flatten()
        
        # This subtraction will now work correctly.
        distances = np.linalg.norm(anchor_vectors - daily_vector, axis=1)
        labels.append(np.argmin(distances))

    # --- 4. Build the Transition Matrix ---
    print("Building the Transition Matrix...")
    num_regimes = len(IV_REGIMES)
    transition_matrix = np.zeros((num_regimes, num_regimes))
    
    for i in range(len(labels) - 1):
        today_regime = labels[i]
        tomorrow_regime = labels[i+1]
        transition_matrix[today_regime, tomorrow_regime] += 1
        
    # Normalize rows to get probabilities
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition_matrix, row_sums, where=row_sums!=0)
    
    # --- 5. Calculate the Stationary Distribution ---
    print("Calculating the Stationary Distribution...")
    # We solve P_transpose * pi = pi, which is finding the eigenvector for eigenvalue 1
    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    stationary_vector = np.real(eigenvectors[:, np.isclose(eigenvalues, 1)])
    stationary_distribution = stationary_vector / stationary_vector.sum()
    stationary_distribution = stationary_distribution.flatten()

    # --- 6. Save the Model ---
    save_path = "zoo/options_zero_game/experts/"
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "iv_transition_matrix.npy"), transition_matrix)
    np.save(os.path.join(save_path, "iv_stationary_distribution.npy"), stationary_distribution)

    print("\n--- IV Regime Analysis Complete! ---")
    print("Transition Matrix:\n", np.round(transition_matrix, 2))
    print("\nStationary Distribution:\n", np.round(stationary_distribution, 2))
    print(f"Models saved to '{save_path}'")

if __name__ == "__main__":
    analyze_iv_regimes()
