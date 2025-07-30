import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

import zoo.options_zero_game.envs.options_zero_game_env
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config

def visualize_market_regimes():
    """
    This function generates and visualizes GARCH price paths for all defined market regimes.
    It uses a two-pass approach to ensure all charts share the same Y-axis scale for
    accurate visual comparison of volatility.
    """
    print("--- Starting GARCH Price Path & Log Return Visualization ---")
    
    output_dir = "garch_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Charts will be saved in the '{output_dir}/' directory.")

    base_env_cfg = main_config.env
    cfg_for_make = copy.deepcopy(base_env_cfg)
    if 'manager' in cfg_for_make: del cfg_for_make.manager
    if 'collector_env_num' in cfg_for_make: del cfg_for_make.collector_env_num
    if 'evaluator_env_num' in cfg_for_make: del cfg_for_make.evaluator_env_num
    if 'n_evaluator_episode' in cfg_for_make: del cfg_for_make.n_evaluator_episode
    if 'env_id' in cfg_for_make: del cfg_for_make.env_id

    all_regimes = base_env_cfg.market_regimes
    regime_data = []

    # --- PASS 1: Generate all data and find global min/max values ---
    print("\n--- Pass 1: Generating data for all regimes... ---")
    global_min_price, global_max_price = float('inf'), float('-inf')
    global_min_return, global_max_return = float('inf'), float('-inf')

    for regime in all_regimes:
        env_cfg = copy.deepcopy(cfg_for_make)
        env_cfg.market_regimes = [regime]
        env = gym.make('OptionsZeroGame-v0', cfg=env_cfg)
        env.reset(seed=42)
        
        price_path = env.price_path
        log_returns = np.diff(np.log(price_path))
        
        # Store data for the second pass
        regime_data.append({'name': regime['name'], 'params': regime, 'price_path': price_path, 'log_returns': log_returns})
        
        # Update global bounds
        global_min_price = min(global_min_price, np.min(price_path))
        global_max_price = max(global_max_price, np.max(price_path))
        global_min_return = min(global_min_return, np.min(log_returns))
        global_max_return = max(global_max_return, np.max(log_returns))

    print("--- Pass 1 Complete. Global bounds calculated. ---")
    
    # Add some padding for better visualization
    price_padding = (global_max_price - global_min_price) * 0.05
    return_padding = (global_max_return - global_min_return) * 0.05

    # --- PASS 2: Render all charts using the same scale ---
    print("\n--- Pass 2: Rendering charts with normalized Y-axis... ---")
    for i, data in enumerate(regime_data):
        regime_name = data['name']
        regime_params = data['params']
        print(f"Rendering chart {i+1}/{len(regime_data)}: {regime_name}")

        price_path = data['price_path']
        log_returns = data['log_returns']
        steps = np.arange(len(price_path))
        return_steps = np.arange(len(log_returns))
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Price Path
        ax1.plot(steps, price_path, label='Simulated GARCH Price', color='deepskyblue', linewidth=1.5)
        ax1.set_ylabel("Price", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend()
        # <<< YOUR BRILLIANT FIX: Set a fixed Y-axis limit
        ax1.set_ylim(global_min_price - price_padding, global_max_price + price_padding)
        
        # Plot 2: Log Returns
        ax2.plot(return_steps, log_returns, label='Log Returns', color='coral', linewidth=1.0)
        ax2.set_ylabel("Log Return", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        # <<< YOUR BRILLIANT FIX: Set a fixed Y-axis limit
        ax2.set_ylim(global_min_return - return_padding, global_max_return + return_padding)
        
        # Overall Formatting
        title = f"Simulated GARCH Path for Market Regime: {regime_name}"
        subtitle = f"Parameters: μ={regime_params['mu']}, ω={regime_params['omega']}, α={regime_params['alpha']}, β={regime_params['beta']}"
        fig.suptitle(title + "\n" + subtitle, fontsize=16)
        plt.xlabel("Simulation Steps", fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        file_path = os.path.join(output_dir, f"regime_{i+1}_{regime_name}.png")
        plt.savefig(file_path)
        plt.close(fig)

    print("\n--- Price Path Visualization Test Finished ---")

if __name__ == "__main__":
    visualize_market_regimes()
