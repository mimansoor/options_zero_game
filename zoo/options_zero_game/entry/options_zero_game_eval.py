import numpy as np

# Import the official evaluation entry point
from lzero.entry import eval_muzero

# Import our game's configuration files
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config, create_config

# We must also import our environment files so the registry can find them.
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

if __name__ == "__main__":
    # ==============================================================
    # 1. Set the path to your best trained model checkpoint
    # ==============================================================
    model_path = './best_ckpt/ckpt_best.pth.tar' # UPDATE THIS PATH

    # ==============================================================
    # 2. Configure the environment for logging
    # ==============================================================
    
    create_config.env.type = 'log_replay'
    
    # <<< THE FIX: Save the log file directly to the 'build' directory
    main_config.env.log_file_path = 'visualizer-ui/build/replay_log.json'

    main_config.env.market_regimes = [
        # Name, mu, omega, alpha, beta, overnight_vol_multiplier
        #{'name': 'Bond_Markets', 'mu': 0.00001, 'omega': 0.000002, 'alpha': 0.05, 'beta': 0.92, 'overnight_vol_multiplier': 1.1},
        #{'name': 'Utilities_Sector', 'mu': 0.00004, 'omega': 0.00001, 'alpha': 0.08, 'beta': 0.90, 'overnight_vol_multiplier': 1.2},
        #{'name': 'Developed_Markets', 'mu': 0.00006, 'omega': 0.000005, 'alpha': 0.08, 'beta': 0.90, 'overnight_vol_multiplier': 1.3},
        #{'name': 'Foreign_Exchange_FX', 'mu': 0.0000, 'omega': 0.000003, 'alpha': 0.08, 'beta': 0.91, 'overnight_vol_multiplier': 1.4},
        #{'name': 'Tech_Sector', 'mu': 0.0001, 'omega': 0.00004, 'alpha': 0.11, 'beta': 0.87, 'overnight_vol_multiplier': 1.5},
        #{'name': 'Individual_Stocks', 'mu': 0.00008, 'omega': 0.00007, 'alpha': 0.10, 'beta': 0.88, 'overnight_vol_multiplier': 1.6},
        #{'name': 'Emerging_Markets', 'mu': 0.0001, 'omega': 0.00005, 'alpha': 0.12, 'beta': 0.86, 'overnight_vol_multiplier': 1.7},
        #{'name': 'Bull_LowVol', 'mu': 0.0005, 'omega': 0.000002, 'alpha': 0.05, 'beta': 0.94, 'overnight_vol_multiplier': 1.4},
        #{'name': 'Crisis_HighVol', 'mu': -0.0005, 'omega': 0.0001, 'alpha': 0.15, 'beta': 0.82, 'overnight_vol_multiplier': 2.2},
        #{'name': 'Frontier_Markets', 'mu': 0.0002, 'omega': 0.0001, 'alpha': 0.20, 'beta': 0.70, 'overnight_vol_multiplier': 1.8},
        #{'name': 'Commodities_Oil', 'mu': 0.0000, 'omega': 0.0002, 'alpha': 0.28, 'beta': 0.70, 'overnight_vol_multiplier': 1.9},
        #{'name': 'Volatility_VIX', 'mu': 0.0, 'omega': 0.0005, 'alpha': 0.25, 'beta': 0.65, 'overnight_vol_multiplier': 2.0},
        #{'name': 'Cryptocurrencies', 'mu': 0.001, 'omega': 0.001, 'alpha': 0.20, 'beta': 0.75, 'overnight_vol_multiplier': 2.5},
        #{'name': 'TSLA_Real', 'mu': 0.001070, 'omega': 0.00003585, 'alpha': 0.0350, 'beta': 0.9418, 'overnight_vol_multiplier': 6.58},
        #{'name': 'SPY_Real', 'mu': 0.000905, 'omega': 0.00000397, 'alpha': 0.1280, 'beta': 0.8404, 'overnight_vol_multiplier': 5.99},
        #{'name': 'RELIANCE.NS_Real', 'mu': 0.000352, 'omega': 0.00000267, 'alpha': 0.0207, 'beta': 0.9666, 'overnight_vol_multiplier': 5.14},
        {'name': '^NSEI_Real', 'mu': 0.000703, 'omega': 0.00000308, 'alpha': 0.0928, 'beta': 0.8733, 'overnight_vol_multiplier': 5.40},
        #{'name': 'TCS.NS_Real', 'mu': 0.000261, 'omega': 0.00002354, 'alpha': 0.0378, 'beta': 0.8327, 'overnight_vol_multiplier': 5.53},
        #{'name': 'ABB.NS_Real', 'mu': 0.001221, 'omega': 0.00025891, 'alpha': 0.1361, 'beta': 0.2717, 'overnight_vol_multiplier': 3.37},
        #{'name': 'BTC-USD_Real', 'mu': 0.001557, 'omega': 0.00002610, 'alpha': 0.0662, 'beta': 0.9083, 'overnight_vol_multiplier': 1.85},
        #{'name': 'INTC_Real', 'mu': -0.000764, 'omega': 0.00033938, 'alpha': 0.3683, 'beta': 0.2445, 'overnight_vol_multiplier': 6.33},
        #{'name': '^DJI_Real', 'mu': 0.000561, 'omega': 0.00000490, 'alpha': 0.1261, 'beta': 0.8208, 'overnight_vol_multiplier': 6.04},
        #{'name': '^IXIC_Real', 'mu': 0.000983, 'omega': 0.00000426, 'alpha': 0.1053, 'beta': 0.8768, 'overnight_vol_multiplier': 6.14},
        #{'name': '^RUT_Real', 'mu': 0.000419, 'omega': 0.00001442, 'alpha': 0.0707, 'beta': 0.8612, 'overnight_vol_multiplier': 6.70},
        #{'name': 'GC=F_Real', 'mu': 0.000309, 'omega': 0.00000454, 'alpha': 0.0361, 'beta': 0.9160, 'overnight_vol_multiplier': 2.02},
        #{'name': 'TLT_Real', 'mu': -0.000399, 'omega': 0.00000113, 'alpha': 0.0301, 'beta': 0.9592, 'overnight_vol_multiplier': 6.58},
        #{'name': 'BND_Real', 'mu': -0.000033, 'omega': 0.00000004, 'alpha': 0.0292, 'beta': 0.9687, 'overnight_vol_multiplier': 6.87},
        #{'name': 'ETH-USD_Real', 'mu': 0.001368, 'omega': 0.00002192, 'alpha': 0.0539, 'beta': 0.9345, 'overnight_vol_multiplier': 1.40},
        #{'name': 'SOL-USD_Real', 'mu': 0.001677, 'omega': 0.00013563, 'alpha': 0.1267, 'beta': 0.8455, 'overnight_vol_multiplier': 1.10},
    ]

    # ==============================================================
    # 3. Configure the evaluation run for a single episode
    # ==============================================================
    main_config.env.evaluator_env_num = 1
    main_config.env.n_evaluator_episode = 1
    create_config.env_manager.type = 'base'

    print("Starting evaluation run to generate replay log...")
    
    returns_mean, returns = eval_muzero(
        [main_config, create_config],
        seed=50,
        num_episodes_each_seed=1,
        print_seed_details=True,
        model_path=model_path
    )
    
    print(f"\nEvaluation run complete. Final Episode PnL: {returns[0][0]:.2f}")
    print(f"The replay_log.json should now be in the visualizer-ui/build/ directory.")
