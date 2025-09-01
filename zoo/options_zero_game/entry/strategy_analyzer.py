import argparse
import copy
from tqdm import tqdm
import json
import numpy as np
import torch
import math

from lzero.entry import eval_muzero
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config, create_config
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

# --- Calculation functions are now only needed by the orchestrator, but we keep them here ---
# --- so the orchestrator can import them. This script does not call them directly. ---

def get_valid_strategies() -> list:
    """
    Returns a definitive, hardcoded list of all specific, low-level strategies
    that can be individually analyzed. This is now decoupled from the agent's
    high-level action space.
    """
    # This list is the single source of truth for the strategy analyzer.
    return [
        # Bullish Strategies
        'OPEN_BULL_PUT_SPREAD',
        'OPEN_SHORT_PUT_ATM',
        'OPEN_BULL_CALL_SPREAD',
        # Bearish Strategies
        'OPEN_BEAR_CALL_SPREAD',
        'OPEN_SHORT_CALL_ATM',
        'OPEN_BEAR_PUT_SPREAD',
        # Neutral Strategies
        'OPEN_SHORT_STRADDLE',
        'OPEN_SHORT_STRANGLE_DELTA_25',
        'OPEN_SHORT_STRANGLE_DELTA_30',
        'OPEN_SHORT_IRON_CONDOR',

        # <<< --- THE DEFINITIVE FIX (Part 3): Add the new strategies to the list --- >>>
        # Advanced Strategies
        'OPEN_JADE_LIZARD',
        'OPEN_REVERSE_JADE_LIZARD',
        'OPEN_BIG_LIZARD',
        'OPEN_REVERSE_BIG_LIZARD',
        'OPEN_PUT_RATIO_SPREAD',
        'OPEN_CALL_RATIO_SPREAD',
    ]

def calculate_statistics(results: list, strategy_name: str) -> dict:
    if not results: return {}
    pnls = [r['pnl'] for r in results]
    # If for some reason the pnls list is empty, return early.
    if not pnls:
        return {"Strategy": strategy_name, "Total_Trades": 0}
    pnl_array = np.array(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    num_wins, num_losses = len(wins), len(losses)
    win_rate = (num_wins / len(pnls)) * 100 if pnls else 0.0
    avg_profit = sum(wins) / num_wins if num_wins > 0 else 0.0
    avg_loss = sum(losses) / num_losses if num_losses > 0 else 0.0
    expectancy = ((win_rate / 100) * avg_profit) + ((1 - win_rate / 100) * avg_loss)

    temp_env_cfg = copy.deepcopy(main_config.env)
    temp_env = zoo.options_zero_game.envs.options_zero_game_env.OptionsZeroGameEnv(cfg=temp_env_cfg)

    # <<< --- THE DEFINITIVE FIX AT THE SOURCE --- >>>
    sum_of_wins = sum(wins)
    sum_of_losses = abs(sum(losses))
    if sum_of_wins > 0:
        if sum_of_losses > 1e-6:
            profit_factor = sum_of_wins / sum_of_losses
        else: # Wins but no losses
            profit_factor = 999.0 # Use a large number to represent infinity
    else: # No wins
        profit_factor = 0.0

    cvar_95 = 0.0
    if losses:
        var_95 = np.percentile(pnl_array, 5)
        tail_losses = pnl_array[pnl_array <= var_95]
        if len(tail_losses) > 0: cvar_95 = np.mean(tail_losses)
        else: cvar_95 = var_95

    max_win_streak, max_loss_streak, current_win_streak, current_loss_streak = 0, 0, 0, 0
    for pnl in pnls:
        if pnl > 0: current_win_streak += 1; current_loss_streak = 0
        else: current_loss_streak += 1; current_win_streak = 0
        max_win_streak = max(max_win_streak, current_win_streak)
        max_loss_streak = max(max_loss_streak, current_loss_streak)

    # <<< --- NEW: Calculate Cumulative PnL, Drawdown, and CORRECTED Kelly Ratio --- >>>
    
    # 1. Cumulative PnL
    cumulative_pnl = sum(pnls)

    # 2. Max Drawdown and Highest High
    highest_high = 0.0
    max_drawdown = 0.0
    equity_curve = 0.0
    for pnl in pnls:
        equity_curve += pnl
        highest_high = max(highest_high, equity_curve)
        drawdown = highest_high - equity_curve
        max_drawdown = max(max_drawdown, drawdown)

    # 3. Half Kelly Ratio (Corrected Implementation)
    half_kelly_ratio = 0.0
    # Check for a positive edge (avg_profit > 0) and that avg_loss is a meaningful number
    if avg_profit > 0 and abs(avg_loss) > 1e-6:
        win_percentage = win_rate / 100.0
        # b = win/loss ratio
        win_loss_ratio = avg_profit / abs(avg_loss)
        
        # F = p - (q / b)
        kelly_criterion = win_percentage - ((1 - win_percentage) / win_loss_ratio)
        
        # Calculate Half Kelly and then apply the user's 30% maximum risk cap.
        calculated_half_kelly = max(0, kelly_criterion / 2.0) * 100 # Expressed as a percentage
        half_kelly_ratio = min(30.0, calculated_half_kelly)

    # <<< --- NEW: Calculation of Advanced Performance Ratios --- >>>
    # Define the annual risk-free rate
    ANNUAL_RISK_FREE_RATE = 0.10 # 10%

    # We need to convert the annual rate to a per-episode rate.
    # The analyzer runs for a fixed 20 trading days per episode.
    TRADING_DAYS_PER_YEAR = 252
    EPISODE_LENGTH_DAYS = 20
    risk_free_return_per_episode = (1 + ANNUAL_RISK_FREE_RATE)**(EPISODE_LENGTH_DAYS / TRADING_DAYS_PER_YEAR) - 1
    
    # The risk-free P&L is based on the initial capital. Using the config default.
    initial_capital = temp_env._cfg.initial_cash
    risk_free_pnl_per_episode = initial_capital * risk_free_return_per_episode

    # 1. Sharpe Ratio
    sharpe_ratio = 0.0
    # Calculate excess returns over the risk-free rate
    excess_returns = pnl_array - risk_free_pnl_per_episode
    std_dev_excess_returns = np.std(excess_returns)
    if std_dev_excess_returns > 1e-6:
        sharpe_ratio = np.mean(excess_returns) / std_dev_excess_returns

    # 2. Sortino Ratio
    sortino_ratio = 0.0
    # Downside deviation is calculated only on the excess returns that are negative.
    negative_excess_returns = excess_returns[excess_returns < 0]
    if len(negative_excess_returns) > 1:
        downside_deviation = np.std(negative_excess_returns)
        if downside_deviation > 1e-6:
            sortino_ratio = np.mean(excess_returns) / downside_deviation
        elif np.mean(excess_returns) > 0:
            sortino_ratio = 999.0 
    elif np.mean(excess_returns) > 0:
        sortino_ratio = 999.0

    # 3. Calmar Ratio
    calmar_ratio = 0.0
    num_episodes = len(pnls)
    total_days = num_episodes * EPISODE_LENGTH_DAYS
    total_years = total_days / TRADING_DAYS_PER_YEAR
    if total_years > 0 and max_drawdown > 1e-6:
        # Annualized return must also be in excess of the risk-free rate
        annualized_total_return = (cumulative_pnl / initial_capital) / total_years
        annualized_excess_return = annualized_total_return - ANNUAL_RISK_FREE_RATE
        calmar_ratio = annualized_excess_return / (max_drawdown / initial_capital)

    # ... (RoMaD and the rest of the function are unchanged) ...
    romad = 0.0
    if max_drawdown > 1e-6:
        romad = cumulative_pnl / max_drawdown
    elif cumulative_pnl > 0:
        romad = 999.0

    stats = {
        "Strategy": strategy_name,
        "Total_Trades": len(pnls),
        "Cumulative_PnL_$": cumulative_pnl,
        "Win_Rate_%": win_rate,
        "Expectancy_$": expectancy,
        "Profit_Factor": profit_factor,
        "Sharpe_Ratio": sharpe_ratio,       # <-- NEW
        "Sortino_Ratio": sortino_ratio,     # <-- NEW
        "Calmar_Ratio": calmar_ratio,       # <-- NEW
        "RoMaD": romad,                     # <-- NEW
        "Avg_Win_$": avg_profit,
        "Avg_Loss_$": avg_loss,
        "Max_Win_$": max(wins) if wins else 0.0,
        "Max_Loss_$": min(losses) if losses else 0.0,
        "Highest_High_$": highest_high,
        "Max_Drawdown_$": max_drawdown,
        "CVaR_95%_$": cvar_95,
        "Win_Streak": max_win_streak,
        "Loss_Streak": max_loss_streak,
        "Half_Kelly_%": half_kelly_ratio,
    }

    # Final sanitation pass to ensure all values are JSON-compatible
    for key, value in stats.items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            stats[key] = None # Convert NaN/inf to None for JSON null

    return stats

def calculate_trader_score(strategy_data):
    expectancy = strategy_data.get("Expectancy_$", 0)
    if expectancy <= 0: return np.tanh(expectancy / 50000)
    profit_factor = strategy_data.get("Profit_Factor", 1.0) or 1.0
    if profit_factor is None:
        profit_factor = 999.0 # Treat 'None' (infinite) as a very high number
    win_rate = strategy_data.get("Win_Rate_%", 0)
    avg_loss = 1 + abs(strategy_data.get("Avg_Loss_$", 0))
    max_loss = 1 + abs(strategy_data.get("Max_Loss_$", 0))
    cvar_95_raw = strategy_data.get("CVaR_95%_$", 0)
    cvar_risk_component = 1 + abs(min(0, cvar_95_raw))
    good_product = expectancy * win_rate * profit_factor
    bad_product = avg_loss + max_loss + cvar_risk_component
    raw_score = good_product / (bad_product + 1e-6)
    log_scaled_score = np.log(1 + raw_score)
    final_score = np.tanh(log_scaled_score)
    return final_score if np.isfinite(final_score) else 0.0

def calculate_elo_ratings(all_pnl_by_strategy: dict, existing_ratings: dict, k_factor=32, initial_rating=1200):
    """
    Calculates Elo ratings for all strategies by simulating a round-robin tournament.
    This version is robust to strategies having different numbers of completed episodes.
    """
    strategies = list(all_pnl_by_strategy.keys())
    if not strategies: return {}

    # --- 1. Initialize ratings for all strategies ---
    elo_ratings = {strategy: existing_ratings.get(strategy, initial_rating) for strategy in strategies}

    # The robust solution is to find the MINIMUM number of completed episodes
    # across all strategies and run the tournament for that many rounds.
    try:
        pnl_lengths = [len(pnl_list) for pnl_list in all_pnl_by_strategy.values() if pnl_list]
        if not pnl_lengths: # Failsafe for empty data
            return elo_ratings
        num_episodes = min(pnl_lengths)
    except Exception: # Catch any other potential errors
        print("Warning: Could not determine a valid number of episodes for Elo tournament. Aborting Elo calculation.")
        return elo_ratings

    print(f"\n--- Starting Elo Tournament Simulation ({num_episodes} rounds) ---")

    # --- 2. The Tournament Simulation Loop (Unchanged) ---
    for i in tqdm(range(num_episodes), desc="Elo Rounds"):
        for j in range(len(strategies)):
            for k in range(j + 1, len(strategies)):
                player_a, player_b = strategies[j], strategies[k]
                pnl_a, pnl_b = all_pnl_by_strategy[player_a][i], all_pnl_by_strategy[player_b][i]

                actual_score_a = 0.5
                if pnl_a > pnl_b: actual_score_a = 1.0
                elif pnl_b > pnl_a: actual_score_a = 0.0

                rating_a, rating_b = elo_ratings[player_a], elo_ratings[player_b]
                expected_score_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
                expected_score_b = 1 - expected_score_a

                elo_ratings[player_a] += k_factor * (actual_score_a - expected_score_a)
                elo_ratings[player_b] += k_factor * ((1 - actual_score_a) - expected_score_b)

    return {k: round(v) for k, v in elo_ratings.items()}

def calculate_pnl_weighted_elo(all_pnl_by_strategy: dict, existing_ratings: dict, k_factor=32, initial_rating=1200, pnl_weight=0.5, base_pnl_unit=10000.0):
    """
    Calculates Elo ratings where the points exchanged are weighted by the
    log-scaled average PNL outperformance between the two strategies.
    """
    strategies = list(all_pnl_by_strategy.keys())
    if not strategies: return {}

    elo_ratings = {strategy: existing_ratings.get(strategy, initial_rating) for strategy in strategies}

    # <<< --- THE DEFINITIVE, ROBUST FIX IS HERE --- >>>
    # This block ensures we only compare strategies over the number of episodes
    # that ALL of them have successfully completed.
    try:
        pnl_lengths = [len(pnl_list) for pnl_list in all_pnl_by_strategy.values() if pnl_list]
        if not pnl_lengths: return elo_ratings
        num_episodes = min(pnl_lengths)
    except Exception:
        return elo_ratings

    if num_episodes == 0: return elo_ratings

    print(f"\n--- Starting PnL-Weighted Elo Calculation ---")

    for i in tqdm(range(len(strategies)), desc="PnL Elo Matchups"):
        for j in range(i + 1, len(strategies)):
            player_a, player_b = strategies[i], strategies[j]

            # 1. Get the PnL lists, truncated to the common number of episodes.
            pnl_a = all_pnl_by_strategy[player_a][:num_episodes]
            pnl_b = all_pnl_by_strategy[player_b][:num_episodes]

            # 2. Calculate Average PNL Outperformance directly.
            #    The old code had a buggy list comprehension. This is the correct way.
            avg_outperformance = np.mean(pnl_a) - np.mean(pnl_b)

            # 3. Determine Win/Loss/Draw
            actual_score_a = 0.5
            if avg_outperformance > 0: actual_score_a = 1.0
            elif avg_outperformance < 0: actual_score_a = 0.0

            # 4. Calculate Log-Scaled Multiplier based on the magnitude of outperformance.
            pnl_diff_abs = abs(avg_outperformance)
            multiplier = math.log(1 + pnl_diff_abs / base_pnl_unit)

            # 5. Update Elo Ratings using a dynamic K-Factor (Unchanged)
            dynamic_k = k_factor * (1 + pnl_weight * multiplier)
            rating_a, rating_b = elo_ratings[player_a], elo_ratings[player_b]
            expected_score_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
            elo_ratings[player_a] += dynamic_k * (actual_score_a - expected_score_a)
            elo_ratings[player_b] += dynamic_k * ((1 - actual_score_a) - (1 - expected_score_a))

    return {k: round(v) for k, v in elo_ratings.items()}

if __name__ == "__main__":
    valid_strategies = get_valid_strategies()
    
    parser = argparse.ArgumentParser(description="Worker script for analyzing a single options trading strategy.")
    parser.add_argument('--strategy', type=str, required=True, choices=valid_strategies, help="The specific opening strategy to test.")
    parser.add_argument('--report_file', type=str, required=True, help="The JSON file to append the raw PNL data to.")
    parser.add_argument('-n', '--episodes', type=int, default=10, help="The number of episodes to run.")
    parser.add_argument('--start_seed', type=int, default=0, help="The starting seed for the episode sequence.")
    parser.add_argument('--model_path', type=str, default='./best_ckpt/ckpt_best.pth.tar')
    parser.add_argument('--symbol', type=str, default='ANY')
    parser.add_argument('--days', type=int, default=20)
    parser.add_argument('--profit_target_pct', type=float, default=None)
    parser.add_argument('--credit_tp_pct', type=float, default=None)
    parser.add_argument('--debit_tp_mult', type=float, default=None)
    parser.add_argument('--exp_name', type=str, default='eval/strategy_analyzer_runs', help="The experiment name prefix for temporary log files.")
    # <<< --- NEW: Add the deterministic flag --- >>>
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help="Enables all determinism flags for a reproducible run."
    )
    args = parser.parse_args()

    all_episode_pnls = []
    for i in range(args.episodes): # Use a simple range, tqdm is now in the orchestrator
        current_main_config = copy.deepcopy(main_config)
        current_create_config = copy.deepcopy(create_config)
        
        current_main_config.exp_name = args.exp_name
        current_seed = args.start_seed + i
        
        if args.profit_target_pct is not None: current_main_config.env.profit_target_pct = args.profit_target_pct
        if args.credit_tp_pct is not None: current_main_config.env.credit_strategy_take_profit_pct = args.credit_tp_pct
        if args.debit_tp_mult is not None: current_main_config.env.debit_strategy_take_profit_multiple = args.debit_tp_mult

        # <<< --- NEW: Apply all determinism settings if the flag is set --- >>>
        if args.deterministic:
            # 1. Set all PyTorch and CUDA seeds and flags
            torch.manual_seed(current_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(current_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # 2. Disable the C++ MCTS to use the pure Python version
            if 'policy' not in current_main_config:
                current_main_config.policy = {}
            current_main_config.policy.mcts_ctree = False
        
        current_main_config.env.price_source = 'historical'
        current_main_config.env.forced_historical_symbol = None if args.symbol.upper() == 'ANY' else args.symbol
        current_main_config.env.forced_opening_strategy_name = args.strategy
        current_main_config.env.is_eval_mode = True
        current_main_config.env.n_evaluator_episode = 1
        current_main_config.env.evaluator_env_num = 1
        current_main_config.env.forced_episode_length = args.days
        
        _, returns = eval_muzero(
            [current_main_config, create_config],
            seed=current_seed,
            num_episodes_each_seed=1,
            print_seed_details=False,
            model_path=args.model_path
        )
        
        if returns and len(returns[0]) > 0:
            all_episode_pnls.append(returns[0][0])
    
    strategy_pnl_data = {
        "strategy": args.strategy,
        "pnl_results": all_episode_pnls
    }

    try:
        with open(args.report_file, 'r') as f:
            report_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        report_data = []
    
    report_data.append(strategy_pnl_data)
    
    with open(args.report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
