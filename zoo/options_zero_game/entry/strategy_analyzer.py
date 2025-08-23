import argparse
import copy
from tqdm import tqdm
import json
import numpy as np

from lzero.entry import eval_muzero
from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config, create_config
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

# --- Calculation functions are now only needed by the orchestrator, but we keep them here ---
# --- so the orchestrator can import them. This script does not call them directly. ---

def get_valid_strategies() -> list:
    temp_env_cfg = copy.deepcopy(main_config.env)
    temp_env = zoo.options_zero_game.envs.options_zero_game_env.OptionsZeroGameEnv(cfg=temp_env_cfg)
    return [name for name in temp_env.actions_to_indices if name.startswith('OPEN_')]

def calculate_statistics(results: list, strategy_name: str) -> dict:
    if not results: return {}
    pnls = [r['pnl'] for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    num_wins, num_losses = len(wins), len(losses)
    win_rate = (num_wins / len(pnls)) * 100 if pnls else 0.0
    avg_profit = sum(wins) / num_wins if num_wins > 0 else 0.0
    avg_loss = sum(losses) / num_losses if num_losses > 0 else 0.0
    expectancy = ((win_rate / 100) * avg_profit) + ((1 - win_rate / 100) * avg_loss)
    profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else None
    cvar_95 = 0.0
    if losses:
        pnl_array = np.array(pnls)
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
    return {"Strategy": strategy_name, "Total_Trades": len(pnls), "Win_Rate_%": win_rate, "Expectancy_$": expectancy, "Profit_Factor": profit_factor, "Avg_Win_$": avg_profit, "Avg_Loss_$": avg_loss, "Max_Win_$": max(wins) if wins else 0.0, "Max_Loss_$": min(losses) if losses else 0.0, "CVaR_95%_$": cvar_95, "Win_Streak": max_win_streak, "Loss_Streak": max_loss_streak}

def calculate_trader_score(strategy_data):
    expectancy = strategy_data.get("Expectancy_$", 0)
    if expectancy <= 0: return np.tanh(expectancy / 50000)
    profit_factor = strategy_data.get("Profit_Factor", 1.0) or 1.0
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

def calculate_elo_ratings(all_pnl_by_strategy: dict, k_factor=32, initial_rating=1200):
    elo_ratings = {strategy: initial_rating for strategy in all_pnl_by_strategy.keys()}
    strategies = list(all_pnl_by_strategy.keys())
    if not strategies: return {}
    num_episodes = len(list(all_pnl_by_strategy.values())[0])
    print(f"\n--- Starting Elo Tournament Simulation ({num_episodes} rounds) ---")
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
    args = parser.parse_args()

    all_episode_pnls = []
    for i in range(args.episodes): # Use a simple range, tqdm is now in the orchestrator
        current_main_config = copy.deepcopy(main_config)
        current_create_config = copy.deepcopy(create_config)
        
        current_main_config.exp_name = args.exp_name
        
        if args.profit_target_pct is not None: current_main_config.env.profit_target_pct = args.profit_target_pct
        if args.credit_tp_pct is not None: current_main_config.env.credit_strategy_take_profit_pct = args.credit_tp_pct
        if args.debit_tp_mult is not None: current_main_config.env.debit_strategy_take_profit_multiple = args.debit_tp_mult
        
        current_main_config.env.price_source = 'historical'
        current_main_config.env.forced_historical_symbol = None if args.symbol.upper() == 'ANY' else args.symbol
        current_main_config.env.forced_opening_strategy_name = args.strategy
        current_main_config.env.is_eval_mode = True
        current_main_config.env.n_evaluator_episode = 1
        current_main_config.env.evaluator_env_num = 1
        current_main_config.env.forced_episode_length = args.days
        
        current_seed = args.start_seed + i
        
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
