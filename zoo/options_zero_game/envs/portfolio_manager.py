# zoo/options_zero_game/envs/portfolio_manager.py
import pandas as pd
import numpy as np
import math
import traceback
from numba import jit
from typing import Dict, List, Any, Tuple, Callable
from .black_scholes_manager import BlackScholesManager, _numba_cdf, _numba_black_scholes
from .market_rules_manager import MarketRulesManager

# <<< --- NEW: THE CANONICAL NUMBA P&L CALCULATOR (INNER FUNCTION) --- >>>
@jit(nopython=True)
def _numba_get_pnl_for_leg(leg_data, price, at_expiry, lot_size, risk_free_rate, bid_ask_spread_pct, bs_func):
    """
    The single, canonical, high-performance function for calculating the P&L of one
    option leg at a specific underlying price.
    """
    # Unpack the leg data from the numpy array
    strike_price = leg_data[0]
    entry_premium = leg_data[1]
    is_call = leg_data[2] == 1
    pnl_multiplier = 1.0 if leg_data[3] == 1 else -1.0
    days_to_expiry = leg_data[5]

    current_value = 0.0
    if at_expiry:
        # --- Expiry P&L (Intrinsic Value) ---
        if is_call:
            current_value = max(0.0, price - strike_price)
        else: # Put
            current_value = max(0.0, strike_price - price)
    else:
        # --- T+0 P&L (Black-Scholes) ---
        # THE CRITICAL FIX: Convert days to annual time for Black-Scholes
        T_annual = days_to_expiry / 365.25
        # We can use a simplified vol for the simulation
        vol = 0.25 
        
        mid_price = bs_func(price, strike_price, T_annual, risk_free_rate, vol, is_call)
        
        # Apply bid-ask spread to get the closing price
        is_closing_a_short = pnl_multiplier == -1.0
        if is_closing_a_short: # We must buy to close
            current_value = mid_price * (1 + bid_ask_spread_pct)
        else: # We sell to close
            current_value = mid_price * (1 - bid_ask_spread_pct)

    pnl_per_share = current_value - entry_premium
    return pnl_per_share * pnl_multiplier * lot_size

# <<< --- NEW: THE NUMBA SIMULATION ENGINE (OUTER FUNCTION) --- >>>
@jit(nopython=True)
def _numba_run_pnl_simulation(legs_array, realized_pnl, start_price, lot_size, risk_free_rate, bid_ask_spread_pct):
    # ... (simulation setup is unchanged) ...
    sim_low = start_price * 0.25
    sim_high = start_price * 2.5
    price_range = np.linspace(sim_low, sim_high, 500)
    pnl_values = np.empty(len(price_range), dtype=np.float32)
    breakevens = []
    pnl_prev = 0.0
    for i in range(len(price_range)):
        price_curr = price_range[i]
        unrealized_pnl = 0.0
        for j in range(len(legs_array)):
            leg = legs_array[j]
            unrealized_pnl += _numba_get_pnl_for_leg(
                leg, price_curr, at_expiry=True, lot_size=lot_size,
                risk_free_rate=risk_free_rate, bid_ask_spread_pct=bid_ask_spread_pct,
                bs_func=_numba_black_scholes
            )
        total_pnl = unrealized_pnl + realized_pnl
        pnl_values[i] = total_pnl
        if i > 0:
            if np.sign(total_pnl) != np.sign(pnl_prev):
                price_prev = price_range[i-1]
                # Add a failsafe for division by zero in breakeven calculation
                denominator = total_pnl - pnl_prev
                if abs(denominator) > 1e-9:
                    be = price_prev - pnl_prev * (price_curr - price_prev) / denominator
                    # Ensure the calculated breakeven is a finite number
                    if np.isfinite(be):
                        breakevens.append(be)
        pnl_prev = total_pnl

    # --- THE DEFINITIVE FIX AT THE SOURCE ---
    # 1. Calculate max profit and loss. If the array is empty or all NaN, default to realized_pnl.
    if np.all(np.isnan(pnl_values)) or pnl_values.size == 0:
        max_profit = realized_pnl
        max_loss = realized_pnl
    else:
        max_profit = np.nanmax(pnl_values)
        max_loss = np.nanmin(pnl_values)

    # 2. Calculate profit factor with robust checks.
    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values <= 0]
    sum_wins = np.sum(wins)
    sum_losses = np.abs(np.sum(losses))

    profit_factor = 0.0 # Default for loss-only or flat P&L
    if sum_wins > 0:
        if sum_losses > 1e-6:
            profit_factor = sum_wins / sum_losses
        else:
            profit_factor = 999.0 # Effectively infinite if there are no losses

    # Ensure all return values are finite.
    max_profit = max_profit if np.isfinite(max_profit) else 0.0
    max_loss = max_loss if np.isfinite(max_loss) else 0.0

    return max_profit, max_loss, profit_factor, breakevens

class PortfolioManager:
    """
    Manages the state of the agent's portfolio, including all trades,
    PnL calculations, and constructing the portfolio observation state.
    This class is stateful and is the heart of the trading logic.
    """
    def __init__(self, cfg: Dict, bs_manager: BlackScholesManager, 
                 market_rules_manager: MarketRulesManager, 
                 iv_calculator_func: Callable,
                 strategy_name_to_id: Dict):
        # Config
        self.cfg = cfg
        self.initial_cash = cfg['initial_cash']
        self.lot_size = cfg['lot_size']
        self.bid_ask_spread_pct = cfg['bid_ask_spread_pct']
        self.max_positions = cfg['max_positions']
        self.start_price = cfg['start_price']
        self.strike_distance = cfg['strike_distance']
        self.undefined_risk_cap = self.initial_cash
        self.strategy_name_to_id = strategy_name_to_id
        self.close_short_leg_on_profit_threshold = cfg.get('close_short_leg_on_profit_threshold', 0.0)
        self.is_eval_mode = cfg.get('is_eval_mode', False)
        self.brokerage_per_leg = cfg.get('brokerage_per_leg', 0.0)
        self.max_strike_offset = cfg.get('max_strike_offset', 30)
        self.receipts_for_current_step: List[dict] = []
        self.steps_per_day = cfg.get('steps_per_day', 1)
        self.butterfly_target_cost_pct = cfg.get('butterfly_target_cost_pct', 0.01)
        self.hedge_roll_search_width = cfg.get('hedge_roll_search_width', 10)
        self.id_to_strategy_name = {v: k for k, v in self.strategy_name_to_id.items()}
        
        # Managers
        self.bs_manager = bs_manager
        self.market_rules_manager = market_rules_manager

        self.iv_calculator = iv_calculator_func

        # State variables
        self.portfolio = pd.DataFrame()
        self.realized_pnl: float = 0.0
        self.high_water_mark: float = 0.0
        self.next_creation_id: int = 0
        self.portfolio_columns = ['type', 'direction', 'entry_step', 'strike_price', 'entry_premium', 'days_to_expiry', 'creation_id', 'strategy_id', 'strategy_max_profit', 'strategy_max_loss', 'is_hedged']
        self.portfolio_dtypes = {'type': 'object', 'direction': 'object', 'entry_step': 'int16', 'strike_price': 'float32', 'entry_premium': 'float32', 'days_to_expiry': 'float32', 'creation_id': 'int16', 'strategy_id': 'int16', 'strategy_max_profit': 'float32', 'strategy_max_loss': 'float32', 'is_hedged': 'bool'}
        self.highest_realized_profit = 0.0
        self.lowest_realized_loss = 0.0
        self.mtm_pnl_high = 0.0  # Tracks the highest MtM P&L seen
        self.mtm_pnl_low = 0.0   # Tracks the lowest MtM P&L seen (max drawdown)

        # This guarantees a perfect, two-way symmetrical mapping for all actions.
        
        self.SYMMETRIC_ACTION_MAP = {}
        
        # --- Define the pairs of opposites ---
        action_pairs = [
            # Spreads
            ('OPEN_BULL_CALL_SPREAD', 'OPEN_BEAR_CALL_SPREAD'),
            ('OPEN_BULL_PUT_SPREAD', 'OPEN_BEAR_PUT_SPREAD'),
            # Volatility Strategies
            ('OPEN_LONG_STRADDLE', 'OPEN_SHORT_STRADDLE'),
            ('OPEN_LONG_IRON_CONDOR', 'OPEN_SHORT_IRON_CONDOR'),
            ('OPEN_LONG_IRON_FLY', 'OPEN_SHORT_IRON_FLY'),
        ]
        
        # --- Dynamically add pairs for strangles, butterflies, and naked legs ---
        # Delta Strangles
        for delta in [15, 20, 25, 30]:
            action_pairs.append(
                (f'OPEN_LONG_STRANGLE_DELTA_{delta}', f'OPEN_SHORT_STRANGLE_DELTA_{delta}')
            )
        # Fixed-Width Butterflies
        for width in [1, 2]:
            for opt_type in ['CALL', 'PUT']:
                action_pairs.append(
                    (f'OPEN_LONG_{opt_type}_FLY_{width}', f'OPEN_SHORT_{opt_type}_FLY_{width}')
                )
        # Naked Legs (from -10 to +10)
        for offset in range(-10, 11):
            strike_str = f"ATM{offset:+d}"
            for opt_type in ['CALL', 'PUT']:
                action_pairs.append(
                    (f'OPEN_LONG_{opt_type}_{strike_str}', f'OPEN_SHORT_{opt_type}_{strike_str}')
                )

        # --- Populate the final map with perfect two-way symmetry ---
        for action1, action2 in action_pairs:
            self.SYMMETRIC_ACTION_MAP[action1] = action2
            self.SYMMETRIC_ACTION_MAP[action2] = action1

        # This map connects strategies with the same directional bias but opposite
        # premium types (debit vs. credit), following the user's elegant logic.
        self.STRATEGY_TYPE_MAP = {}

        # Define the conceptual base pairings. The logic below will handle all variations.
        action_pairs = [
            # --- Directional Pairs ---
            ('OPEN_LONG_CALL_ATM', 'OPEN_SHORT_PUT_ATM'),
            ('OPEN_BULL_CALL_SPREAD', 'OPEN_BULL_PUT_SPREAD'),
            ('OPEN_LONG_PUT_ATM', 'OPEN_SHORT_CALL_ATM'),
            ('OPEN_BEAR_PUT_SPREAD', 'OPEN_BEAR_CALL_SPREAD'),

            # --- Volatility Pairs (Debit vs. Credit) ---
            ('OPEN_LONG_STRADDLE', 'OPEN_SHORT_IRON_FLY'),
            ('OPEN_LONG_STRANGLE_DELTA_', 'OPEN_SHORT_IRON_CONDOR'),
        ]

        # Programmatically build the full map from the base pairs
        for base1, base2 in action_pairs:
            # Handle naked legs with ATM offsets
            if 'ATM' in base1 and 'ATM' in base2:
                # Use the agent_max_open_offset from the config for consistency
                agent_max_open_offset = cfg.get('agent_max_open_offset', 2)
                for offset in range(-agent_max_open_offset, agent_max_open_offset + 1):
                    # Create the full action names
                    action1 = f"{base1}{offset:+d}"
                    action2 = f"{base2}{offset:+d}"
                    # Create the two-way mapping
                    self.STRATEGY_TYPE_MAP[action1] = action2
                    self.STRATEGY_TYPE_MAP[action2] = action1

            # Handle delta-based strangles vs. iron condors
            elif 'DELTA' in base1 and 'CONDOR' in base2:
                for delta in [15, 20, 25, 30]:
                    action1 = f"{base1}{delta}"
                    action2 = base2
                    self.STRATEGY_TYPE_MAP[action1] = action2
                    self.STRATEGY_TYPE_MAP[action2] = action1

            # Handle simple named strategies (spreads, straddles, etc.)
            else:
                self.STRATEGY_TYPE_MAP[base1] = base2
                self.STRATEGY_TYPE_MAP[base2] = base1

    def reset(self):
        """Resets the portfolio to an empty state for a new episode."""
        self.portfolio = pd.DataFrame(columns=self.portfolio_columns).astype(self.portfolio_dtypes)
        self.realized_pnl = 0.0
        self.high_water_mark = self.initial_cash
        self.next_creation_id = 0
        self.receipts_for_current_step = []

        # Reset the MtM trackers for the new episode
        self.mtm_pnl_high = 0.0
        self.mtm_pnl_low = 0.0
        self.highest_realized_profit = 0.0
        self.lowest_realized_loss = 0.0

    # --- Public Methods (called by the main environment) ---
    def hedge_portfolio_by_rolling_leg(self, leg_index: int, current_price: float, iv_bin_index: int, current_step: int):
        """
        Executes the portfolio delta hedge by finding the optimal strike for
        the specified leg and performing an atomic roll.
        """
        if not (0 <= leg_index < len(self.portfolio)): return

        new_strike = self._find_portfolio_neutral_strike(leg_index, current_price, iv_bin_index)
        
        if new_strike is None:
            return

        # --- 2. The Definitive, Atomic Roll Logic ---
        leg_to_roll_df = self.portfolio.iloc[[leg_index]]
        leg_to_roll_dict = leg_to_roll_df.iloc[0].to_dict()
        
        other_legs_df = self.portfolio.drop(self.portfolio.index[leg_index])
        
        self._process_leg_closures(leg_to_roll_df, current_price, current_step)
        
        # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
        # a) Create the new leg by copying the old one to preserve all essential keys.
        new_leg_def = leg_to_roll_dict.copy()
        new_leg_def['strike_price'] = new_strike
        new_leg_def['entry_step'] = current_step
        # Remove the old premium; it will be recalculated.
        if 'entry_premium' in new_leg_def: del new_leg_def['entry_premium']

        # b) Price the newly defined leg. _price_legs expects a list.
        priced_new_leg_list = self._price_legs([new_leg_def], current_price, iv_bin_index)

        if not priced_new_leg_list:
            self.portfolio = other_legs_df.reset_index(drop=True)
            self._update_hedge_status()
            return

        # c) Re-assemble the full portfolio and re-calculate its profile
        final_legs_list = other_legs_df.to_dict('records') + priced_new_leg_list
        pnl_profile = self._calculate_universal_risk_profile(final_legs_list, self.realized_pnl)
        
        # We must preserve the original strategy ID after the roll
        pnl_profile['strategy_id'] = leg_to_roll_dict['strategy_id']

        # d) Atomically replace the entire portfolio with the new, correct state
        for leg in final_legs_list:
            leg.update(pnl_profile)
        self.portfolio = pd.DataFrame(final_legs_list).astype(self.portfolio_dtypes)
        self._update_hedge_status()

    def get_raw_greeks_for_legs(self, legs: List[Dict], current_price: float, iv_bin_index: int) -> Dict:
        """
        A helper method for testing that calculates the raw, un-normalized, aggregated
        Greeks for an arbitrary list of leg dictionaries under specific market conditions.
        """
        total_delta, total_gamma, total_theta, total_vega = 0.0, 0.0, 0.0, 0.0
        if not legs:
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

        atm_price = self.market_rules_manager.get_atm_price(current_price)
        for pos in legs:
            direction_multiplier = 1 if pos['direction'] == 'long' else -1
            is_call = pos['type'] == 'call'
            offset = round((pos['strike_price'] - atm_price) / self.strike_distance)
            vol = self.iv_calculator(offset, pos['type'])
            
            greeks = self.bs_manager.get_all_greeks_and_price(
                current_price, pos['strike_price'], pos['days_to_expiry'], vol, is_call
            )
            
            total_delta += greeks['delta'] * self.lot_size * direction_multiplier
            total_gamma += greeks['gamma'] * (self.lot_size**2 * current_price**2 / 100) * direction_multiplier
            total_theta += greeks['theta'] * self.lot_size * direction_multiplier
            total_vega += greeks['vega'] * self.lot_size * direction_multiplier
            
        return {'delta': total_delta, 'gamma': total_gamma, 'theta': total_theta, 'vega': total_vega}

    def add_hedge(self, position_index: int, current_price: float, iv_bin_index: int, current_step: int):
        """
        Adds a protective leg to a specified un-hedged leg, correctly identifying
        the resulting strategy (e.g., a standard vertical spread or a custom position).
        """
        try:
            # --- 1. Failsafe Checks and Component Identification ---
            if not (0 <= position_index < len(self.portfolio)): return
            if len(self.portfolio) >= self.max_positions: return

            leg_to_hedge_df = self.portfolio.iloc[[position_index]]
            leg_to_hedge = leg_to_hedge_df.iloc[0]
            if leg_to_hedge['is_hedged']: return
            
            original_creation_id = leg_to_hedge['creation_id']
            untouched_legs = self.portfolio[self.portfolio['creation_id'] != original_creation_id]
            sibling_legs = self.portfolio[(self.portfolio['creation_id'] == original_creation_id) & (self.portfolio.index != position_index)]

            # --- 2. Determine and Create the Hedge Leg ---
            hedge_type = leg_to_hedge['type']
            days_to_expiry = leg_to_hedge['days_to_expiry']
            hedge_direction = 'short' if leg_to_hedge['direction'] == 'long' else 'long'
            
            naked_strike = leg_to_hedge['strike_price']
            is_itm = (hedge_type == 'call' and current_price > naked_strike) or (hedge_type == 'put' and current_price < naked_strike)
            if is_itm:
                hedge_strike = self.market_rules_manager.get_atm_price(current_price) + (self.strike_distance if hedge_type == 'call' else -self.strike_distance)
            else:
                breakeven_price = naked_strike + leg_to_hedge['entry_premium'] if hedge_type == 'call' else naked_strike - leg_to_hedge['entry_premium']
                closest_valid_strike = self.market_rules_manager.get_atm_price(breakeven_price)
                hedge_strike = closest_valid_strike if closest_valid_strike != naked_strike else (naked_strike + self.strike_distance if hedge_type == 'call' else naked_strike - self.strike_distance)

            if hedge_strike <= 0: return
            
            hedge_leg_def = [{'type': hedge_type, 'direction': hedge_direction, 'strike_price': hedge_strike, 'days_to_expiry': days_to_expiry, 'entry_step': current_step}]
            priced_hedge_leg = self._price_legs(hedge_leg_def, current_price, iv_bin_index)
            if not priced_hedge_leg: return

            # --- 3. Re-assemble and Intelligently Identify the New Strategy ---
            modified_strategy_legs_list = sibling_legs.to_dict('records') + leg_to_hedge_df.to_dict('records') + priced_hedge_leg
            
            # <<< --- THE DEFINITIVE LOGIC FIX --- >>>
            if len(modified_strategy_legs_list) == 2:
                # If the result is a 2-leg position, it's a standard vertical spread.
                temp_df = pd.DataFrame(modified_strategy_legs_list)
                new_strategy_name = self._identify_strategy_from_legs(temp_df)
            else:
                # If the result is a 3+ leg position, it's a custom hedged structure.
                new_strategy_name = f"CUSTOM_{len(modified_strategy_legs_list)}_LEGS"
            
            pnl_profile = self._calculate_universal_risk_profile(modified_strategy_legs_list, self.realized_pnl)
            pnl_profile['strategy_id'] = self.strategy_name_to_id.get(new_strategy_name, -1)
            
            # --- 4. Atomically rebuild the portfolio ---
            self.portfolio = untouched_legs.copy()
            self._execute_trades(modified_strategy_legs_list, pnl_profile)
            
        except (ValueError, IndexError, KeyError) as e:
            print(f"Warning: Could not execute add_hedge action. Error: {e}")

    def open_best_available_vertical(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """
        Public method to execute the opening of a vertical spread.
        It first tries to find an ideal spread using the tiered R:R solver.
        If that fails, it falls back to creating a robust, fixed-width spread.
        """
        # --- 1. Parse Action Intent ---
        parts = action_name.replace('OPEN_', '').split('_')
        direction_name = '_'.join(parts[0:2])
        option_type = parts[1].lower()
        is_credit_spread = 'BULL_PUT' in direction_name or 'BEAR_CALL' in direction_name

        # <<< --- THE DEFINITIVE FIX: The solver logic is now at the top level --- >>>
        # --- 2. Attempt to Find the Ideal Spread ---
        found_legs = self._find_best_available_spread(
            option_type, direction_name, current_price, iv_bin_index, days_to_expiry, is_credit_spread
        )

        # --- 3. Robust Fallback (only runs if the solver fails) ---
        if not found_legs:
            # print(f"DEBUG: Tiered R:R solver failed for {action_name}. Using fixed-width fallback.")
            
            atm_strike = self.market_rules_manager.get_atm_price(current_price)
            wing_offset = self.strike_distance * 2
            anchor_direction = 'short' if is_credit_spread else 'long'
            wing_direction = 'long' if is_credit_spread else 'short'
            
            if direction_name == 'BULL_CALL_SPREAD':
                anchor_strike = atm_strike
                wing_strike = atm_strike + wing_offset
            elif direction_name == 'BEAR_CALL_SPREAD':
                anchor_strike = atm_strike
                wing_strike = atm_strike + wing_offset
            elif direction_name == 'BULL_PUT_SPREAD':
                anchor_strike = atm_strike
                wing_strike = atm_strike - wing_offset
            else: # BEAR_PUT_SPREAD
                anchor_strike = atm_strike
                wing_strike = atm_strike - wing_offset # A Bear Put is SHORT the higher strike, so this is correct.

            fallback_legs_def = [
                {'type': option_type, 'direction': anchor_direction, 'strike_price': anchor_strike},
                {'type': option_type, 'direction': wing_direction, 'strike_price': wing_strike}
            ]
            for leg in fallback_legs_def: leg['days_to_expiry'] = days_to_expiry

            found_legs = self._price_legs(fallback_legs_def, current_price, iv_bin_index, check_short_rule=is_credit_spread)

        # --- 4. Finalize and Execute the Trade ---
        if found_legs:
            for leg in found_legs:
                leg['entry_step'] = current_step

            pnl_profile = self._calculate_universal_risk_profile(found_legs, self.realized_pnl)
            pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
            self._execute_trades(found_legs, pnl_profile)

    def update_mtm_water_marks(self, current_total_pnl: float):
        """
        Updates the high-water mark and low-water mark (max drawdown) for the
        total Mark-to-Market P&L of the portfolio.
        This should be called EVERY step by the environment.
        """
        #current_total_pnl includes initial_cash
        self.mtm_pnl_high = max(self.mtm_pnl_high, (current_total_pnl-self.initial_cash))
        self.mtm_pnl_low = min(self.mtm_pnl_low, (current_total_pnl-self.initial_cash))
        self.high_water_mark = max(self.high_water_mark, current_total_pnl)

    def get_raw_portfolio_stats(self, current_price: float, iv_bin_index: int) -> dict:
        """
        Calculates and returns a dictionary of the key, un-normalized portfolio
        statistics for logging and visualization.
        """
        # This first check is the primary guard clause for an empty portfolio.
        if self.portfolio.empty:
            return {
                'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0,
                'max_profit': 0.0, 'max_loss': 0.0, 'rr_ratio': 0.0,
                'prob_profit': 0.0, 'profit_factor': 0.0,
                'mtm_pnl_high': self.mtm_pnl_high, 'mtm_pnl_low': self.mtm_pnl_low,
                'highest_realized_profit': self.highest_realized_profit,
                'lowest_realized_loss': self.lowest_realized_loss,
                'net_premium': 0.0,
                'breakevens': []
            }

        summary = self.get_portfolio_summary(current_price, iv_bin_index)

        total_delta, total_gamma, total_theta, total_vega = 0.0, 0.0, 0.0, 0.0
        current_net_premium = 0.0
        
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        for _, pos in self.portfolio.iterrows():
            direction_multiplier = 1 if pos['direction'] == 'long' else -1
            is_call = pos['type'] == 'call'
            offset = round((pos['strike_price'] - atm_price) / self.strike_distance)
            vol = self.iv_calculator(offset, pos['type'])
            leg_greeks = self.bs_manager.get_all_greeks_and_price(current_price, pos['strike_price'], pos['days_to_expiry'], vol, is_call)

            total_delta += leg_greeks['delta'] * self.lot_size * direction_multiplier
            total_gamma += leg_greeks['gamma'] * self.lot_size * direction_multiplier
            total_theta += leg_greeks['theta'] * self.lot_size * direction_multiplier
            total_vega += leg_greeks['vega'] * self.lot_size * direction_multiplier

            current_premium = self.bs_manager.get_price_with_spread(
                leg_greeks['price'], is_buy=(pos['direction'] == 'short'), bid_ask_spread_pct=self.bid_ask_spread_pct
            )
            current_net_premium += current_premium * direction_multiplier

        final_net_premium = current_net_premium * self.lot_size
        legs_from_portfolio = self.portfolio.to_dict(orient='records')
        pnl_profile  = self._calculate_universal_risk_profile(legs_from_portfolio, self.realized_pnl)

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
            'max_profit': summary['max_profit'],
            'max_loss': summary['max_loss'],
            'rr_ratio': summary['rr_ratio'],
            'prob_profit': summary['prob_profit'],
            'breakevens': pnl_profile['breakevens'],
            'profit_factor': pnl_profile['profit_factor'],
            'highest_realized_profit': self.highest_realized_profit,
            'lowest_realized_loss': self.lowest_realized_loss,
            'mtm_pnl_high': self.mtm_pnl_high,
            'mtm_pnl_low': self.mtm_pnl_low,
            'net_premium': final_net_premium,
        }

    def get_portfolio(self) -> pd.DataFrame:
        """Public API to get the portfolio state."""
        return self.portfolio

    def open_strategy(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """Routes any 'OPEN_' action to the correct specialized private method."""
        if len(self.portfolio) >= self.max_positions: return

        disable_solver = self.cfg.get('disable_spread_solver', False)

        # Route by most complex/specific keywords first to avoid misrouting
        if 'JADE_LIZARD' in action_name:
            self._open_jade_lizard(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'BIG_LIZARD' in action_name:
            self._open_big_lizard(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'RATIO_SPREAD' in action_name:
            self._open_ratio_spread(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'SPREAD' in action_name and not disable_solver:
            self.open_best_available_vertical(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'CONDOR' in action_name and 'IRON' not in action_name:
            self._open_condor(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'FLY' in action_name and 'IRON' not in action_name:
            self._open_butterfly(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'IRON_CONDOR' in action_name:
            self._open_iron_condor(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'IRON_FLY' in action_name:
            self._open_iron_fly(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'STRANGLE' in action_name:
            self._open_strangle(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'STRADDLE' in action_name:
            self._open_straddle(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'ATM' in action_name:
            self._open_single_leg(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        else:
            print(f"Warning: Unrecognized action format in open_strategy: {action_name}")

    def get_positions_state(self, vec: np.ndarray, start_idx: int, state_size: int, pos_idx_map: Dict, current_price: float, iv_bin_index: int, current_step: int, total_steps: int):
        """Fills the provided numpy vector `vec` with the detailed state of each open position."""
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        
        for i, pos in self.portfolio.iterrows():
            if i >= self.max_positions: break

            is_call = pos['type'] == 'call'
            direction_multiplier = 1.0 if pos['direction'] == 'long' else -1.0
            
            offset = round((pos['strike_price'] - atm_price) / self.strike_distance)
            vol = self.iv_calculator(offset, pos['type'])
            greeks = self.bs_manager.get_all_greeks_and_price(current_price, pos['strike_price'], pos['days_to_expiry'], vol, is_call)
            
            current_pos_idx = start_idx + i * state_size
            vec[current_pos_idx + pos_idx_map['IS_OCCUPIED']] = 1.0
            vec[current_pos_idx + pos_idx_map['TYPE_NORM']] = 1.0 if is_call else -1.0
            vec[current_pos_idx + pos_idx_map['DIRECTION_NORM']] = direction_multiplier
            vec[current_pos_idx + pos_idx_map['STRIKE_DIST_NORM']] = (pos['strike_price'] - atm_price) / (self.max_strike_offset * self.strike_distance)
            vec[current_pos_idx + pos_idx_map['DAYS_HELD_NORM']] = (current_step - pos.get('entry_step', current_step)) / total_steps

            # --- THE FIX IS HERE ---
            # Call the directly imported _numba_cdf function instead of through the manager instance
            if is_call: pop = _numba_cdf(greeks['d2']) if pos['direction'] == 'long' else 1 - _numba_cdf(greeks['d2'])
            else: pop = 1 - _numba_cdf(greeks['d2']) if pos['direction'] == 'long' else _numba_cdf(greeks['d2'])
            vec[current_pos_idx + pos_idx_map['PROB_OF_PROFIT']] = pop if math.isfinite(pop) else 0.5
            
            vec[current_pos_idx + pos_idx_map['MAX_PROFIT_NORM']] = math.tanh(pos['strategy_max_profit'] / self.initial_cash)
            vec[current_pos_idx + pos_idx_map['MAX_LOSS_NORM']] = math.tanh(pos['strategy_max_loss'] / self.initial_cash)
            # We now provide the true risk contribution of each leg to the agent.
            
            # Delta is not normalized, just signed.
            vec[current_pos_idx + pos_idx_map['DELTA']] = greeks['delta'] * direction_multiplier
            
            # For other greeks, we sign them *before* normalization.
            signed_gamma = greeks['gamma'] * self.lot_size * direction_multiplier
            signed_theta = greeks['theta'] * self.lot_size * direction_multiplier
            signed_vega = greeks['vega'] * self.lot_size * direction_multiplier
            
            vec[current_pos_idx + pos_idx_map['GAMMA']] = math.tanh(signed_gamma)
            vec[current_pos_idx + pos_idx_map['THETA']] = math.tanh(signed_theta)
            vec[current_pos_idx + pos_idx_map['VEGA']] = math.tanh(signed_vega)
            # Convert the boolean to a float for the neural network.
            vec[current_pos_idx + pos_idx_map['IS_HEDGED']] = 1.0 if pos['is_hedged'] else 0.0
           
    def get_total_pnl(self, current_price: float, iv_bin_index: int) -> float:
        """
        Calculates the total Mark-to-Market P&L of the portfolio.
        """
        if self.portfolio.empty:
            return self.realized_pnl

        unrealized_pnl = sum(
            self._get_t0_pnl_for_leg(leg, current_price)
            for _, leg in self.portfolio.iterrows()
        )
        
        return self.realized_pnl + unrealized_pnl

    def get_current_equity(self, current_price: float, iv_bin_index: int) -> float:
        return self.initial_cash + self.get_total_pnl(current_price, iv_bin_index)

    def update_positions_after_time_step(self, days_of_decay: int, current_price: float, iv_bin_index: int, current_step: int):
        """
        Applies time decay and handles expirations. Crucially,
        """
        
        self.receipts_for_current_step = []

        # Now, the rest of the method proceeds as before.
        if days_of_decay == 0 or self.portfolio.empty:
            return

        self.portfolio['days_to_expiry'] = (self.portfolio['days_to_expiry'] - days_of_decay).clip(lower=0)
        
        expired_indices = self.portfolio[self.portfolio['days_to_expiry'] == 0].index
        if not expired_indices.empty:
            for idx in sorted(expired_indices, reverse=True):
                self.close_position(idx, current_price, iv_bin_index, current_step)

        # Finally, check for any profitable short legs that can be closed for cheap.
        self._close_worthless_short_legs(current_price, iv_bin_index, current_step)

    def close_position(self, position_index: int, current_price: float, iv_bin_index: int, current_step: int):
        """
        Atomically closes a single leg. It now correctly isolates the target
        strategy, re-profiles only its remaining legs, and leaves all other
        strategies in the portfolio untouched. This is the definitive version.
        """
        # --- 1. Robust Failsafe Check ---
        portfolio_size = len(self.portfolio)
        if not (-portfolio_size <= position_index < portfolio_size):
            print(f"Warning: Attempted to close invalid index: {position_index} on portfolio of size {portfolio_size}. Aborting.")
            return
        
        # --- 2. Isolate the Target Leg and its Strategy ---
        leg_to_close = self.portfolio.iloc[position_index].copy()
        target_creation_id = leg_to_close['creation_id']
        absolute_index_to_close = self.portfolio.index[position_index]

        # Isolate all other, untouched strategies
        untouched_strategies_df = self.portfolio[self.portfolio['creation_id'] != target_creation_id]
        # Isolate the legs of the strategy we are actually modifying
        strategy_to_modify_df = self.portfolio[self.portfolio['creation_id'] == target_creation_id]

        # --- 3. Process the Closing Leg ---
        self._process_leg_closures(pd.DataFrame([leg_to_close.to_dict()]), current_price, current_step)

        # --- 4. Intelligently Re-Profile the REMAINDER of the TARGET STRATEGY ---
        remaining_legs_df = strategy_to_modify_df.drop(absolute_index_to_close)

        if not remaining_legs_df.empty:
            remaining_legs_list = remaining_legs_df.to_dict(orient='records')
            new_strategy_name = self._identify_strategy_from_legs(remaining_legs_df)
            
            pnl_profile = self._calculate_universal_risk_profile(remaining_legs_list, self.realized_pnl)
            pnl_profile['strategy_id'] = self.strategy_name_to_id.get(new_strategy_name, -3)
            
            # --- 5. Atomically Rebuild the Portfolio ---
            # Start with the untouched strategies
            self.portfolio = untouched_strategies_df.copy()
            # Add the newly modified strategy back into the portfolio
            self._execute_trades(remaining_legs_list, pnl_profile)
        else:
            # If no legs remain from the target strategy, the portfolio is just the untouched ones.
            self.portfolio = untouched_strategies_df.copy().reset_index(drop=True)

    def _identify_strategy_from_legs(self, legs_df: pd.DataFrame) -> str:
        """
        A definitive, intelligent helper that identifies the name of a strategy
        based on the properties of its constituent legs. This version can
        identify single legs, all four vertical spread types, and the two most
        common 3-legged ratio spreads.
        """
        num_legs = len(legs_df)
        
        # --- Case 1: Single Leg ---
        if num_legs == 1:
            leg = legs_df.iloc[0]
            return f"{leg['direction'].upper()}_{leg['type'].upper()}"
        
        # --- Case 2: Two Legs ---
        if num_legs == 2:
            if len(legs_df['type'].unique()) == 1 and len(legs_df['direction'].unique()) == 2:
                leg1, leg2 = legs_df.iloc[0], legs_df.iloc[1]
                option_type = leg1['type'].upper()
                
                if leg1['strike_price'] > leg2['strike_price']:
                    higher_strike_leg, lower_strike_leg = leg1, leg2
                else:
                    higher_strike_leg, lower_strike_leg = leg2, leg1
                
                # <<< --- THE DEFINITIVE LOGIC FIX --- >>>
                if option_type == 'CALL':
                    # A Bull Call Spread has the SHORT leg at the HIGHER strike.
                    return 'OPEN_BULL_CALL_SPREAD' if higher_strike_leg['direction'] == 'short' else 'OPEN_BEAR_CALL_SPREAD'
                else: # PUT
                    # A Bear Put Spread has the SHORT leg at the LOWER strike.
                    return 'OPEN_BEAR_PUT_SPREAD' if lower_strike_leg['direction'] == 'short' else 'OPEN_BULL_PUT_SPREAD'
            else:
                return "CUSTOM_2_LEGS"

        # --- Case 3: Three Legs ---
        if num_legs == 3:
            direction_counts = legs_df['direction'].value_counts()
            
            # Check for a 2-to-1 debit-style ratio spread
            if direction_counts.get('long', 0) == 2 and direction_counts.get('short', 0) == 1:
                return "LONG_RATIO_SPREAD"
            
            # Check for a 2-to-1 credit-style ratio spread
            elif direction_counts.get('short', 0) == 2 and direction_counts.get('long', 0) == 1:
                return "SHORT_RATIO_SPREAD"
            
            else:
                # Fallback for other, rarer 3-leg combos (e.g., 3 longs)
                return "CUSTOM_3_LEGS"
        
        # --- Fallback for all other complex combinations (e.g., 4+ legs) ---
        # This is a critical fallback to ensure the function always returns a string.
        return f"CUSTOM_{num_legs}_LEGS"

    def close_all_positions(self, current_price: float, iv_bin_index: int, current_step: int):
        """Closes all open positions in the portfolio one by one."""
        # We loop while the portfolio is not empty, always closing the last element.
        while not self.portfolio.empty:
            self.close_position(-1, current_price, iv_bin_index, current_step)

    def get_pnl_verification(self, current_price: float, iv_bin_index: int) -> dict:
        """Calculates the components for the P&L verification panel."""
        unrealized_pnl = self.get_total_pnl(current_price, iv_bin_index) - self.realized_pnl
        return {
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'verified_total_pnl': self.realized_pnl + unrealized_pnl
        }

    def get_payoff_data(self, current_price: float, iv_bin_index: int) -> dict:
        """
        Calculates a rich data set for the P&L payoff diagram.
        MODIFIED: This version is now fully optimized and only sends the bare
        minimum data required for the frontend to render everything.
        """
        if self.portfolio.empty:
            return {'spot_price': current_price, 'sigma_levels': {}}

        # We still need to calculate the expected move for the sigma bands.
        current_dte = self.portfolio.iloc[0]['days_to_expiry']
        atm_iv = self.market_rules_manager.get_implied_volatility(0, 'call', iv_bin_index)
        expected_move_points = 0
        if current_dte > 0:
            expected_move_points = current_price * atm_iv * math.sqrt(current_dte / 365.25)

        sigma_levels = {
            'plus_one': current_price + expected_move_points,
            'minus_one': current_price - expected_move_points,
        }

        # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
        # We no longer send 'expiry_pnl' OR 'current_pnl'.
        # The frontend will reconstruct both from the portfolio state, which will
        # now include the live_iv for each leg.
        return {
            'spot_price': current_price,
            'sigma_levels': sigma_levels
        }

    # --- Private Methods ---
    def _open_jade_lizard(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """Opens a 3-leg Jade Lizard or Reverse Jade Lizard."""
        if len(self.portfolio) > self.max_positions - 3: return

        is_reverse = 'REVERSE' in action_name
        
        # 1. Define the naked leg
        naked_leg_type = 'call' if is_reverse else 'put'
        
        # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
        # The delta for a put must be negative, and for a call must be positive.
        naked_leg_delta = 0.20 if naked_leg_type == 'call' else -0.20

        naked_strike = self._find_strike_for_delta(naked_leg_delta, naked_leg_type, current_price, iv_bin_index, days_to_expiry)
        if naked_strike is None: return

        naked_leg_def = {'type': naked_leg_type, 'direction': 'short', 'strike_price': naked_strike}

        # 2. Define the spread
        spread_type = 'put' if is_reverse else 'call'
        spread_direction = 'BULL_PUT_SPREAD' if is_reverse else 'BEAR_CALL_SPREAD'
        
        spread_legs_def = self._find_best_available_spread(spread_type, spread_direction, current_price, iv_bin_index, days_to_expiry, is_credit_spread=True)
        if not spread_legs_def: return
            
        # 3. Assemble, price, and execute
        final_legs_def = [naked_leg_def] + spread_legs_def
        for leg in final_legs_def:
            leg.update({'entry_step': current_step, 'days_to_expiry': days_to_expiry})
        
        priced_legs = self._price_legs(final_legs_def, current_price, iv_bin_index, check_short_rule=True)
        if not priced_legs: return

        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(priced_legs, pnl_profile)

    def _open_big_lizard(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """Opens a 3-leg Big Lizard or Reverse Big Lizard."""
        if len(self.portfolio) > self.max_positions - 3: return

        is_reverse = 'REVERSE' in action_name

        # 1. Define the Strangle component based on delta
        strangle_put_delta = -0.55 if is_reverse else -0.45
        strangle_call_delta = 0.45 if is_reverse else 0.55

        strike_put = self._find_strike_for_delta(strangle_put_delta, 'put', current_price, iv_bin_index, days_to_expiry)
        strike_call = self._find_strike_for_delta(strangle_call_delta, 'call', current_price, iv_bin_index, days_to_expiry)
        if strike_put is None or strike_call is None: return
        
        strangle_legs_def = [
            {'type': 'put', 'direction': 'short', 'strike_price': strike_put},
            {'type': 'call', 'direction': 'short', 'strike_price': strike_call}
        ]
        
        # 2. Define the long hedge leg
        hedge_leg_type = 'put' if is_reverse else 'call'
        hedge_leg_delta = -0.45 if is_reverse else 0.45
        hedge_strike = self._find_strike_for_delta(hedge_leg_delta, hedge_leg_type, current_price, iv_bin_index, days_to_expiry)
        if hedge_strike is None: return

        hedge_leg_def = {'type': hedge_leg_type, 'direction': 'long', 'strike_price': hedge_strike}

        # 3. Assemble, price, and execute
        final_legs_def = strangle_legs_def + [hedge_leg_def]
        for leg in final_legs_def:
            leg.update({'entry_step': current_step, 'days_to_expiry': days_to_expiry})
            
        priced_legs = self._price_legs(final_legs_def, current_price, iv_bin_index, check_short_rule=True)
        if not priced_legs: return
            
        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(priced_legs, pnl_profile)

    def _open_ratio_spread(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """Opens a 1x2 ratio spread (1 long, 2 short)."""
        if len(self.portfolio) > self.max_positions - 3: return
        
        option_type = 'put' if 'PUT' in action_name else 'call'

        # 1. Find the two short legs @ 20 delta
        short_delta = -0.20 if option_type == 'put' else 0.20
        short_strike = self._find_strike_for_delta(short_delta, option_type, current_price, iv_bin_index, days_to_expiry)
        if short_strike is None: return
        
        # 2. Find the one long leg @ 30-35 delta
        long_delta_range = [-0.30, -0.35] if option_type == 'put' else [0.30, 0.35]
        long_strike = self._find_strike_from_delta_list(long_delta_range, option_type, current_price, iv_bin_index, days_to_expiry)
        if long_strike is None: return

        # 3. Assemble the 3 legs (with two identical short legs)
        final_legs_def = [
            {'type': option_type, 'direction': 'short', 'strike_price': short_strike},
            {'type': option_type, 'direction': 'short', 'strike_price': short_strike},
            {'type': option_type, 'direction': 'long', 'strike_price': long_strike}
        ]
        
        for leg in final_legs_def:
            leg.update({'entry_step': current_step, 'days_to_expiry': days_to_expiry})
            
        priced_legs = self._price_legs(final_legs_def, current_price, iv_bin_index, check_short_rule=True)
        if not priced_legs: return
            
        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(priced_legs, pnl_profile)

    def _find_portfolio_neutral_strike(self, leg_index: int, current_price: float, iv_bin_index: int) -> float or None:
        """
        A powerful solver that finds the optimal new strike for a specific leg that
        will bring the entire portfolio's net delta as close to zero as possible,
        searching only within a constrained "liquidity window" around the ATM strike.
        """
        leg_to_roll = self.portfolio.iloc[leg_index]
        other_legs = self.portfolio.drop(leg_to_roll.name).to_dict('records')
        
        best_strike = None
        min_abs_portfolio_delta = abs(self.get_raw_portfolio_stats(current_price, iv_bin_index)['delta'])
        
        atm_price = self.market_rules_manager.get_atm_price(current_price)

        # Instead of searching the full range, we only search within the liquid window.
        search_width = self.hedge_roll_search_width
        for offset in range(-search_width, search_width + 1):
            candidate_strike = atm_price + (offset * self.strike_distance)
            if candidate_strike <= 0 or candidate_strike == leg_to_roll['strike_price']:
                continue

            # ... (the rest of the function's logic for checking the candidate is unchanged) ...
            candidate_leg = leg_to_roll.to_dict()
            candidate_leg['strike_price'] = candidate_strike
            
            if candidate_leg['direction'] == 'short':
                is_call = candidate_leg['type'] == 'call'
                vol = self.iv_calculator(offset, candidate_leg['type'])
                greeks_single_leg = self.bs_manager.get_all_greeks_and_price(current_price, candidate_strike, candidate_leg['days_to_expiry'], vol, is_call)
                premium = self.bs_manager.get_price_with_spread(greeks_single_leg['price'], is_buy=False, bid_ask_spread_pct=self.bid_ask_spread_pct)
                if premium < self.close_short_leg_on_profit_threshold:
                    continue

            hypothetical_portfolio_legs = other_legs + [candidate_leg]
            greeks = self.get_raw_greeks_for_legs(hypothetical_portfolio_legs, current_price, iv_bin_index)
            
            if abs(greeks['delta']) < min_abs_portfolio_delta:
                min_abs_portfolio_delta = abs(greeks['delta'])
                best_strike = candidate_strike
        
        return best_strike

    def _get_total_expiry_pnl_at_price(self, legs_df: pd.DataFrame, price: float) -> float:
        """
        Calculates the total P&L for a given DataFrame of legs at a specific price
        AT EXPIRATION (using intrinsic value).
        """
        if legs_df.empty:
            return self.realized_pnl

        unrealized_pnl = 0.0
        for _, leg in legs_df.iterrows():
            # --- We put the simple, fast intrinsic value logic directly here ---
            is_call = leg['type'] == 'call'
            pnl_multiplier = 1 if leg['direction'] == 'long' else -1
            
            if is_call:
                intrinsic_value = max(0.0, price - leg['strike_price'])
            else: # Put
                intrinsic_value = max(0.0, leg['strike_price'] - price)
            
            pnl_per_share = intrinsic_value - leg['entry_premium']
            unrealized_pnl += pnl_per_share * pnl_multiplier * self.lot_size
        
        return self.realized_pnl + unrealized_pnl

    # The name is now specific, and the dead code is removed.
    def _get_t0_pnl_for_leg(self, leg: pd.Series, price: float) -> float:
        """
        The single, canonical function for calculating the current Mark-to-Market (T+0)
        P&L of one option leg using the full-fidelity Black-Scholes model.
        """
        is_call = leg['type'] == 'call'
        pnl_multiplier = 1 if leg['direction'] == 'long' else -1

        # This function ONLY calculates T+0 P&L
        atm_price = self.market_rules_manager.get_atm_price(price)
        offset = round((leg['strike_price'] - atm_price) / self.strike_distance)
        vol = self.iv_calculator(offset, leg['type'])
        
        greeks = self.bs_manager.get_all_greeks_and_price(
            price, leg['strike_price'], leg['days_to_expiry'], vol, is_call
        )
        
        current_value = self.bs_manager.get_price_with_spread(
            greeks['price'], is_buy=(leg['direction'] == 'short'), bid_ask_spread_pct=self.bid_ask_spread_pct
        )

        pnl_per_share = current_value - leg['entry_premium']
        return pnl_per_share * pnl_multiplier * self.lot_size

    # --- HELPER METHOD FOR "DE-MORPHING" ACTIONS ---
    def _process_leg_closures(self, legs_to_close: pd.DataFrame, current_price: float, current_step: int):
        """A helper to correctly process P&L and receipts for a set of closing legs."""
        if legs_to_close.empty:
            return

        for _, leg in legs_to_close.iterrows():
            vol = self.iv_calculator(0, leg['type'])
            greeks = self.bs_manager.get_all_greeks_and_price(current_price, leg['strike_price'], leg['days_to_expiry'], vol, leg['type'] == 'call')
            exit_premium = self.bs_manager.get_price_with_spread(greeks['price'], is_buy=(leg['direction'] == 'short'), bid_ask_spread_pct=self.bid_ask_spread_pct)
            pnl = (exit_premium - leg['entry_premium']) * (1 if leg['direction'] == 'long' else -1) * self.lot_size
            self.realized_pnl += pnl - self.brokerage_per_leg
            if self.is_eval_mode:
                receipt = {
                    'position': f"{leg['direction'].upper()} {leg['type'].upper()}", 'strike': leg['strike_price'],
                    'entry_day': (leg['entry_step'] // self.steps_per_day) + 1,
                    'exit_day': (current_step // self.steps_per_day) + 1,
                    'entry_prem': leg['entry_premium'], 'exit_prem': exit_premium, 'realized_pnl': pnl
                }
                self.receipts_for_current_step.append(receipt)

    def _close_worthless_short_legs(self, current_price: float, iv_bin_index: int, current_step: int):
        """
        Automatically closes any short leg that is profitable and whose current
        premium has decayed below a configured threshold (e.g., $1.00).
        """
        # Check if the rule is disabled or the portfolio is empty
        if self.close_short_leg_on_profit_threshold <= 0 or self.portfolio.empty:
            return

        indices_to_close = []
        for i, pos in self.portfolio.iterrows():
            # Condition 1: Must be a short position
            if pos['direction'] != 'short':
                continue

            # Calculate the current premium of the leg
            is_call = pos['type'] == 'call'
            atm_price = self.market_rules_manager.get_atm_price(current_price)
            offset = round((pos['strike_price'] - atm_price) / self.strike_distance)
            vol = self.iv_calculator(offset, pos['type'])
            greeks = self.bs_manager.get_all_greeks_and_price(current_price, pos['strike_price'], pos['days_to_expiry'], vol, is_call)
            
            # The "ask" price is what we would pay to buy it back to close
            current_premium = self.bs_manager.get_price_with_spread(greeks['price'], is_buy=True, bid_ask_spread_pct=self.bid_ask_spread_pct)

            # Condition 2: The current premium must be below our threshold
            if current_premium < self.close_short_leg_on_profit_threshold:
                # Condition 3: The position must be profitable (entry premium was higher than what we're paying now)
                if pos['entry_premium'] > current_premium:
                    indices_to_close.append(i)

        # Close the identified positions in reverse order to avoid index shifting issues
        if indices_to_close:
            #print(f"DEBUG: Auto-closing {len(indices_to_close)} worthless short positions.") # Optional debug print
            for index in sorted(indices_to_close, reverse=True):
                self.close_position(index, current_price, iv_bin_index, current_step)

    def _update_hedge_status(self):
        """
        The definitive, user-designed risk engine. It uses a simple and robust
        pairing algorithm to determine the hedge status of each leg.
        Any long option can hedge any short option of the same type, and vice-versa.
        """
        if self.portfolio.empty:
            return

        # Start by assuming all legs are Naked until paired.
        self.portfolio['is_hedged'] = False
        
        # --- Create mutable lists of the indices for pairing ---
        calls = self.portfolio[self.portfolio['type'] == 'call']
        long_call_indices = calls[calls['direction'] == 'long'].index.to_list()
        short_call_indices = calls[calls['direction'] == 'short'].index.to_list()

        puts = self.portfolio[self.portfolio['type'] == 'put']
        long_put_indices = puts[puts['direction'] == 'long'].index.to_list()
        short_put_indices = puts[puts['direction'] == 'short'].index.to_list()

        # Pair as many calls as possible
        while long_call_indices and short_call_indices:
            # Take one of each, mark them as Hedged, and remove them.
            lc_idx = long_call_indices.pop()
            sc_idx = short_call_indices.pop()
            self.portfolio.loc[lc_idx, 'is_hedged'] = True
            self.portfolio.loc[sc_idx, 'is_hedged'] = True
            
        # Pair as many puts as possible
        while long_put_indices and short_put_indices:
            lp_idx = long_put_indices.pop()
            sp_idx = short_put_indices.pop()
            self.portfolio.loc[lp_idx, 'is_hedged'] = True
            self.portfolio.loc[sp_idx, 'is_hedged'] = True

        # Any leg whose index remains in any of the lists is, by definition,
        # un-paired and Naked. Since we defaulted 'is_hedged' to False,
        # no further action is needed.

    def shift_position(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int):
        """
        Shifts a single leg of a strategy, correctly preserving all other legs
        and strategies in the portfolio. This is the definitive, robust version.
        """
        try:
            parts = action_name.split('_')
            direction, position_index = parts[1].upper(), int(parts[3])
            if not (-len(self.portfolio) <= position_index < len(self.portfolio)): return

            # --- 1. Identify all components ---
            leg_to_shift_df = self.portfolio.iloc[[position_index]]
            leg_to_shift = leg_to_shift_df.iloc[0].to_dict()
            original_creation_id = leg_to_shift['creation_id']
            
            # Isolate all other, untouched legs from all other strategies
            untouched_legs = self.portfolio[self.portfolio['creation_id'] != original_creation_id]
            # Isolate the sibling legs of the strategy being modified
            sibling_legs = self.portfolio[(self.portfolio['creation_id'] == original_creation_id) & (self.portfolio.index != position_index)]

            # --- 2. Process the "close" part of the shift for the target leg ---
            self._process_leg_closures(leg_to_shift_df, current_price, current_step)
            
            # --- 3. Define and price the new, shifted leg ---
            strike_modifier = self.strike_distance if direction == 'UP' else -self.strike_distance
            new_leg_def = [{'type': leg_to_shift['type'], 'direction': leg_to_shift['direction'],
                            'strike_price': leg_to_shift['strike_price'] + strike_modifier,
                            'days_to_expiry': leg_to_shift['days_to_expiry'], 'entry_step': current_step}]
            priced_new_leg = self._price_legs(new_leg_def, current_price, iv_bin_index)
            
            # --- 4. Re-assemble the modified strategy and calculate its new profile ---
            modified_strategy_legs = sibling_legs.to_dict(orient='records') + priced_new_leg
            pnl_profile = self._calculate_universal_risk_profile(modified_strategy_legs, self.realized_pnl)
            pnl_profile['strategy_id'] = leg_to_shift['strategy_id'] # Preserve original strategy ID

            # --- 5. Atomically rebuild the entire portfolio ---
            # Start with the legs that were never touched
            self.portfolio = untouched_legs.copy()
            # Add the newly modified strategy
            self._execute_trades(modified_strategy_legs, pnl_profile)
            
            self._update_hedge_status()

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse shift action '{action_name}'. Error: {e}")

    def shift_to_atm(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int):
        """Shifts a single leg to the ATM strike, correctly preserving other strategies."""
        try:
            position_index = int(action_name.split('_')[-1])
            if not (-len(self.portfolio) <= position_index < len(self.portfolio)): return
            
            leg_to_shift_df = self.portfolio.iloc[[position_index]]
            leg_to_shift = leg_to_shift_df.iloc[0].to_dict()
            original_creation_id = leg_to_shift['creation_id']
            
            new_atm_strike = self.market_rules_manager.get_atm_price(current_price)
            if leg_to_shift['strike_price'] == new_atm_strike: return

            untouched_legs = self.portfolio[self.portfolio['creation_id'] != original_creation_id]
            sibling_legs = self.portfolio[(self.portfolio['creation_id'] == original_creation_id) & (self.portfolio.index != position_index)]

            self._process_leg_closures(leg_to_shift_df, current_price, current_step)
            
            new_leg_def = [{'type': leg_to_shift['type'], 'direction': leg_to_shift['direction'],
                            'strike_price': new_atm_strike, 'days_to_expiry': leg_to_shift['days_to_expiry'],
                            'entry_step': current_step}]
            priced_new_leg = self._price_legs(new_leg_def, current_price, iv_bin_index)
            
            modified_strategy_legs = sibling_legs.to_dict(orient='records') + priced_new_leg
            pnl_profile = self._calculate_universal_risk_profile(modified_strategy_legs, self.realized_pnl)
            pnl_profile['strategy_id'] = leg_to_shift['strategy_id']

            self.portfolio = untouched_legs.copy()
            self._execute_trades(modified_strategy_legs, pnl_profile)
            
            self._update_hedge_status()
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse shift_to_atm action '{action_name}'. Error: {e}")

    def get_portfolio_summary(self, current_price: float, iv_bin_index: int) -> dict:
        """
        The definitive and universal summary calculator. It uses the universal risk engine
        to get breakeven points and then calculates POP for ANY strategy shape.
        This version includes an ultra-robust calculation for rr_ratio to prevent
        non-finite values from poisoning the observation vector.
        """
        # --- 1. Handle Empty Portfolio ---
        if self.portfolio.empty:
            is_profitable = self.realized_pnl > 0
            return {
                'max_profit': self.realized_pnl, 'max_loss': self.realized_pnl,
                'rr_ratio': 0.0, 'prob_profit': 1.0 if is_profitable else 0.0
            }

       # --- 2. Get Risk Profile from Universal Engine ---
        portfolio_legs_as_dict = self.portfolio.to_dict(orient='records')
        risk_profile = self._calculate_universal_risk_profile(portfolio_legs_as_dict, self.realized_pnl)
        
        max_profit = risk_profile['total_max_profit']
        max_loss = risk_profile['max_loss']
        breakevens = risk_profile['breakevens']
        
        # This new logic is completely robust against division-by-zero errors.
        rr_ratio = 0.0  # Start with a safe default
        try:
            # Only calculate if there's a profit potential and a finite loss.
            if max_profit > 0 and max_loss < 0:
                calculated_rr = abs(max_profit / max_loss)
                # If the loss is tiny, the ratio can be huge. Cap it for stability.
                rr_ratio = min(calculated_rr, 999.0)
        except (TypeError, ZeroDivisionError):
            # If max_loss is 0, we can consider it an infinite R:R.
            if max_profit > 0:
                rr_ratio = 999.0
            else:
                rr_ratio = 0.0
        
        days_to_expiry = self.portfolio.iloc[0]['days_to_expiry']
        vol = self.iv_calculator(0, 'call') # Use injected iv_calculator
        
        # --- 3. Calculate POP based on the found breakevens ---
        total_pop = 0.0

        if not breakevens:
            if max_profit > 0 and max_loss >= 0: total_pop = 1.0
            else: total_pop = 0.0
        
        elif len(breakevens) == 1:
            be = breakevens[0]
            test_price = be + (self.strike_distance * 0.1) 
            
            if self._get_total_expiry_pnl_at_price(self.portfolio, test_price) > 0:
                greeks = self.bs_manager.get_all_greeks_and_price(current_price, be, days_to_expiry, vol, is_call=True)
                total_pop = _numba_cdf(greeks['d2'])
            else:
                greeks = self.bs_manager.get_all_greeks_and_price(current_price, be, days_to_expiry, vol, is_call=False)
                total_pop = _numba_cdf(-greeks['d2'])
       
        else: # Two or more breakevens
            boundaries = [-np.inf] + breakevens + [np.inf]
            
            for i in range(len(boundaries) - 1):
                lower_b, upper_b = boundaries[i], boundaries[i+1]
                
                test_price = (lower_b + upper_b) / 2 if -np.inf < lower_b and np.inf > upper_b else \
                             (upper_b - self.strike_distance if lower_b == -np.inf else lower_b + self.strike_distance)
                
                if self._get_total_expiry_pnl_at_price(self.portfolio, test_price) > 0:
                    greeks_upper = self.bs_manager.get_all_greeks_and_price(current_price, upper_b, days_to_expiry, vol, False) if upper_b != np.inf else None
                    prob_below_upper = _numba_cdf(-greeks_upper['d2']) if greeks_upper else 1.0

                    greeks_lower = self.bs_manager.get_all_greeks_and_price(current_price, lower_b, days_to_expiry, vol, False) if lower_b != -np.inf else None
                    prob_below_lower = _numba_cdf(-greeks_lower['d2']) if greeks_lower else 0.0
                    
                    total_pop += (prob_below_upper - prob_below_lower)

        return {
            'max_profit': max_profit, 'max_loss': max_loss,
            'rr_ratio': rr_ratio, 'prob_profit': np.clip(total_pop, 0.0, 1.0)
        }

    def get_portfolio_greeks(self, current_price: float, iv_bin_index: int) -> Dict:
        """Calculates and returns a dictionary of normalized portfolio-level Greeks."""
        total_delta, total_gamma, total_theta, total_vega = 0.0, 0.0, 0.0, 0.0
        if self.portfolio.empty:
            return {'delta_norm': 0.0, 'gamma_norm': 0.0, 'theta_norm': 0.0, 'vega_norm': 0.0}

        atm_price = self.market_rules_manager.get_atm_price(current_price)

        for _, pos in self.portfolio.iterrows():
            direction_multiplier = 1 if pos['direction'] == 'long' else -1
            is_call = pos['type'] == 'call'
            
            # --- THE FIX IS HERE ---
            # 1. We must first calculate sigma (implied volatility) for the leg.
            offset = round((pos['strike_price'] - atm_price) / self.strike_distance)
            vol = self.iv_calculator(offset, pos['type'])
            
            # 2. Now we can call get_all_greeks_and_price with all 5 required arguments.
            greeks = self.bs_manager.get_all_greeks_and_price(
                current_price,
                pos['strike_price'],
                pos['days_to_expiry'],
                vol,       # <-- The missing `sigma` argument
                is_call    # <-- The final `is_call` argument
            )
            
            total_delta += greeks['delta'] * self.lot_size * direction_multiplier
            total_gamma += greeks['gamma'] * (self.lot_size**2 * current_price**2 / 100) * direction_multiplier
            total_theta += greeks['theta'] * self.lot_size * direction_multiplier
            total_vega += greeks['vega'] * self.lot_size * direction_multiplier

        max_delta_exposure = self.max_positions * self.lot_size
        return {
            'delta_norm': np.clip(total_delta / max_delta_exposure, -1.0, 1.0) if max_delta_exposure > 0 else 0.0,
            'gamma_norm': math.tanh(total_gamma / self.initial_cash),
            'theta_norm': math.tanh(total_theta / self.initial_cash),
            'vega_norm': math.tanh(total_vega / self.initial_cash)
        }

    def sort_portfolio(self):
        """
        Sorts the portfolio using a multi-key, stable sorting algorithm. This
        ensures an absolutely deterministic order for the DataFrame's index,
        which is critical for both the RL agent's state representation and for
        allowing human analysis to correlate actions (e.g., SHIFT_POS_1) with
        the correct leg in the UI. This logic is used for all modes.
        """
        if not self.portfolio.empty:
            # Define the deterministic sorting hierarchy.
            sort_keys = ['creation_id', 'type', 'strike_price']
            
            # Use a stable sorting algorithm (mergesort) to break any potential
            # (though highly unlikely) remaining ties consistently.
            self.portfolio = self.portfolio.sort_values(
                by=sort_keys, 
                kind='mergesort'
            ).reset_index(drop=True)

    def hedge_delta_with_atm_option(self, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        if len(self.portfolio) >= self.max_positions: return

        stats = self.get_raw_portfolio_stats(current_price, iv_bin_index)
        current_delta = stats['delta']

        hedge_leg_def = None
        atm_strike = self.market_rules_manager.get_atm_price(current_price)
        
        if current_delta < -1:
            hedge_leg_def = [{'type': 'call', 'direction': 'long', 'strike_price': atm_strike}]
        elif current_delta > 1:
            hedge_leg_def = [{'type': 'put', 'direction': 'long', 'strike_price': atm_strike}]
        else: return

        for leg in hedge_leg_def:
            leg.update({'entry_step': current_step, 'days_to_expiry': days_to_expiry})
        
        priced_legs = self._price_legs(hedge_leg_def, current_price, iv_bin_index)
        if not priced_legs: return

        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl)
        
        direction_str = priced_legs[0]['direction'].upper()
        type_str = priced_legs[0]['type'].upper()
        # The ID for a naked leg strategy is its simple internal name (e.g., "LONG_CALL")
        internal_strategy_name = f"{direction_str}_{type_str}"
        
        strategy_id = self.strategy_name_to_id.get(internal_strategy_name)
        
        assert strategy_id is not None, f"FATAL: Could not find strategy ID for hedge key: '{internal_strategy_name}'"
        pnl_profile['strategy_id'] = strategy_id
        
        self._execute_trades(priced_legs, pnl_profile)
     
    def render(self, current_price: float, current_step: int, iv_bin_index: int, steps_per_day: int):
        total_pnl = self.get_total_pnl(current_price, iv_bin_index)
        day = current_step // steps_per_day + 1
        print(f"Step: {current_step:04d} | Day: {day:02d} | Price: ${current_price:9.2f} | Positions: {len(self.portfolio):1d} | Total PnL: ${total_pnl:9.2f}")
        if not self.portfolio.empty:
            print(self.portfolio.to_string(index=False))
        return

    def _execute_trades(self, trades_to_execute: List[Dict], strategy_pnl: Dict):
        """
        The definitive method for adding legs to the portfolio. It is the single
        source of truth for updating hedge status.
        This version ensures state is finalized even when trades_to_execute is empty.
        """

        # --- 1. Add New Trades (if any) ---
        if trades_to_execute:
            # Deduct brokerage from realized PnL for each new leg being opened.
            opening_brokerage = len(trades_to_execute) * self.brokerage_per_leg
            self.realized_pnl -= opening_brokerage
            
            transaction_id = self.next_creation_id
            self.next_creation_id += 1
            
            strategy_id = strategy_pnl.get('strategy_id', -1)
            assert strategy_id != -1, (f"CRITICAL ERROR: Strategy ID not found for PnL object: {strategy_pnl}")
            
            for trade in trades_to_execute:
                trade['creation_id'] = transaction_id
                trade['is_hedged'] = False # Default to False before the update
                trade['strategy_id'] = strategy_id
                trade['strategy_max_profit'] = strategy_pnl.get('max_profit', 0.0)
                trade['strategy_max_loss'] = strategy_pnl.get('max_loss', 0.0)
            
            new_positions_df = pd.DataFrame(trades_to_execute).astype(self.portfolio_dtypes)
            self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)

        # --- 2. Finalize State (ALWAYS runs) ---
        # This ensures that after any modification (add, close, shift),
        # the hedge status and the post-action snapshot are correctly updated.
        self._update_hedge_status()

    def _price_legs(self, legs: List[Dict], current_price: float, iv_bin_index: int, check_short_rule: bool = False) -> List[Dict] or None:
        """
        Calculates the entry premium for a list of legs.
        MODIFIED: Now with extensive debugging prints.
        """
        # <<< --- DEBUG PRINT 1: Entry Point --- >>>
        #print(f"\n[DEBUG] --- Entering _price_legs ---")
        #print(f"[DEBUG] check_short_rule = {check_short_rule}")
        #print(f"[DEBUG] Legs to price ({len(legs)} total):")
        #for leg in legs:
            #print(f"  -> {leg}")

        atm_price = self.market_rules_manager.get_atm_price(current_price)
        priced_legs = []
        for i, leg in enumerate(legs):
            try:
                offset = round((leg['strike_price'] - atm_price) / self.strike_distance)
                vol = self.iv_calculator(offset, leg['type'])
                greeks = self.bs_manager.get_all_greeks_and_price(current_price, leg['strike_price'], leg['days_to_expiry'], vol, leg['type'] == 'call')
                
                is_opening_a_long_position = (leg['direction'] == 'long')
                entry_premium = self.bs_manager.get_price_with_spread(greeks['price'], is_buy=is_opening_a_long_position, bid_ask_spread_pct=self.bid_ask_spread_pct)

                # <<< --- DEBUG PRINT 2: Pricing Details --- >>>
                #print(f"[DEBUG] Leg {i+1}: {leg['direction']} {leg['type']} @ {leg['strike_price']:.2f} | Calculated Premium = {entry_premium:.4f}")

                if check_short_rule and not is_opening_a_long_position:
                    if entry_premium < self.close_short_leg_on_profit_threshold:
                        # <<< --- DEBUG PRINT 3: Failure Point (Rule) --- >>>
                        #print(f"[DEBUG] !!! RULE VIOLATION: Short leg premium {entry_premium:.4f} is below threshold {self.close_short_leg_on_profit_threshold}. Aborting.")
                        return None

                new_leg = leg.copy()
                new_leg['entry_premium'] = entry_premium
                priced_legs.append(new_leg)
            except Exception as e:
                # <<< --- DEBUG PRINT 4: Failure Point (Exception) --- >>>
                print(f"[DEBUG] !!! EXCEPTION during pricing leg {i+1}: {e}")
                traceback.print_exc()
                return None
        
        # <<< --- DEBUG PRINT 5: Success Point --- >>>
        #print(f"[DEBUG] --- _price_legs successful. Returning {len(priced_legs)} legs. ---")
        return priced_legs

    def _prepare_legs_for_numba(self, legs: List[Dict]) -> np.ndarray:
        """
        Safely converts a list of leg dictionaries into a NumPy array suitable
        for the Numba JIT function. It handles cases where keys might be missing.
        """
        # [strike, entry_premium, type_code, direction_code, creation_id, days_to_expiry]
        legs_array = np.empty((len(legs), 6), dtype=np.float32)
        for i, leg in enumerate(legs):
            legs_array[i, 0] = leg['strike_price']
            legs_array[i, 1] = leg.get('entry_premium', 0.0) # Default to 0 if not priced yet
            legs_array[i, 2] = 1 if leg['type'] == 'call' else 0
            legs_array[i, 3] = 1 if leg['direction'] == 'long' else -1
            legs_array[i, 4] = leg.get('creation_id', -1) # Default to -1 if not created yet
            legs_array[i, 5] = leg['days_to_expiry']
        return legs_array

    def _calculate_universal_risk_profile(self, legs: List[Dict], realized_pnl: float) -> Dict:
        """
        Calculates the risk profile. This version includes a final validation
        step to ensure no NaN values ever leave this function.
        """
        # ... (initial logic is unchanged) ...
        if not legs:
            return {'strategy_max_profit': realized_pnl, 'total_max_profit': realized_pnl, 'max_loss': realized_pnl, 'profit_factor': 0.0, 'breakevens': []}

        # --- Call the (now hardened) Numba simulation ---
        strategy_max_profit, strategy_max_loss, _, _ = _numba_run_pnl_simulation(
            self._prepare_legs_for_numba(legs), 0.0, self.start_price, self.lot_size,
            self.bs_manager.risk_free_rate, self.bid_ask_spread_pct
        )
        total_max_profit, total_max_loss, profit_factor, breakevens = _numba_run_pnl_simulation(
            self._prepare_legs_for_numba(legs), realized_pnl, self.start_price, self.lot_size,
            self.bs_manager.risk_free_rate, self.bid_ask_spread_pct
        )

        # <<< --- LAYER 2: FINAL VALIDATION --- >>>
        # This acts as a final guard rail before returning the data.
        final_profile = {
            'strategy_max_profit': min(self.undefined_risk_cap, strategy_max_profit if np.isfinite(strategy_max_profit) else 0.0),
            'total_max_profit': min(self.undefined_risk_cap, total_max_profit if np.isfinite(total_max_profit) else 0.0),
            'max_loss': max(-self.undefined_risk_cap, total_max_loss if np.isfinite(total_max_loss) else 0.0),
            'profit_factor': profit_factor if np.isfinite(profit_factor) else 0.0,
            'breakevens': sorted([be for be in breakevens if np.isfinite(be)]) # Filter out any non-finite breakevens
        }
        
        return final_profile

    def _open_single_leg(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) < self.max_positions, "Illegal attempt to open a leg when portfolio is full. Action mask failed."

        if len(self.portfolio) >= self.max_positions: return
        parts = action_name.split('_')
        direction, option_type, offset_str = parts[1].lower(), parts[2].lower(), parts[3].replace('ATM', '')
        offset = int(offset_str)
        strike_price = self.market_rules_manager.get_atm_price(current_price) + (offset * self.strike_distance)
        
        legs = [{'type': option_type, 'direction': direction, 'strike_price': strike_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}]
        # We now explicitly ask _price_legs to enforce the rule for this new position.
        priced_legs = self._price_legs(legs, current_price, iv_bin_index, check_short_rule=True)
        
        # If pricing failed because of our new rule, abort the action.
        if not priced_legs:
            return
        
        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        #print(f"DEBUG: {pnl_profile['strategy_id']} {action_name}")
        self._execute_trades(priced_legs, pnl_profile)

    def _open_straddle(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) <= self.max_positions - 2, "Illegal attempt to open a straddle when there are less than two positions available."
        if len(self.portfolio) > self.max_positions - 2: return
        direction = 'long' if 'LONG' in action_name else 'short'
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        legs = [{'type': 'call', 'direction': direction, 'strike_price': atm_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': direction, 'strike_price': atm_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}]
        legs = self._price_legs(legs, current_price, iv_bin_index)
        pnl  = self._calculate_universal_risk_profile(legs, self.realized_pnl)
        # 3. Now get the strategy ID.
        pnl['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(legs, pnl)

    def _open_strangle(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """
        Opens a two-leg strangle. This is a smart dispatcher that can handle
        both modern delta-based strangles and legacy fixed-width strangles.
        """
        if len(self.portfolio) > self.max_positions - 2: return
        
        parts = action_name.split('_')
        direction = parts[1].lower()
        legs = []

        # --- Dispatcher Logic ---
        if 'DELTA' in action_name:
            # --- INTELLIGENT, DELTA-BASED LOGIC ---
            try:
                target_delta_int = int(parts[4])
            except (ValueError, IndexError): return # Invalid format

            all_deltas = [15, 20, 25, 30]
            start_index = all_deltas.index(target_delta_int) if target_delta_int in all_deltas else 0
            delta_search_list = [d / 100.0 for d in all_deltas[start_index:]]

            strike_call = self._find_strike_from_delta_list(delta_search_list, 'call', current_price, iv_bin_index, days_to_expiry)
            strike_put = self._find_strike_from_delta_list([-d for d in delta_search_list], 'put', current_price, iv_bin_index, days_to_expiry)
            
            if strike_call and strike_put and strike_put < strike_call:
                legs = [{'type': 'call', 'direction': direction, 'strike_price': strike_call},
                        {'type': 'put', 'direction': direction, 'strike_price': strike_put}]
            else:
                return # Fallback if no valid delta strikes are found
       
        # --- Finalize the Trade (common to both logic paths) ---
        for leg in legs:
            leg['entry_step'] = current_step
            leg['days_to_expiry'] = days_to_expiry
        
        priced_legs = self._price_legs(legs, current_price, iv_bin_index)
        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(priced_legs, pnl_profile)

    def _find_strike_from_delta_list(
        self, target_deltas: List[float], option_type: str,
        current_price: float, iv_bin_index: int, days_to_expiry: float
    ) -> float or None:
        """
        Tries to find a strike matching a delta from a prioritized list.
        It attempts each delta in the list in order and returns the first one
        that finds a valid strike within the tolerance.
        Returns the strike price, or None if no delta in the list finds a match.
        """
        for delta in target_deltas:
            strike = self._find_strike_for_delta(delta, option_type, current_price, iv_bin_index, days_to_expiry)
            if strike is not None:
                return strike
        
        return None

    def _find_strike_for_delta(
        self, target_delta: float, option_type: str,
        current_price: float, iv_bin_index: int, days_to_expiry: float,
        tolerance: float = 0.05 # <<< NEW: Add a tolerance parameter
    ) -> float or None: # <<< NEW: Can now return None
        """
        Iteratively finds the strike price for an option that is closest to a target delta.
        Returns the strike price if a match within the tolerance is found, otherwise returns None.
        """
        is_call = option_type == 'call'
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        
        best_strike = None
        smallest_delta_diff = float('inf')
        
        for offset in range(-self.max_strike_offset, self.max_strike_offset + 1):
            strike_price = atm_price + (offset * self.strike_distance)
            if strike_price <= 0: continue

            vol = self.iv_calculator(offset, option_type)
            greeks = self.bs_manager.get_all_greeks_and_price(current_price, strike_price, days_to_expiry, vol, is_call)
            
            current_delta = greeks['delta']
            delta_diff = abs(current_delta - target_delta)
            
            if delta_diff < smallest_delta_diff:
                smallest_delta_diff = delta_diff
                best_strike = strike_price
        
        # --- THE NEW GUARD RAIL ---
        # Only return the strike if we found a match within our acceptable tolerance.
        if smallest_delta_diff <= tolerance:
            return best_strike
        else:
            return None # Signal that no acceptable strike was found

    def _open_iron_condor(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """
        Opens a SHORT Iron Condor using a tiered search for delta strikes (e.g., 30/15, then 35/20).
        If no dynamic strikes are valid, it falls back to a fixed-width condor.
        """
        if len(self.portfolio) > self.max_positions - 4: return
        
        direction = 'long' if 'LONG' in action_name else 'short'
        legs = []

        if direction == 'short':
            # --- THE NEW TIERED SEARCH LOGIC ---
            # 1. Define the prioritized list of delta combinations to try.
            delta_tiers = [
                {'short': 20, 'long': 15}, # Tier 1: Most conservative
                {'short': 25, 'long': 20}, # Tier 2
                {'short': 30, 'long': 15}, # Tier 3
                {'short': 30, 'long': 25}, # Tier 4
                {'short': 35, 'long': 20}, # Tier 5: Most aggressive
            ]

            for tier in delta_tiers:
                # print(f"DEBUG: Attempting Iron Condor tier {tier['short']}/{tier['long']}") # Optional for debugging
                short_delta = tier['short'] / 100.0
                long_delta = tier['long'] / 100.0
                
                # 2. Attempt to find all four strikes for the current tier.
                strike_short_put = self._find_strike_for_delta(-short_delta, 'put', current_price, iv_bin_index, days_to_expiry)
                strike_short_call = self._find_strike_for_delta(short_delta, 'call', current_price, iv_bin_index, days_to_expiry)
                strike_long_put = self._find_strike_for_delta(-long_delta, 'put', current_price, iv_bin_index, days_to_expiry)
                strike_long_call = self._find_strike_for_delta(long_delta, 'call', current_price, iv_bin_index, days_to_expiry)

                # 3. Run the robust guard rail on the results of this tier.
                all_strikes_found = all(s is not None for s in [strike_long_put, strike_short_put, strike_short_call, strike_long_call])
                if all_strikes_found and (strike_long_put < strike_short_put < strike_short_call < strike_long_call):
                    # 4. If this tier is successful, build the legs and BREAK the loop.
                    legs = [
                        {'type': 'put', 'direction': 'short', 'strike_price': strike_short_put, 'days_to_expiry': days_to_expiry},
                        {'type': 'call', 'direction': 'short', 'strike_price': strike_short_call, 'days_to_expiry': days_to_expiry},
                        {'type': 'put', 'direction': 'long', 'strike_price': strike_long_put, 'days_to_expiry': days_to_expiry},
                        {'type': 'call', 'direction': 'long', 'strike_price': strike_long_call, 'days_to_expiry': days_to_expiry}
                    ]
                    #print(f"DEBUG: Successfully found strikes for tier {tier['short']}/{tier['long']}.")
                    break # Exit the loop as we have found a valid strategy
            
            if not legs:
                 print(f"Warning: All dynamic delta tiers for Iron Condor failed. Falling back to fixed-width.")

        # --- Fallback Logic ---
        # This block is now triggered for a LONG condor OR if all dynamic tiers failed for a SHORT condor.
        if not legs:
            atm_price = self.market_rules_manager.get_atm_price(current_price)
            body_direction = 'long' if direction == 'long' else 'short'
            wing_direction = 'short' if direction == 'long' else 'long'
            
            legs = [
                {'type': 'put', 'direction': body_direction, 'strike_price': atm_price - self.strike_distance, 'days_to_expiry': days_to_expiry},
                {'type': 'call', 'direction': body_direction, 'strike_price': atm_price + self.strike_distance, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': wing_direction, 'strike_price': atm_price - (2 * self.strike_distance), 'days_to_expiry': days_to_expiry},
                {'type': 'call', 'direction': wing_direction, 'strike_price': atm_price + (2 * self.strike_distance), 'days_to_expiry': days_to_expiry}
            ]

        # --- Finalize the Trade (Unchanged) ---
        for leg in legs:
            leg['entry_step'] = current_step
        
        legs = self._price_legs(legs, current_price, iv_bin_index)
        pnl  = self._calculate_universal_risk_profile(legs, self.realized_pnl)
        pnl['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(legs, pnl)

    def _open_condor(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """Opens a four-leg Call or Put Condor with extensive debugging."""
        # <<< --- DEBUG PRINT 6: Entry Point --- >>>
        #print(f"\n[DEBUG] --- Entering _open_condor ---")
        #print(f"[DEBUG] Action: {action_name}")

        if len(self.portfolio) > self.max_positions - 4:
            print("[DEBUG] Aborting: Portfolio is too full.")
            return

        parts = action_name.split('_')
        direction = parts[1].lower()
        option_type = parts[2].lower()

        atm_price = self.market_rules_manager.get_atm_price(current_price)
        
        s1 = atm_price - (2 * self.strike_distance)
        s2 = atm_price - (1 * self.strike_distance)
        s3 = atm_price + (1 * self.strike_distance)
        s4 = atm_price + (2 * self.strike_distance)

        # <<< --- DEBUG PRINT 7: Strike Calculation --- >>>
        #print(f"[DEBUG] ATM: {atm_price:.2f}, Strike Distance: {self.strike_distance}")
        #print(f"[DEBUG] Calculated Strikes: s1={s1:.2f}, s2={s2:.2f}, s3={s3:.2f}, s4={s4:.2f}")

        if s1 <= 0:
            print(f"[DEBUG] Aborting: Invalid strike price calculated (s1={s1}).")
            return

        # ... (leg definition logic is unchanged) ...
        body_direction = 'long' if direction == 'long' else 'short'
        wing_direction = 'short' if direction == 'long' else 'long'
        legs_def = [
            {'type': option_type, 'direction': wing_direction, 'strike_price': s1},
            {'type': option_type, 'direction': body_direction, 'strike_price': s2},
            {'type': option_type, 'direction': body_direction, 'strike_price': s3},
            {'type': option_type, 'direction': wing_direction, 'strike_price': s4}
        ]
        
        for leg in legs_def:
            leg['entry_step'] = current_step
            leg['days_to_expiry'] = days_to_expiry
        
        should_check_rule = (direction == 'short')
        
        #print(f"[DEBUG] Calling _price_legs with should_check_rule={should_check_rule}")
        priced_legs = self._price_legs(legs_def, current_price, iv_bin_index, check_short_rule=should_check_rule)
        
        if not priced_legs:
            # <<< --- DEBUG PRINT 8: Failure Point --- >>>
            print("[DEBUG] !!! ABORTING _open_condor because _price_legs returned None.")
            return

        #print("[DEBUG] _open_condor proceeding to calculate PnL and execute trades.")
        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(priced_legs, pnl_profile)
        #print(f"[DEBUG] --- _open_condor finished successfully. --- {len(self.portfolio)}")

    def _open_iron_fly(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """
        Opens an Iron Fly with a tiered, dynamic search for the optimal wing width.
        - SHORT Iron Fly (credit): Tries to set wings based on premium, then walks inward to find a valid width.
        - LONG Iron Fly (debit): Uses a fixed width.
        """
        if len(self.portfolio) > self.max_positions - 4: return
        
        direction = 'long' if 'LONG' in action_name else 'short'
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        legs = []

        if direction == 'short':
            # --- THE NEW TIERED SEARCH LOGIC ---
            
            # 1. First, define and price the body to get the net credit.
            body_legs = [
                {'type': 'call', 'direction': 'short', 'strike_price': atm_price, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': 'short', 'strike_price': atm_price, 'days_to_expiry': days_to_expiry}
            ]
            body_legs = self._price_legs(body_legs, current_price, iv_bin_index)
            net_credit_received = body_legs[0]['entry_premium'] + body_legs[1]['entry_premium']
            
            # 2. Define the search boundaries for the wing width.
            ideal_width = round(net_credit_received / self.strike_distance) * self.strike_distance
            MAX_SENSIBLE_WIDTH = 22 * self.strike_distance
            MIN_SENSIBLE_WIDTH = 5 * self.strike_distance

            # 3. Create a prioritized list of widths to try, walking inward from the ideal/max.
            start_width = min(ideal_width, MAX_SENSIBLE_WIDTH)
            
            # Convert to integer multipliers for a clean loop
            start_multiplier = int(start_width / self.strike_distance)
            end_multiplier = int(MIN_SENSIBLE_WIDTH / self.strike_distance)

            # 4. Loop from the widest sensible width down to the narrowest.
            for width_multiplier in range(start_multiplier, end_multiplier - 1, -1):
                wing_width = width_multiplier * self.strike_distance
                
                # The loop itself is the guard rail. We know the width is valid.
                strike_long_put = atm_price - wing_width
                strike_long_call = atm_price + wing_width
                
                wing_legs = [
                    {'type': 'put', 'direction': 'long', 'strike_price': strike_long_put, 'days_to_expiry': days_to_expiry},
                    {'type': 'call', 'direction': 'long', 'strike_price': strike_long_call, 'days_to_expiry': days_to_expiry}
                ]
                
                # Assemble the full strategy and break the loop on the first success.
                legs = body_legs + wing_legs
                # print(f"DEBUG: Found valid Iron Fly width: {wing_width}") # Optional for debugging
                break # We found the best possible fit, so we exit.
            
            if not legs:
                print(f"Warning: All dynamic wing widths for Iron Fly were out of bounds. Falling back to fixed-width.")

        # --- Fallback Logic for LONG fly or if the tiered search failed ---
        if not legs:
            body_direction = 'long' if direction == 'long' else 'short'
            wing_direction = 'short' if direction == 'long' else 'long'
            wing_width = self.strike_distance # Safe, fixed 1-strike width
            
            strike_long_put = atm_price - wing_width
            strike_long_call = atm_price + wing_width

            legs = [
                {'type': 'call', 'direction': body_direction, 'strike_price': atm_price, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': body_direction, 'strike_price': atm_price, 'days_to_expiry': days_to_expiry},
                {'type': 'call', 'direction': wing_direction, 'strike_price': strike_long_call, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': wing_direction, 'strike_price': strike_long_put, 'days_to_expiry': days_to_expiry}
            ]
        
        # --- Finalize the Trade (Unchanged) ---
        for leg in legs:
            leg['entry_step'] = current_step
        
        legs = self._price_legs(legs, current_price, iv_bin_index)
        pnl  = self._calculate_universal_risk_profile(legs, self.realized_pnl)
        pnl['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(legs, pnl)

    def _find_best_available_spread(
        self, option_type: str, direction_name: str,
        current_price: float, iv_bin_index: int, days_to_expiry: float,
        is_credit_spread: bool
    ) -> List[Dict] or None:
        """A tiered solver that searches for the best possible vertical spread."""
        target_ratios = [3.0, 2.0, 1.0]
        tolerance = 0.3
        
        atm_strike = self.market_rules_manager.get_atm_price(current_price)
        anchor_direction = 'short' if is_credit_spread else 'long'
        anchor_leg = {'type': option_type, 'direction': anchor_direction, 'strike_price': atm_strike}

        for target_rr in target_ratios: # This loop is now correctly placed
            max_search_strikes = 20
            for i in range(1, max_search_strikes + 1):
                wing_direction = 'long' if is_credit_spread else 'short'
                strike_offset = i * self.strike_distance
                
                if direction_name in ['BULL_CALL_SPREAD', 'BEAR_CALL_SPREAD']:
                    wing_strike = atm_strike + strike_offset
                else:
                    wing_strike = atm_strike - strike_offset
                
                if wing_strike <= 0 or wing_strike == anchor_leg['strike_price']:
                    continue

                wing_leg = {'type': option_type, 'direction': wing_direction, 'strike_price': wing_strike}
                
                candidate_legs_def = [anchor_leg.copy(), wing_leg.copy()]
                for leg in candidate_legs_def: leg['days_to_expiry'] = days_to_expiry

                priced_legs = self._price_legs(candidate_legs_def, current_price, iv_bin_index, check_short_rule=is_credit_spread)
                if not priced_legs: continue
                
                net_premium = sum(leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1) for leg in priced_legs)
                spread_width = abs(anchor_leg['strike_price'] - wing_leg['strike_price'])
                
                if is_credit_spread:
                    max_profit = abs(net_premium)
                    max_loss = spread_width - max_profit
                else:
                    max_loss = abs(net_premium)
                    max_profit = spread_width - max_loss
                    
                if max_loss < 1e-6: continue
                
                current_rr = max_profit / max_loss
                
                if abs(current_rr - target_rr) < tolerance:
                    return priced_legs

        return None

    def _open_butterfly(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """
        Opens a three-strike butterfly with a dynamic width. It searches for the
        wing width that results in a net debit (cost) closest to a predefined
        target cost. This version is hardened against pricing failures.
        """
        if len(self.portfolio) > self.max_positions - 4:
            return

        parts = action_name.split('_')
        direction, option_type = parts[1].lower(), parts[2].lower()

        # --- 1. Setup the Search ---
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        target_cost = atm_price * self.butterfly_target_cost_pct

        wing_direction = 'long' if direction == 'long' else 'short'
        body_direction = 'short' if direction == 'long' else 'long'
        
        # A SHORT butterfly is a CREDIT trade, so we must check the short premium rule.
        should_check_rule = (direction == 'short')

        body_legs_def = [
            {'type': option_type, 'direction': body_direction, 'strike_price': atm_price, 'days_to_expiry': days_to_expiry},
            {'type': option_type, 'direction': body_direction, 'strike_price': atm_price, 'days_to_expiry': days_to_expiry}
        ]

        best_legs_found = None
        smallest_cost_diff = float('inf')

        # --- 2. Search Loop ---
        max_search_width = 20
        for i in range(1, max_search_width + 1):
            wing_width = i * self.strike_distance
            strike_lower = atm_price - wing_width
            strike_upper = atm_price + wing_width

            if strike_lower <= 0: continue

            wing_legs_def = [
                {'type': option_type, 'direction': wing_direction, 'strike_price': strike_lower, 'days_to_expiry': days_to_expiry},
                {'type': option_type, 'direction': wing_direction, 'strike_price': strike_upper, 'days_to_expiry': days_to_expiry}
            ]

            candidate_legs_def = body_legs_def + wing_legs_def
            
            # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
            priced_legs = self._price_legs(candidate_legs_def, current_price, iv_bin_index, check_short_rule=should_check_rule)
            
            # CRITICAL: If pricing failed for this candidate, skip to the next width.
            if not priced_legs:
                continue

            net_premium = sum(
                leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1)
                for leg in priced_legs
            )

            # For a LONG butterfly (debit trade), we only care about positive net premiums (costs)
            if direction == 'long' and net_premium <= 0:
                continue
            
            # For a SHORT butterfly (credit trade), we only care about negative net premiums (credits)
            if direction == 'short' and net_premium >= 0:
                continue

            cost_diff = abs(abs(net_premium) - target_cost)
            if cost_diff < smallest_cost_diff:
                smallest_cost_diff = cost_diff
                best_legs_found = priced_legs

        # --- 3. Finalize the Trade ---
        if best_legs_found:
            for leg in best_legs_found:
                leg['entry_step'] = current_step

            pnl_profile = self._calculate_universal_risk_profile(best_legs_found, self.realized_pnl)
            pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
            self._execute_trades(best_legs_found, pnl_profile)
        else:
            # This is the debug print you saw. It is now correctly the final step before exiting.
            print(f"DEBUG: no suitable legs for butterfly were found for {action_name}")

    def convert_to_iron_condor(self, current_price: float, iv_bin_index: int, current_step: int):
        """Adds long wings to a short strangle using the atomic transformation pattern."""
        current_id = self.portfolio.iloc[0]['strategy_id']
        current_name = self.id_to_strategy_name.get(current_id, '')
        if len(self.portfolio) != 2 or 'OPEN_SHORT_STRANGLE' not in current_name: return
        if len(self.portfolio) > self.max_positions - 2: return

        original_legs = self.portfolio.to_dict(orient='records')
        original_creation_id = original_legs[0]['creation_id']
        days_to_expiry = original_legs[0]['days_to_expiry']
        
        put_leg = next(leg for leg in original_legs if leg['type'] == 'put')
        call_leg = next(leg for leg in original_legs if leg['type'] == 'call')
        wing_width = self.strike_distance * 10
        strike_long_put = put_leg['strike_price'] - wing_width
        strike_long_call = call_leg['strike_price'] + wing_width
        if strike_long_put <= 0: return

        new_wing_legs_def = [
            {'type': 'put', 'direction': 'long', 'strike_price': strike_long_put, 'entry_step': current_step, 'days_to_expiry': days_to_expiry},
            {'type': 'call', 'direction': 'long', 'strike_price': strike_long_call, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}
        ]

        # A short strangle converts to a SHORT iron condor (credit trade), so we check the rule.
        priced_wings = self._price_legs(new_wing_legs_def, current_price, iv_bin_index, check_short_rule=True)
        if not priced_wings: return

        final_condor_legs = original_legs + priced_wings
        
        pnl_profile = self._calculate_universal_risk_profile(final_condor_legs, self.realized_pnl)

        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_SHORT_IRON_CONDOR')
        
        self.portfolio = self.portfolio[self.portfolio['creation_id'] != original_creation_id].reset_index(drop=True)
        self._execute_trades(final_condor_legs, pnl_profile)

    def convert_to_condor(self, option_type: str, current_price: float, iv_bin_index: int, current_step: int):
        """
        Converts an existing 2-leg vertical spread into a 4-leg Condor by adding
        a corresponding spread on the other side.
        """
        # --- 1. Guard Clauses: Ensure the action is legal ---
        if len(self.portfolio) != 2: return
        if not all(self.portfolio['type'] == option_type): return
        if len(self.portfolio['direction'].unique()) != 2: return
        
        # Check if there are enough available slots for the 2 new legs.
        if len(self.portfolio) > self.max_positions - 2:
            return

        # --- 2. Setup and Leg Identification ---
        original_legs = self.portfolio.to_dict(orient='records')
        original_creation_id = original_legs[0]['creation_id']
        days_to_expiry = original_legs[0]['days_to_expiry']
        
        strikes = sorted([leg['strike_price'] for leg in original_legs])
        lower_strike, upper_strike = strikes[0], strikes[1]
        
        lower_strike_leg_direction = next(leg['direction'] for leg in original_legs if leg['strike_price'] == lower_strike)
        upper_strike_leg_direction = next(leg['direction'] for leg in original_legs if leg['strike_price'] == upper_strike)

        is_bull_spread = False
        if option_type == 'call':
            if lower_strike_leg_direction == 'long' and upper_strike_leg_direction == 'short':
                is_bull_spread = True
        else: # put
            if upper_strike_leg_direction == 'short' and lower_strike_leg_direction == 'long':
                is_bull_spread = True
        
        # --- 3. Define the New Legs to be Added ---
        spread_width = upper_strike - lower_strike
        new_legs_def = []
        if is_bull_spread:
            new_lower_strike = upper_strike + self.strike_distance
            new_upper_strike = new_lower_strike + spread_width
            
            short_dir = 'short' if option_type == 'call' else 'long'
            long_dir = 'long' if option_type == 'call' else 'short'
            
            new_legs_def = [
                {'type': option_type, 'direction': short_dir, 'strike_price': new_lower_strike},
                {'type': option_type, 'direction': long_dir, 'strike_price': new_upper_strike}
            ]
        else: # is_bear_spread
            new_upper_strike = lower_strike - self.strike_distance
            new_lower_strike = new_upper_strike - spread_width
            if new_lower_strike <= 0: return

            long_dir = 'long' if option_type == 'call' else 'short'
            short_dir = 'short' if option_type == 'call' else 'long'

            new_legs_def = [
                {'type': option_type, 'direction': long_dir, 'strike_price': new_lower_strike},
                {'type': option_type, 'direction': short_dir, 'strike_price': new_upper_strike}
            ]

        # --- 4. Price, Assemble, and Execute ---
        for leg in new_legs_def:
            leg.update({'entry_step': current_step, 'days_to_expiry': days_to_expiry})
        
        priced_new_legs = self._price_legs(new_legs_def, current_price, iv_bin_index)
        if not priced_new_legs: return

        final_condor_legs = original_legs + priced_new_legs
        
        final_direction = 'SHORT' if original_legs[0]['strategy_max_loss'] < 0 else 'LONG'
        final_strategy_name = f'OPEN_{final_direction}_{option_type.upper()}_CONDOR'
        
        pnl_profile = self._calculate_universal_risk_profile(final_condor_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get(final_strategy_name, -1)
        
        self.portfolio = self.portfolio[self.portfolio['creation_id'] != original_creation_id].reset_index(drop=True)
        self._execute_trades(final_condor_legs, pnl_profile)

    def convert_to_iron_fly(self, current_price: float, iv_bin_index: int, current_step: int):
        """
        Adds long wings to a short straddle to define its risk.
        This definitive version includes robust failsafes and debug printing.
        """
        # --- 1. Failsafe Checks and Setup ---
        current_id = self.portfolio.iloc[0]['strategy_id']
        current_name = self.id_to_strategy_name.get(current_id, '')
        if len(self.portfolio) != 2 or current_name != 'OPEN_SHORT_STRADDLE':
            # This is not a bug, just the agent choosing an illegal action. No print needed.
            return
        if len(self.portfolio) > self.max_positions - 2:
            print("DEBUG: convert_to_iron_fly aborted. Not enough available positions.")
            return

        original_legs = self.portfolio.to_dict(orient='records')
        original_creation_id = original_legs[0]['creation_id']
        days_to_expiry = original_legs[0]['days_to_expiry']
        atm_strike = original_legs[0]['strike_price']
        
        # --- 2. Dynamically Determine Wing Width ---
        current_straddle_credit = sum(self.bs_manager.get_price_with_spread(self.bs_manager.get_all_greeks_and_price(current_price, leg['strike_price'], days_to_expiry, self.iv_calculator(0, leg['type']), leg['type'] == 'call')['price'], is_buy=True, bid_ask_spread_pct=self.bid_ask_spread_pct) for leg in original_legs)
        wing_width = self.market_rules_manager._round_to_strike_increment(current_straddle_credit)
        
        # --- 3. THE FIX: Add Robust Safety Checks and Debug Prints ---
        strike_long_put = atm_strike - wing_width
        
        if strike_long_put <= self.strike_distance: # Ensure strike is positive and not too close
            print(f"DEBUG: convert_to_iron_fly aborted. Calculated wing width ({wing_width}) is too large for the current ATM strike ({atm_strike}), resulting in an invalid put strike ({strike_long_put}).")
            return # Abort the action
        
        strike_long_call = atm_strike + wing_width

        # --- 4. Define, Price, and Assemble the Final Strategy ---
        new_wing_legs_def = [
            {'type': 'put', 'direction': 'long', 'strike_price': strike_long_put, 'entry_step': current_step, 'days_to_expiry': days_to_expiry},
            {'type': 'call', 'direction': 'long', 'strike_price': strike_long_call, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}
        ]

        # A short straddle converts to a SHORT iron fly (credit trade), so we check the rule.
        priced_wings = self._price_legs(new_wing_legs_def, current_price, iv_bin_index, check_short_rule=True)
        if not priced_wings: return
        
        final_fly_legs = original_legs + priced_wings
        
        # --- 5. Calculate Unified Profile and Atomically Update Portfolio ---
        pnl_profile = self._calculate_universal_risk_profile(final_fly_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_SHORT_IRON_FLY')
        
        self.portfolio = self.portfolio[self.portfolio['creation_id'] != original_creation_id].reset_index(drop=True)
        self._execute_trades(final_fly_legs, pnl_profile)

    # --- "REMOVE LEGS" ACTIONS (FILTER, CLOSE, AND REBUILD PATTERN) ---

    def convert_to_strangle(self, current_price: float, iv_bin_index: int, current_step: int):
        """Removes the long wings from an Iron Condor to become a strangle."""
        if len(self.portfolio) != 4: return
        original_creation_id = self.portfolio.iloc[0]['creation_id']
        legs_to_keep = self.portfolio[self.portfolio['direction'] == 'short'].to_dict(orient='records')
        legs_to_close = self.portfolio[self.portfolio['direction'] == 'long']
        self._process_leg_closures(legs_to_close, current_price, current_step)
        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_SHORT_STRANGLE_DELTA_20')
        self.portfolio = self.portfolio[self.portfolio['creation_id'] != original_creation_id].reset_index(drop=True)
        self._execute_trades(legs_to_keep, pnl_profile)

    def convert_to_straddle(self, current_price: float, iv_bin_index: int, current_step: int):
        """Removes the long wings from an Iron Fly to become a straddle."""
        if len(self.portfolio) != 4: return
        original_creation_id = self.portfolio.iloc[0]['creation_id']
        legs_to_keep = self.portfolio[self.portfolio['direction'] == 'short'].to_dict(orient='records')
        legs_to_close = self.portfolio[self.portfolio['direction'] == 'long']
        self._process_leg_closures(legs_to_close, current_price, current_step)
        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_SHORT_STRADDLE')
        self.portfolio = self.portfolio[self.portfolio['creation_id'] != original_creation_id].reset_index(drop=True)
        self._execute_trades(legs_to_keep, pnl_profile)

    def convert_to_bull_call_spread(self, current_price: float, iv_bin_index: int, current_step: int):
        """
        Decomposes a 4-leg position into its Bull Call Spread component.
        - For an Iron Condor, this closes the put legs.
        - For a Call Butterfly/Condor, this keeps the two lowest-strike calls.
        """
        if len(self.portfolio) != 4: return
        
        call_legs = self.portfolio[self.portfolio['type'] == 'call']
        if len(call_legs) < 2: return # Not enough calls to form a spread
        
        legs_to_keep_df = call_legs.sort_values(by='strike_price').head(2)
        legs_to_close_df = self.portfolio.drop(legs_to_keep_df.index)
        
        self._process_leg_closures(legs_to_close_df, current_price, current_step)
        
        legs_to_keep_list = legs_to_keep_df.to_dict(orient='records')
        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep_list, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_BULL_CALL_SPREAD')

        self.portfolio = pd.DataFrame()
        self._execute_trades(legs_to_keep_list, pnl_profile)

    def convert_to_bear_call_spread(self, current_price: float, iv_bin_index: int, current_step: int):
        """
        Decomposes a 4-leg position into its Bear Call Spread component.
        - For an Iron Condor, this closes the put legs.
        - For a Call Butterfly/Condor, this keeps the two highest-strike calls.
        """
        if len(self.portfolio) != 4: return
        
        call_legs = self.portfolio[self.portfolio['type'] == 'call']
        if len(call_legs) < 2: return
        
        legs_to_keep_df = call_legs.sort_values(by='strike_price').tail(2)
        legs_to_close_df = self.portfolio.drop(legs_to_keep_df.index)
        
        self._process_leg_closures(legs_to_close_df, current_price, current_step)
        
        legs_to_keep_list = legs_to_keep_df.to_dict(orient='records')
        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep_list, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_BEAR_CALL_SPREAD')

        self.portfolio = pd.DataFrame()
        self._execute_trades(legs_to_keep_list, pnl_profile)

    def convert_to_bull_put_spread(self, current_price: float, iv_bin_index: int, current_step: int):
        """
        Decomposes a 4-leg position into its Bull Put Spread component.
        - For an Iron Condor, this closes the call legs.
        - For a Put Butterfly/Condor, this keeps the two highest-strike puts.
        """
        if len(self.portfolio) != 4: return
        
        put_legs = self.portfolio[self.portfolio['type'] == 'put']
        if len(put_legs) < 2: return

        legs_to_keep_df = put_legs.sort_values(by='strike_price').tail(2)
        legs_to_close_df = self.portfolio.drop(legs_to_keep_df.index)
        
        self._process_leg_closures(legs_to_close_df, current_price, current_step)
        
        legs_to_keep_list = legs_to_keep_df.to_dict(orient='records')
        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep_list, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_BULL_PUT_SPREAD')

        self.portfolio = pd.DataFrame()
        self._execute_trades(legs_to_keep_list, pnl_profile)

    def convert_to_bear_put_spread(self, current_price: float, iv_bin_index: int, current_step: int):
        """
        Decomposes a 4-leg position into its Bear Put Spread component.
        - For an Iron Condor, this closes the call legs.
        - For a Put Butterfly/Condor, this keeps the two lowest-strike puts.
        """
        if len(self.portfolio) != 4: return
        
        put_legs = self.portfolio[self.portfolio['type'] == 'put']
        if len(put_legs) < 2: return

        legs_to_keep_df = put_legs.sort_values(by='strike_price').head(2)
        legs_to_close_df = self.portfolio.drop(legs_to_keep_df.index)
        
        self._process_leg_closures(legs_to_close_df, current_price, current_step)
        
        legs_to_keep_list = legs_to_keep_df.to_dict(orient='records')
        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep_list, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_BEAR_PUT_SPREAD')

        self.portfolio = pd.DataFrame()
        self._execute_trades(legs_to_keep_list, pnl_profile)

    def recenter_volatility_position(self, current_price: float, iv_bin_index: int, current_step: int):
        """
        Adjusts a 2-leg strangle or straddle by closing the old position and
        opening a new one with the SAME STRIKE WIDTH, perfectly centered
        around the new at-the-money price.
        """
        # --- 1. Guard Clauses & Get Original Position Info ---
        if len(self.portfolio) != 2 or not (self.portfolio['type'].isin(['call', 'put']).all() and len(self.portfolio['type'].unique()) == 2):
            return

        original_legs = self.portfolio.to_dict('records')
        original_creation_id = original_legs[0]['creation_id']
        original_strategy_id = original_legs[0]['strategy_id']
        days_to_expiry = original_legs[0]['days_to_expiry']
        
        call_leg_before = next(leg for leg in original_legs if leg['type'] == 'call')
        put_leg_before = next(leg for leg in original_legs if leg['type'] == 'put')
        
        # --- 2. Calculate New Strikes based on Original Width ---
        original_width = abs(call_leg_before['strike_price'] - put_leg_before['strike_price'])
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        
        new_put_strike = self.market_rules_manager._round_to_strike_increment(atm_price - (original_width / 2))
        new_call_strike = self.market_rules_manager._round_to_strike_increment(atm_price + (original_width / 2))

        if (new_put_strike == put_leg_before['strike_price'] and new_call_strike == call_leg_before['strike_price']):
            return # No change needed

        # --- 3. Close the old position ---
        self.close_all_positions(current_price, iv_bin_index, current_step)

        # --- 4. Define and open the new, re-centered position ---
        new_legs_def = [
            {'type': 'put', 'direction': put_leg_before['direction'], 'strike_price': new_put_strike},
            {'type': 'call', 'direction': call_leg_before['direction'], 'strike_price': new_call_strike}
        ]
        for leg in new_legs_def:
            leg.update({'days_to_expiry': days_to_expiry, 'entry_step': current_step})

        should_check_rule = (put_leg_before['direction'] == 'short')
        priced_new_legs = self._price_legs(new_legs_def, current_price, iv_bin_index, check_short_rule=should_check_rule)
        
        if not priced_new_legs:
            print("DEBUG: Re-centering failed due to illegal new legs. Position is now flat.")
            return

        # --- 5. Finalize the New Position ---
        # The portfolio is now empty. We create the new one, preserving the original strategy ID.
        pnl_profile = self._calculate_universal_risk_profile(priced_new_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = original_strategy_id
        self._execute_trades(priced_new_legs, pnl_profile)

    def debug_print_portfolio(self, current_price: float, step: int, day: int, action_taken: str):
        return
        """
        Prints a detailed, human-readable snapshot of the current portfolio state,
        including all stored data and live calculated values for each leg.
        This method is only active when is_eval_mode is True.
        """
        # --- 1. Guard Clause: Only run in evaluation mode ---
        if not self.is_eval_mode:
            return

        print("\n" + "="*120)
        print(f"--- Portfolio Debug Snapshot (After Action: '{action_taken}' at Step: {step}, Day: {day}) ---")

        # --- 2. Handle Empty Portfolio ---
        if self.portfolio.empty:
            print("Portfolio is empty.")
            print("="*120 + "\n")
            return

        # --- 3. Prepare to Collect Detailed Data ---
        debug_data = []
        atm_price = self.market_rules_manager.get_atm_price(current_price)

        # --- 4. Loop Through Portfolio and Calculate Live Data ---
        for _, leg in self.portfolio.iterrows():
            is_call = leg['type'] == 'call'

            # Get live Greeks and premium
            offset = round((leg['strike_price'] - atm_price) / self.strike_distance)
            vol = self.iv_calculator(offset, leg['type'])
            greeks = self.bs_manager.get_all_greeks_and_price(
                current_price, leg['strike_price'], leg['days_to_expiry'], vol, is_call
            )
            current_premium = self.bs_manager.get_price_with_spread(
                greeks['price'], is_buy=(leg['direction'] == 'short'), bid_ask_spread_pct=self.bid_ask_spread_pct
            )

            # Calculate live P&L for this leg
            direction_multiplier = 1 if leg['direction'] == 'long' else -1
            pnl_per_share = current_premium - leg['entry_premium']
            live_pnl = pnl_per_share * direction_multiplier * self.lot_size

            # --- 5. Assemble Data for this Leg ---
            leg_details = {
                "ID": leg['creation_id'],
                "Strat_ID": leg['strategy_id'],
                "Type": f"{leg['direction'].upper()} {leg['type'].upper()}",
                "Strike": leg['strike_price'],
                "Entry Prem": leg['entry_premium'],
                "Current Prem": current_premium,
                "Live PnL": live_pnl,
                "DTE": leg['days_to_expiry'],
                "Hedged": leg['is_hedged'],
                "Delta": greeks['delta'] * self.lot_size * direction_multiplier,
                "Gamma": greeks['gamma'] * (self.lot_size**2 * current_price**2 / 100) * direction_multiplier,
                "Theta": greeks['theta'] * self.lot_size * direction_multiplier,
                "Vega": greeks['vega'] * self.lot_size * direction_multiplier,
            }
            debug_data.append(leg_details)

        # --- 6. Format and Print the DataFrame ---
        debug_df = pd.DataFrame(debug_data)

        # Set formatting for better readability
        pd.options.display.float_format = '{:,.2f}'.format

        print(debug_df.to_string())
        print("="*120 + "\n")
