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
                denominator = total_pnl - pnl_prev
                if abs(denominator) > 1e-9:
                    be = price_prev - pnl_prev * (price_curr - price_prev) / denominator
                    if np.isfinite(be):
                        breakevens.append(be)
        pnl_prev = total_pnl

    # --- DEFINITIVE FIX AT THE SOURCE ---
    if np.any(np.isfinite(pnl_values)):
        max_profit = np.nanmax(pnl_values)
        max_loss = np.nanmin(pnl_values)
    else:
        max_profit = realized_pnl
        max_loss = realized_pnl

    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values <= 0]
    sum_wins = np.sum(wins)
    sum_losses = np.abs(np.sum(losses))

    profit_factor = 0.0
    if sum_wins > 1e-6:
        if sum_losses > 1e-6:
            profit_factor = sum_wins / sum_losses
        else:
            profit_factor = 999.0

    max_profit = max_profit if np.isfinite(max_profit) else realized_pnl
    max_loss = max_loss if np.isfinite(max_loss) else realized_pnl

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
                 strategy_name_to_id: Dict,
                 np_random: any):
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
        self.np_random = np_random

        # 1. Pre-calculate the 5-day opportunity cost of the initial capital.
        # This becomes our minimum profit hurdle for any new trade.
        annual_risk_free_rate = cfg.get('risk_free_rate', 0.10)
        thirty_day_rate = (1 + annual_risk_free_rate)**(5 / 365.25) - 1
        self.min_profit_hurdle = cfg['initial_cash'] * thirty_day_rate
        
        #print(f"[PortfolioManager] Initialized with a minimum profit hurdle of ${self.min_profit_hurdle:,.2f}")

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

        self.current_portfolio_stance: str = "FLAT"

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
        # Butterflies
        for opt_type in ['CALL', 'PUT']:
            action_pairs.append(
                (f'OPEN_LONG_{opt_type}_FLY', f'OPEN_SHORT_{opt_type}_FLY')
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

    def _set_new_portfolio_state(self, legs_to_set: List[Dict], strategy_pnl: Dict):
        """
        A safe, internal helper that atomically sets the portfolio to a new state.
        It handles NO financial calculations (e.g., brokerage). Its only job is
        to correctly structure and assign the new DataFrame and update hedge status.
        """
        # This is a failsafe in case this is called with an empty list.
        if not legs_to_set:
            self.portfolio = pd.DataFrame(columns=self.portfolio_columns).astype(self.portfolio_dtypes)
            return

        transaction_id = self.next_creation_id
        self.next_creation_id += 1
        
        strategy_id = strategy_pnl.get('strategy_id', -1)
        assert strategy_id != -1, "CRITICAL ERROR: Strategy ID not found for PnL object."

        for trade in legs_to_set:
            # Only assign a new creation_id if one doesn't already exist from a previous state.
            if 'creation_id' not in trade:
                trade['creation_id'] = transaction_id
            
            trade['is_hedged'] = False # Default before update
            trade['strategy_id'] = strategy_id
            trade['strategy_max_profit'] = strategy_pnl.get('strategy_max_profit', 0.0)
            trade['strategy_max_loss'] = strategy_pnl.get('strategy_max_loss', 0.0)
        
        new_positions_df = pd.DataFrame(legs_to_set).astype(self.portfolio_dtypes)
        self.portfolio = new_positions_df # Atomically replace the old portfolio
        self._update_hedge_status()

    # <<< --- NEW: The Central Sanitization Function --- >>>
    def _sanitize_dict(self, data_dict: Dict) -> Dict:
        """
        Aggressively sanitizes a dictionary to ensure all numerical
        values are finite and JSON-compatible. This is the final gate.
        """
        sanitized = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                # Recursively sanitize nested dictionaries (like sigma_levels)
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, (list, np.ndarray)):
                # Sanitize lists (like breakevens or pnl curves)
                sanitized_list = []
                for item in value:
                    if isinstance(item, dict):
                        sanitized_list.append(self._sanitize_dict(item))
                    elif isinstance(item, (float, np.floating, int, np.integer)):
                        sanitized_list.append(float(item) if np.isfinite(item) else None)
                    else:
                        sanitized_list.append(item)
                sanitized[key] = sanitized_list
            elif isinstance(value, (float, np.floating, int, np.integer)):
                # Use None (which becomes JSON null) for invalid numbers
                sanitized[key] = float(value) if np.isfinite(value) else None
            else:
                sanitized[key] = value
        return sanitized

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

        # Ensure the stance is reset for each new episode
        self.current_portfolio_stance = "FLAT"

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
        
        self._process_leg_closures(leg_to_roll_df, current_price, iv_bin_index, current_step)
        
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

            # 1. Handle financial logic for the NEW leg only.
            self.realized_pnl -= self.brokerage_per_leg

            # 2. Assemble the complete, final list of all legs.
            final_legs_list = untouched_legs.to_dict('records') + sibling_legs.to_dict('records') + leg_to_hedge_df.to_dict('records') + priced_hedge_leg

            modified_strategy_legs_list = sibling_legs.to_dict('records') + leg_to_hedge_df.to_dict('records') + priced_hedge_leg
            
            if len(modified_strategy_legs_list) == 2:
                # If the result is a 2-leg position, it's a standard vertical spread.
                temp_df = pd.DataFrame(modified_strategy_legs_list)
                new_strategy_name = self._identify_strategy_from_legs(temp_df)
            else:
                # If the result is a 3+ leg position, it's a custom hedged structure.
                new_strategy_name = f"CUSTOM_{len(modified_strategy_legs_list)}_LEGS"

            # 3. Calculate the unified profile and set the new state.
            pnl_profile = self._calculate_universal_risk_profile(final_legs_list, self.realized_pnl)
            pnl_profile['strategy_id'] = self.strategy_name_to_id.get(new_strategy_name, -1)
            self._set_new_portfolio_state(final_legs_list, pnl_profile)
            
        except (ValueError, IndexError, KeyError) as e:
            print(f"Warning: Could not execute add_hedge action. Error: {e}")

    def _open_best_available_vertical(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """
        Public method to execute the opening of a vertical spread.
        It first tries to find an ideal spread using the tiered R:R solver.
        If that fails, it now robustly falls back to creating a fixed-width spread.
        """
        # --- 1. Parse Action Intent ---
        parts = action_name.replace('OPEN_', '').split('_')
        direction_name = '_'.join(parts[0:3]) # e.g., BULL_CALL_SPREAD
        option_type = parts[1].lower()
        is_credit_spread = 'BULL_PUT' in direction_name or 'BEAR_CALL' in direction_name

        # --- 2. Attempt to Find the Ideal Spread using the Solver ---
        found_legs = self._find_best_available_spread(
            option_type, direction_name, current_price, iv_bin_index, days_to_expiry, is_credit_spread
        )

        # <<< --- THE DEFINITIVE FIX IS HERE: Implement a robust fallback --- >>>
        if not found_legs:
            # This block now runs if the intelligent solver fails to find a perfect R:R match.
            # print(f"DEBUG: Tiered R:R solver failed for {action_name}. Using fixed-width fallback.")
            
            atm_strike = self.market_rules_manager.get_atm_price(current_price)
            # Use a wide, robust default width that is guaranteed to be available.
            wing_offset = self.strike_distance * 10 
            
            anchor_strike = 0
            wing_strike = 0
            anchor_dir = ''
            wing_dir = ''

            # This logic explicitly handles all four cases correctly.
            if direction_name == 'BULL_CALL_SPREAD': # Debit, Bullish
                anchor_strike, wing_strike = atm_strike, atm_strike + wing_offset
                anchor_dir, wing_dir = 'long', 'short'
            elif direction_name == 'BEAR_PUT_SPREAD': # Debit, Bearish
                anchor_strike, wing_strike = atm_strike, atm_strike - wing_offset
                anchor_dir, wing_dir = 'long', 'short'
            elif direction_name == 'BEAR_CALL_SPREAD': # Credit, Bearish
                anchor_strike, wing_strike = atm_strike, atm_strike + wing_offset
                anchor_dir, wing_dir = 'short', 'long'
            elif direction_name == 'BULL_PUT_SPREAD': # Credit, Bullish
                anchor_strike, wing_strike = atm_strike, atm_strike - wing_offset
                anchor_dir, wing_dir = 'short', 'long'

            if wing_strike <= 0: return False # Failsafe for invalid strikes

            fallback_legs_def = [
                {'type': option_type, 'direction': anchor_dir, 'strike_price': anchor_strike},
                {'type': option_type, 'direction': wing_dir, 'strike_price': wing_strike}
            ]
            for leg in fallback_legs_def: leg['days_to_expiry'] = days_to_expiry

            # Price the fallback legs.
            found_legs = self._price_legs(fallback_legs_def, current_price, iv_bin_index)

        # --- 4. Finalize and Execute the Trade (This part is now universal) ---
        if found_legs:
            for leg in found_legs:
                leg['entry_step'] = current_step

            pnl_profile = self._calculate_universal_risk_profile(found_legs, self.realized_pnl, is_new_trade=True)
            if pnl_profile['strategy_max_profit'] <= self.min_profit_hurdle: return False
            pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
            self._execute_trades(found_legs, pnl_profile)
            return True
        else:
            # If even the fallback fails (which is highly unlikely), abort.
            return False

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

        # We explicitly sanitize each potentially problematic value at the last possible
        # moment, providing a safe, JSON-compliant default if it is NaN, inf, or otherwise invalid.
        
        # 1. Get the raw values from the calculation functions.
        max_profit_raw = summary.get('max_profit')
        max_loss_raw = summary.get('max_loss')
        profit_factor_raw = pnl_profile.get('profit_factor')
        breakevens_raw = pnl_profile.get('breakevens')

        # 2. Assemble the final dictionary with robust, inline sanitation.
        stats = {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
            
            # --- Sanitized Values ---
            'max_profit': float(max_profit_raw) if np.isfinite(max_profit_raw) else 0.0,
            'max_loss': float(max_loss_raw) if np.isfinite(max_loss_raw) else 0.0,
            'rr_ratio': summary.get('rr_ratio', 0.0),
            'prob_profit': summary.get('prob_profit', 0.0),
            'profit_factor': float(profit_factor_raw) if np.isfinite(profit_factor_raw) else 0.0,
            
            # For breakevens, ensure it's a list and all its values are finite.
            'breakevens': [be for be in breakevens_raw if np.isfinite(be)] if isinstance(breakevens_raw, list) else [],
            
            'highest_realized_profit': self.highest_realized_profit,
            'lowest_realized_loss': self.lowest_realized_loss,
            'mtm_pnl_high': self.mtm_pnl_high,
            'mtm_pnl_low': self.mtm_pnl_low,
            'net_premium': final_net_premium,
        }

        # The _sanitize_dict provides a final layer of safety, but the inline checks are primary.
        return self._sanitize_dict(stats)

    def get_portfolio(self) -> pd.DataFrame:
        """Public API to get the portfolio state."""
        return self.portfolio

    def resolve_and_open_strategy(self, intent_or_specific_action: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float, volatility_bias: str) -> Tuple[bool, str]:
        """
        The "Resolver." Takes a high-level strategic intent from the agent
        (or a specific forced action), selects the best concrete trade to execute,
        and sets the portfolio's directional stance upon success.
        
        Returns a tuple of (was_opened_successfully, resolved_action_name).
        """
        
        # --- Part 1: Handle Direct, Low-Level Strategy Execution (The Bypass) ---
        # This is used by the regression suite and strategy analyzer.
        if 'POSITION' not in intent_or_specific_action:
            was_opened = self.open_strategy(intent_or_specific_action, current_price, iv_bin_index, current_step, days_to_expiry)
            
            # If a specific strategy was forced, we infer its stance and set the state.
            if was_opened:
                if 'BULL' in intent_or_specific_action or 'SHORT_PUT' in intent_or_specific_action or 'JADE' in intent_or_specific_action or 'BIG_LIZARD' in intent_or_specific_action or 'PUT_RATIO' in intent_or_specific_action:
                    self.current_portfolio_stance = "BULLISH"
                elif 'BEAR' in intent_or_specific_action or 'SHORT_CALL' in intent_or_specific_action or 'REVERSE' in intent_or_specific_action or 'CALL_RATIO' in intent_or_specific_action:
                    self.current_portfolio_stance = "BEARISH"
                else:
                    self.current_portfolio_stance = "NEUTRAL"

            return was_opened, intent_or_specific_action

        # --- Part 2: High-Level Intent Resolution ---
        candidate_actions = []
        new_stance = "FLAT" # Default stance

        if intent_or_specific_action == 'OPEN_BULLISH_POSITION':
            new_stance = "BULLISH"
            candidate_actions = [
                'OPEN_BULL_PUT_SPREAD', 'OPEN_SHORT_PUT_ATM', 'OPEN_BULL_CALL_SPREAD',
                'OPEN_BIG_LIZARD', 'OPEN_JADE_LIZARD', 'OPEN_PUT_RATIO_SPREAD'
            ]
        elif intent_or_specific_action == 'OPEN_BEARISH_POSITION':
            new_stance = "BEARISH"
            candidate_actions = [
                'OPEN_BEAR_CALL_SPREAD', 'OPEN_SHORT_CALL_ATM', 'OPEN_BEAR_PUT_SPREAD',
                'OPEN_REVERSE_BIG_LIZARD', 'OPEN_REVERSE_JADE_LIZARD', 'OPEN_CALL_RATIO_SPREAD'
            ]
        elif intent_or_specific_action == 'OPEN_NEUTRAL_POSITION':
            new_stance = "NEUTRAL"
            if "High" in volatility_bias:
                candidate_actions = ['OPEN_SHORT_STRADDLE', 'OPEN_SHORT_STRANGLE_DELTA_25', 'OPEN_SHORT_STRANGLE_DELTA_30', 'OPEN_SHORT_IRON_CONDOR']
            else:
                candidate_actions = ['OPEN_SHORT_STRANGLE_DELTA_25', 'OPEN_SHORT_STRANGLE_DELTA_30', 'OPEN_SHORT_IRON_CONDOR']
        
        if not candidate_actions:
            return False, "NONE"

        # The resolver's core logic: pick a random valid strategy that matches the intent.
        selected_action = self.np_random.choice(candidate_actions)
        
        #print(f"(INFO) Agent Intent: '{intent_or_specific_action}', Resolver chose to execute: '{selected_action}'")

        # Call the low-level executor.
        was_opened = self.open_strategy(selected_action, current_price, iv_bin_index, current_step, days_to_expiry)
        
        # --- Part 3: Commit the New Stance to the State Variable on Success ---
        if was_opened:
            self.current_portfolio_stance = new_stance
        
        return was_opened, selected_action

    def open_strategy(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """Routes any 'OPEN_' action to the correct specialized private method."""
        if len(self.portfolio) >= self.max_positions: return False

        disable_solver = self.cfg.get('disable_spread_solver', False)

        # Route by most complex/specific keywords first to avoid misrouting.
        
        # 1. Most specific 3-leg strategies
        if 'JADE_LIZARD' in action_name:
            return self._open_jade_lizard(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'BIG_LIZARD' in action_name:
            return self._open_big_lizard(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'RATIO_SPREAD' in action_name:
            return self._open_ratio_spread(action_name, current_price, iv_bin_index, current_step, days_to_expiry)

        # 2. Four-leg strategies
        elif 'FLY' in action_name and 'IRON' not in action_name:
            return self._open_butterfly(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'IRON_CONDOR' in action_name:
            return self._open_iron_condor(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'IRON_FLY' in action_name:
            return self._open_iron_fly(action_name, current_price, iv_bin_index, current_step, days_to_expiry)

        # 3. Two-leg volatility strategies
        elif 'STRANGLE' in action_name:
            return self._open_strangle(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'STRADDLE' in action_name:
            return self._open_straddle(action_name, current_price, iv_bin_index, current_step, days_to_expiry)

        # 4. Two-leg directional SPREADS (less specific, checked later)
        elif 'SPREAD' in action_name and not disable_solver:
            return self._open_best_available_vertical(action_name, current_price, iv_bin_index, current_step, days_to_expiry)

        # 5. Single-leg strategies (least specific, checked last)
        elif 'ATM' in action_name:
            return self._open_single_leg(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
            
        else:
            print(f"Warning: Unrecognized action format in open_strategy: {action_name}")
            return False

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
        self._process_leg_closures(pd.DataFrame([leg_to_close.to_dict()]), current_price, iv_bin_index, current_step)

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
                
                # Check for straddle/strangle (different types, same direction)
                if leg1['type'] != leg2['type'] and leg1['direction'] == leg2['direction']:
                    direction = leg1['direction'].upper()
                    if leg1['strike_price'] == leg2['strike_price']:
                        return f'OPEN_{direction}_STRADDLE'
                    else:
                        # For strangles, we can't know the delta, so we use a generic name
                        return 'OPEN_SHORT_STRANGLE_DELTA_25' # Return a valid, representative 

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
        pnl_dict = {
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'verified_total_pnl': self.realized_pnl + unrealized_pnl
        }

        return self._sanitize_dict(pnl_dict)

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
        payoff_dict = {
            'spot_price': current_price,
            'sigma_levels': sigma_levels
        }

        # <<< --- Pass the final dictionary through the sanitizer --- >>>
        return self._sanitize_dict(payoff_dict)

    # --- Private Methods ---
    def _open_jade_lizard(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """Opens a 3-leg Jade Lizard or Reverse Jade Lizard with a robust fallback for the spread component."""
        if len(self.portfolio) > self.max_positions - 3: return False

        is_reverse = 'REVERSE' in action_name
        
        # 1. Define the naked leg
        naked_leg_type = 'call' if is_reverse else 'put'
        naked_leg_delta = 0.20 if naked_leg_type == 'call' else -0.20
        naked_strike = self._find_strike_for_delta(naked_leg_delta, naked_leg_type, current_price, iv_bin_index, days_to_expiry)
        if naked_strike is None: return False

        naked_leg_def = {'type': naked_leg_type, 'direction': 'short', 'strike_price': naked_strike}

        # 2. Attempt to define the spread using the ideal R:R solver
        spread_type = 'put' if is_reverse else 'call'
        spread_direction_name = 'BULL_PUT_SPREAD' if is_reverse else 'BEAR_CALL_SPREAD'
        
        spread_legs_def = self._find_best_available_spread(spread_type, spread_direction_name, current_price, iv_bin_index, days_to_expiry, is_credit_spread=True)
        
        # <<< --- THE DEFINITIVE FIX IS HERE: Implement a robust fallback --- >>>
        if not spread_legs_def:
            # If the ideal solver fails, we construct a simple, fixed-width credit spread
            # based on the strategy's original delta definition.
            
            # Find the short leg of the spread at ~35 delta.
            short_leg_delta = -0.35 if spread_type == 'put' else 0.35
            short_strike = self._find_strike_for_delta(short_leg_delta, spread_type, current_price, iv_bin_index, days_to_expiry)
            
            if short_strike is None: return False # Abort if we can't even find the anchor leg.
            
            # Create a simple 2-strike wide spread for robustness.
            wing_width = self.strike_distance * 2
            long_strike = short_strike - wing_width if spread_type == 'put' else short_strike + wing_width
            if long_strike <= 0: return False # Failsafe for invalid strikes
            
            spread_legs_def = [
                {'type': spread_type, 'direction': 'short', 'strike_price': short_strike},
                {'type': spread_type, 'direction': 'long', 'strike_price': long_strike}
            ]
        # --- End of Fix ---
            
        # 3. Assemble, price, and execute
        final_legs_def = [naked_leg_def] + spread_legs_def
        for leg in final_legs_def:
            leg.update({'entry_step': current_step, 'days_to_expiry': days_to_expiry})
        
        priced_legs = self._price_legs(final_legs_def, current_price, iv_bin_index)
        if not priced_legs: return False

        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl, is_new_trade=True)
        # The strategy is now rejected if its max profit is not above our dynamic hurdle rate.
        if pnl_profile['strategy_max_profit'] <= self.min_profit_hurdle: return False
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(priced_legs, pnl_profile)
        return True

    def _open_big_lizard(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """Opens a 3-leg Big Lizard or Reverse Big Lizard."""
        if len(self.portfolio) > self.max_positions - 3: return False

        is_reverse = 'REVERSE' in action_name

        # 1. Define the Strangle component based on delta
        strangle_put_delta = -0.55 if is_reverse else -0.45
        strangle_call_delta = 0.45 if is_reverse else 0.55

        strike_put = self._find_strike_for_delta(strangle_put_delta, 'put', current_price, iv_bin_index, days_to_expiry)
        strike_call = self._find_strike_for_delta(strangle_call_delta, 'call', current_price, iv_bin_index, days_to_expiry)
        if strike_put is None or strike_call is None: return False
        
        strangle_legs_def = [
            {'type': 'put', 'direction': 'short', 'strike_price': strike_put},
            {'type': 'call', 'direction': 'short', 'strike_price': strike_call}
        ]
        
        # 2. Define the long hedge leg
        hedge_leg_type = 'put' if is_reverse else 'call'
        hedge_leg_delta = -0.45 if is_reverse else 0.45
        hedge_strike = self._find_strike_for_delta(hedge_leg_delta, hedge_leg_type, current_price, iv_bin_index, days_to_expiry)
        if hedge_strike is None: return False

        hedge_leg_def = {'type': hedge_leg_type, 'direction': 'long', 'strike_price': hedge_strike}

        # 3. Assemble, price, and execute
        final_legs_def = strangle_legs_def + [hedge_leg_def]
        for leg in final_legs_def:
            leg.update({'entry_step': current_step, 'days_to_expiry': days_to_expiry})
            
        priced_legs = self._price_legs(final_legs_def, current_price, iv_bin_index, check_short_rule=True)
        if not priced_legs: return False
            
        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl, is_new_trade=True)
        if pnl_profile['strategy_max_profit'] <= self.min_profit_hurdle: return False
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(priced_legs, pnl_profile)
        return True

    def _open_ratio_spread(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """Opens a 1x2 ratio spread (1 long, 2 short)."""
        if len(self.portfolio) > self.max_positions - 3: return False
        
        option_type = 'put' if 'PUT' in action_name else 'call'

        # 1. Find the two short legs @ 20 delta
        short_delta = -0.20 if option_type == 'put' else 0.20
        short_strike = self._find_strike_for_delta(short_delta, option_type, current_price, iv_bin_index, days_to_expiry)
        if short_strike is None: return False
        
        # 2. Find the one long leg @ 30-35 delta
        long_delta_range = [-0.30, -0.35] if option_type == 'put' else [0.30, 0.35]
        long_strike = self._find_strike_from_delta_list(long_delta_range, option_type, current_price, iv_bin_index, days_to_expiry)
        if long_strike is None: return False

        # 3. Assemble the 3 legs (with two identical short legs)
        final_legs_def = [
            {'type': option_type, 'direction': 'short', 'strike_price': short_strike},
            {'type': option_type, 'direction': 'short', 'strike_price': short_strike},
            {'type': option_type, 'direction': 'long', 'strike_price': long_strike}
        ]
        
        for leg in final_legs_def:
            leg.update({'entry_step': current_step, 'days_to_expiry': days_to_expiry})
            
        priced_legs = self._price_legs(final_legs_def, current_price, iv_bin_index, check_short_rule=True)
        if not priced_legs: return False
            
        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl, is_new_trade=True)
        if pnl_profile['strategy_max_profit'] <= self.min_profit_hurdle: return False
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(priced_legs, pnl_profile)
        return True

    def _find_portfolio_neutral_strike(self, leg_index: int, current_price: float, iv_bin_index: int) -> float or None:
        """
        A powerful solver that finds the optimal new strike for a specific leg that
        will bring the entire portfolio's net delta as close to zero as possible.
        MODIFIED: Now searches in a window around the leg's ORIGINAL strike price.
        """
        if not (0 <= leg_index < len(self.portfolio)): return None

        leg_to_roll = self.portfolio.iloc[leg_index]
        other_legs = self.portfolio.drop(leg_to_roll.name).to_dict('records')
        
        best_strike = None
        min_abs_portfolio_delta = abs(self.get_raw_portfolio_stats(current_price, iv_bin_index)['delta'])
        
        # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
        # 1. The search is now centered on the leg's original strike, not the ATM price.
        original_strike = leg_to_roll['strike_price']
        search_width = self.hedge_roll_search_width # e.g., 10 strikes
        
        # 2. Iterate through a window of offsets from the original strike.
        for offset in range(-search_width, search_width + 1):
            if offset == 0: continue # Skip the current strike

            candidate_strike = original_strike + (offset * self.strike_distance)
            if candidate_strike <= 0: continue

            # ... (the rest of the validation and delta calculation logic is unchanged) ...
            candidate_leg = leg_to_roll.to_dict()
            candidate_leg['strike_price'] = candidate_strike
            
            # This check for premium is still a good safety rail
            if candidate_leg['direction'] == 'short':
                is_call = candidate_leg['type'] == 'call'
                # For an accurate premium check, we need to calculate the offset from ATM for the *new* strike
                atm_price = self.market_rules_manager.get_atm_price(current_price)
                new_offset_from_atm = round((candidate_strike - atm_price) / self.strike_distance)
                vol = self.iv_calculator(new_offset_from_atm, candidate_leg['type'])
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
    def _process_leg_closures(self, legs_to_close: pd.DataFrame, current_price: float, iv_bin_index: int, current_step: int):
        """A helper to correctly process P&L and receipts for a set of closing legs."""
        if legs_to_close.empty:
            return

        atm_price = self.market_rules_manager.get_atm_price(current_price)
        for _, leg in legs_to_close.iterrows():
            # Calculate the offset from ATM to get the correct IV from the skew
            offset = round((leg['strike_price'] - atm_price) / self.strike_distance)
            
            # The iv_calculator now receives the correct, live iv_bin_index
            vol = self.iv_calculator(offset, leg['type'])
            
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
        try:
            parts = action_name.split('_')
            direction, position_index = parts[1].upper(), int(parts[3])
            if not (-len(self.portfolio) <= position_index < len(self.portfolio)): return

            # --- 1. Identify all components (Unchanged) ---
            leg_to_shift_df = self.portfolio.iloc[[position_index]]
            leg_to_shift = leg_to_shift_df.iloc[0].to_dict()
            sibling_legs = self.portfolio.drop(self.portfolio.index[position_index]) # Simplified

            # --- 2. Process the "close" part, which correctly charges exit brokerage ---
            self._process_leg_closures(leg_to_shift_df, current_price, iv_bin_index, current_step)
            
            # --- 3. Define and price the new, shifted leg (Unchanged) ---
            strike_modifier = self.strike_distance if direction == 'UP' else -self.strike_distance
            new_leg_def = [{'type': leg_to_shift['type'], 'direction': leg_to_shift['direction'],
                            'strike_price': leg_to_shift['strike_price'] + strike_modifier,
                            'days_to_expiry': leg_to_shift['days_to_expiry'], 'entry_step': current_step}]
            priced_new_leg = self._price_legs(new_leg_def, current_price, iv_bin_index)

            if not priced_new_leg: # Failsafe if the new leg can't be priced
                self._set_new_portfolio_state(sibling_legs.to_dict(orient='records'), {'strategy_id': leg_to_shift['strategy_id']})
                return

            # --- 4. Manually charge the entry brokerage for the new leg ---
            self.realized_pnl -= self.brokerage_per_leg

            # --- 5. Assemble the complete, final list of legs and set the state ---
            modified_strategy_legs = sibling_legs.to_dict(orient='records') + priced_new_leg
            pnl_profile = self._calculate_universal_risk_profile(modified_strategy_legs, self.realized_pnl)
            pnl_profile['strategy_id'] = leg_to_shift['strategy_id'] # Preserve original strategy ID
            self._set_new_portfolio_state(modified_strategy_legs, pnl_profile)

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse shift action '{action_name}'. Error: {e}")

    def shift_to_atm(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int):
        try:
            position_index = int(action_name.split('_')[-1])
            if not (-len(self.portfolio) <= position_index < len(self.portfolio)): return
            
            # --- 1. Identify all components (Unchanged) ---
            leg_to_shift_df = self.portfolio.iloc[[position_index]]
            leg_to_shift = leg_to_shift_df.iloc[0].to_dict()
            sibling_legs = self.portfolio.drop(self.portfolio.index[position_index]) # Simplified
            
            new_atm_strike = self.market_rules_manager.get_atm_price(current_price)
            if leg_to_shift['strike_price'] == new_atm_strike: return

            # --- 2. Process the "close" part (Unchanged) ---
            self._process_leg_closures(leg_to_shift_df, current_price, iv_bin_index, current_step)
            
            # --- 3. Define and price the new, shifted leg (Unchanged) ---
            new_leg_def = [{'type': leg_to_shift['type'], 'direction': leg_to_shift['direction'],
                            'strike_price': new_atm_strike, 'days_to_expiry': leg_to_shift['days_to_expiry'],
                            'entry_step': current_step}]
            priced_new_leg = self._price_legs(new_leg_def, current_price, iv_bin_index)

            if not priced_new_leg: # Failsafe
                self._set_new_portfolio_state(sibling_legs.to_dict(orient='records'), {'strategy_id': leg_to_shift['strategy_id']})
                return
            
            # --- 4. Manually charge entry brokerage ---
            self.realized_pnl -= self.brokerage_per_leg

            # --- 5. Assemble final state and use the safe state-setter ---
            modified_strategy_legs = sibling_legs.to_dict(orient='records') + priced_new_leg
            pnl_profile = self._calculate_universal_risk_profile(modified_strategy_legs, self.realized_pnl)
            pnl_profile['strategy_id'] = leg_to_shift['strategy_id']
            self._set_new_portfolio_state(modified_strategy_legs, pnl_profile)
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

        summary = {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'rr_ratio': rr_ratio,
            'prob_profit': np.clip(total_pop, 0.0, 1.0)
        }
        
        # <<< --- Pass the summary through the sanitizer before returning --- >>>
        return self._sanitize_dict(summary)

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

        # 1. Handle financial logic for the NEW leg only.
        self.realized_pnl -= self.brokerage_per_leg

        # 2. Assemble the complete, final list of all legs (old portfolio + new leg).
        final_legs_list = self.portfolio.to_dict('records') + priced_legs

        # 3. Calculate the unified profile and set the new state.
        # Since this creates a custom hedged position, we use a generic ID.
        pnl_profile = self._calculate_universal_risk_profile(final_legs_list, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('CUSTOM_HEDGED', -1)
        self._set_new_portfolio_state(final_legs_list, pnl_profile)

    def render(self, current_price: float, current_step: int, iv_bin_index: int, steps_per_day: int):
        total_pnl = self.get_total_pnl(current_price, iv_bin_index)
        day = current_step // steps_per_day + 1
        print(f"Step: {current_step:04d} | Day: {day:02d} | Price: ${current_price:9.2f} | Positions: {len(self.portfolio):1d} | Total PnL: ${total_pnl:9.2f}")
        if not self.portfolio.empty:
            print(self.portfolio.to_string(index=False))
        return

    def _execute_trades(self, trades_to_execute: List[Dict], strategy_pnl: Dict):
        """ 
        The definitive method for adding NEW legs to the portfolio. It correctly
        handles brokerage and then uses the state-setter helper.
        """
        if not trades_to_execute:
            return

        # 1. Handle financial logic: Brokerage for NEW legs.
        opening_brokerage = len(trades_to_execute) * self.brokerage_per_leg
        self.realized_pnl -= opening_brokerage
        
        per_share_net_premium = sum(
            leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1) 
            for leg in trades_to_execute
        )
        self.initial_net_premium = per_share_net_premium * self.lot_size

        # 2. Call the new helper to handle the data manipulation.
        self._set_new_portfolio_state(trades_to_execute, strategy_pnl)

    def _price_legs(self, legs: List[Dict], current_price: float, iv_bin_index: int, check_short_rule: bool = False) -> List[Dict] or None:
        """
        Calculates the entry premium for a list of legs.
        MODIFIED: Now supports an optional 'entry_premium' override. If a leg
        dictionary contains this key, its value will be used directly instead
        of being calculated via Black-Scholes.
        """
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        priced_legs = []
        for leg in legs:
            # Always work with a copy to avoid side effects on the original list
            new_leg = leg.copy()

            # <<< --- THE DEFINITIVE FIX IS HERE --- >>>
            # Check if the user has provided a manual entry premium.
            if 'entry_premium' in new_leg and new_leg['entry_premium'] is not None:
                # If a premium is provided, we skip the Black-Scholes calculation for this leg.
                # The user's value is treated as the source of truth.
                pass 
            else:
                # If no premium is provided, calculate it using the standard market logic.
                try:
                    offset = round((new_leg['strike_price'] - atm_price) / self.strike_distance)
                    vol = self.iv_calculator(offset, new_leg['type'])
                    greeks = self.bs_manager.get_all_greeks_and_price(current_price, new_leg['strike_price'], new_leg['days_to_expiry'], vol, new_leg['type'] == 'call')
                    
                    is_opening_a_long_position = (new_leg['direction'] == 'long')
                    calculated_premium = self.bs_manager.get_price_with_spread(greeks['price'], is_buy=is_opening_a_long_position, bid_ask_spread_pct=self.bid_ask_spread_pct)
                    new_leg['entry_premium'] = calculated_premium
                except Exception as e:
                    print(f"[ERROR] in _price_legs calculation: {e}")
                    return None # Abort if any leg fails pricing

            # This rule applies regardless of whether the premium was calculated or provided.
            if check_short_rule and new_leg['direction'] == 'short':
                if new_leg['entry_premium'] < self.close_short_leg_on_profit_threshold:
                    return None

            priced_legs.append(new_leg)
        
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

    def _calculate_universal_risk_profile(self, legs: List[Dict], realized_pnl: float, is_new_trade: bool = False) -> Dict:
        """
        Calculates the risk profile.
        MODIFIED: Now accepts an 'is_new_trade' flag to factor in brokerage
        costs for the viability check of a new position.
        """
        # <<< --- THE DEFINITIVE, FINAL FIX IS HERE --- >>>
        # If this is a new trade being considered, we must include the cost to open it.
        brokerage_cost = 0.0
        if is_new_trade:
            brokerage_cost = len(legs) * self.brokerage_per_leg

        # We subtract the brokerage cost from the realized P&L for the simulation.
        # This effectively "pre-charges" the trade for its own commission.
        simulated_realized_pnl = realized_pnl - brokerage_cost

        if not legs:
            raw_profile = {'strategy_max_profit': -brokerage_cost, 'total_max_profit': simulated_realized_pnl, 'max_loss': simulated_realized_pnl, 'profit_factor': 0.0, 'breakevens': []}
            return self._sanitize_dict(raw_profile)

        # --- Call the Numba simulation with the adjusted P&L ---
        strategy_max_profit, strategy_max_loss, _, _ = _numba_run_pnl_simulation(
            self._prepare_legs_for_numba(legs),
            -brokerage_cost, # Isolate the strategy's P&L including its own commission
            self.start_price, self.lot_size, self.bs_manager.risk_free_rate, self.bid_ask_spread_pct
        )
        total_max_profit, total_max_loss, profit_factor, breakevens = _numba_run_pnl_simulation(
            self._prepare_legs_for_numba(legs),
            simulated_realized_pnl, # Use the full P&L for the total calculation
            self.start_price, self.lot_size, self.bs_manager.risk_free_rate, self.bid_ask_spread_pct
        )

        raw_profile = {
            'strategy_max_profit': min(self.undefined_risk_cap, strategy_max_profit),
            'total_max_profit': min(self.undefined_risk_cap, total_max_profit),
            'max_loss': max(-self.undefined_risk_cap, total_max_loss),
            'profit_factor': profit_factor,
            'breakevens': breakevens
        }
        
        return self._sanitize_dict(raw_profile)

    def _open_single_leg(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        if len(self.portfolio) >= self.max_positions: return False
        
        # We replace the old, faulty parser with the new, simplified version
        # that correctly handles the "..._ATM" action format.
        try:
            parts = action_name.split('_')
            direction, option_type = parts[1].lower(), parts[2].lower()
            # Since the action name is now just "_ATM", the offset is always 0.
            offset = 0
        except (ValueError, IndexError):
            print(f"Warning: Could not parse single leg action name: {action_name}")
            return False

        strike_price = self.market_rules_manager.get_atm_price(current_price) + (offset * self.strike_distance)
        
        legs = [{'type': option_type, 'direction': direction, 'strike_price': strike_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}]
        priced_legs = self._price_legs(legs, current_price, iv_bin_index, check_short_rule=True)
        
        if not priced_legs: return False

        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl, is_new_trade=True)
        if pnl_profile['strategy_max_profit'] <= self.min_profit_hurdle: return False
        
        # The strategy_id for a naked leg is its internal name, e.g., "SHORT_CALL"
        internal_strategy_name = f"{direction.upper()}_{option_type.upper()}"
        strategy_id = self.strategy_name_to_id.get(internal_strategy_name)
        assert strategy_id is not None, f"Could not find strategy_id for naked leg: {internal_strategy_name}"
        pnl_profile['strategy_id'] = strategy_id

        self._execute_trades(priced_legs, pnl_profile)
        return True

    def _open_straddle(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) <= self.max_positions - 2, "Illegal attempt to open a straddle when there are less than two positions available."
        if len(self.portfolio) > self.max_positions - 2: return False
        direction = 'long' if 'LONG' in action_name else 'short'
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        legs = [{'type': 'call', 'direction': direction, 'strike_price': atm_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': direction, 'strike_price': atm_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}]
        legs = self._price_legs(legs, current_price, iv_bin_index)
        pnl  = self._calculate_universal_risk_profile(legs, self.realized_pnl, is_new_trade=True)
        if pnl['strategy_max_profit'] <= self.min_profit_hurdle: return False
        # 3. Now get the strategy ID.
        pnl['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(legs, pnl)
        return True

    def _open_strangle(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """
        Opens a two-leg strangle. This is a smart dispatcher that can handle
        both modern delta-based strangles and legacy fixed-width strangles.
        """
        if len(self.portfolio) > self.max_positions - 2: return False
        
        parts = action_name.split('_')
        direction = parts[1].lower()
        legs = []

        # --- Dispatcher Logic ---
        if 'DELTA' in action_name:
            # --- INTELLIGENT, DELTA-BASED LOGIC ---
            try:
                target_delta_int = int(parts[4])
            except (ValueError, IndexError): return False # Invalid format

            all_deltas = [15, 20, 25, 30]
            start_index = all_deltas.index(target_delta_int) if target_delta_int in all_deltas else 0
            delta_search_list = [d / 100.0 for d in all_deltas[start_index:]]

            strike_call = self._find_strike_from_delta_list(delta_search_list, 'call', current_price, iv_bin_index, days_to_expiry)
            strike_put = self._find_strike_from_delta_list([-d for d in delta_search_list], 'put', current_price, iv_bin_index, days_to_expiry)
            
            if strike_call and strike_put and strike_put < strike_call:
                legs = [{'type': 'call', 'direction': direction, 'strike_price': strike_call},
                        {'type': 'put', 'direction': direction, 'strike_price': strike_put}]
            else:
                return False # Fallback if no valid delta strikes are found
       
        # --- Finalize the Trade (common to both logic paths) ---
        for leg in legs:
            leg['entry_step'] = current_step
            leg['days_to_expiry'] = days_to_expiry
        
        priced_legs = self._price_legs(legs, current_price, iv_bin_index)
        pnl_profile = self._calculate_universal_risk_profile(priced_legs, self.realized_pnl, is_new_trade=True)
        if pnl_profile['strategy_max_profit'] <= self.min_profit_hurdle: return False
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(priced_legs, pnl_profile)
        return True

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
        if len(self.portfolio) > self.max_positions - 4: return True
        
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
        pnl  = self._calculate_universal_risk_profile(legs, self.realized_pnl, is_new_trade=True)
        # If the theoretical max profit of the strategy is negative, it's a guaranteed loser. Do not open it.
        if pnl['strategy_max_profit'] <= self.min_profit_hurdle: return False
        pnl['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(legs, pnl)
        return True

    def _open_iron_fly(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """
        Opens an Iron Fly with a tiered, dynamic search for the optimal wing width.
        - SHORT Iron Fly (credit): Tries to set wings based on premium, then walks inward to find a valid width.
        - LONG Iron Fly (debit): Uses a fixed width.
        """
        if len(self.portfolio) > self.max_positions - 4: return False
        
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
        pnl  = self._calculate_universal_risk_profile(legs, self.realized_pnl, is_new_trade=True)
        if pnl['strategy_max_profit'] <= self.min_profit_hurdle: return False
        pnl['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
        self._execute_trades(legs, pnl)
        return True

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

    def _open_butterfly(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float) -> bool:
        """
        Opens a three-strike butterfly with a dynamic width.
        This version uses separate, correct logic for LONG (debit, seeks target cost)
        and SHORT (credit, seeks max profitable credit) butterflies.
        """
        if len(self.portfolio) > self.max_positions - 4:
            return False

        parts = action_name.split('_')
        direction, option_type = parts[1].lower(), parts[2].lower()
        atm_price = self.market_rules_manager.get_atm_price(current_price)

        # --- Define Body and Wing Directions ---
        wing_direction = 'long' if direction == 'long' else 'short'
        body_direction = 'short' if direction == 'long' else 'long'
        body_legs_def = [
            {'type': option_type, 'direction': body_direction, 'strike_price': atm_price},
            {'type': option_type, 'direction': body_direction, 'strike_price': atm_price}
        ]

        best_legs_found = None
        max_search_width = 20
        min_search_width_multiplier = 2
        
        if direction == 'long': # --- Logic for LONG (DEBIT) butterflies ---
            smallest_cost_diff = float('inf')
            target_cost = atm_price * self.butterfly_target_cost_pct
            
            for i in range(min_search_width_multiplier, max_search_width + 1):
                wing_width = i * self.strike_distance
                strike_lower, strike_upper = atm_price - wing_width, atm_price + wing_width
                if strike_lower <= 0: continue

                wing_legs_def = [{'type': option_type, 'direction': wing_direction, 'strike_price': strike_lower}, {'type': option_type, 'direction': wing_direction, 'strike_price': strike_upper}]
                candidate_legs_def = body_legs_def + wing_legs_def
                for leg in candidate_legs_def: leg['days_to_expiry'] = days_to_expiry
                
                priced_legs = self._price_legs(candidate_legs_def, current_price, iv_bin_index)
                if not priced_legs: continue

                # For a debit trade, check that it's actually a debit.
                net_premium = sum(leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1) for leg in priced_legs)
                if net_premium <= 0: continue
                
                # Check if this trade is viable after commissions.
                pnl_profile = self._calculate_universal_risk_profile(priced_legs, 0.0, is_new_trade=True)
                if pnl_profile['strategy_max_profit'] <= self.min_profit_hurdle:
                    continue

                # If viable, check if it's the best fit for our target cost.
                cost_diff = abs(net_premium - target_cost)
                if cost_diff < smallest_cost_diff:
                    smallest_cost_diff = cost_diff
                    best_legs_found = priced_legs
        
        else: # --- New, Correct Logic for SHORT (CREDIT) butterflies ---
            highest_net_credit = 0
            
            for i in range(min_search_width_multiplier, max_search_width + 1):
                wing_width = i * self.strike_distance
                strike_lower, strike_upper = atm_price - wing_width, atm_price + wing_width
                if strike_lower <= 0: continue

                wing_legs_def = [{'type': option_type, 'direction': wing_direction, 'strike_price': strike_lower}, {'type': option_type, 'direction': wing_direction, 'strike_price': strike_upper}]
                candidate_legs_def = body_legs_def + wing_legs_def
                for leg in candidate_legs_def: leg['days_to_expiry'] = days_to_expiry
                
                priced_legs = self._price_legs(candidate_legs_def, current_price, iv_bin_index)
                if not priced_legs: continue

                # For a credit trade, check that it's actually a credit.
                net_premium = sum(leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1) for leg in priced_legs)
                if net_premium >= 0: continue

                # Check if the trade is viable after commissions.
                pnl_profile = self._calculate_universal_risk_profile(priced_legs, 0.0, is_new_trade=True)
                if pnl_profile['strategy_max_profit'] <= self.min_profit_hurdle:
                    continue 

                # If viable, check if it's the most credit we can receive.
                net_credit = abs(net_premium)
                if net_credit > highest_net_credit:
                    highest_net_credit = net_credit
                    best_legs_found = priced_legs

        # --- Finalize the Trade ---
        if best_legs_found:
            for leg in best_legs_found: leg['entry_step'] = current_step
            # We must recalculate the final PnL profile with the true realized PnL
            final_pnl_profile = self._calculate_universal_risk_profile(best_legs_found, self.realized_pnl, is_new_trade=True)
            final_pnl_profile['strategy_id'] = self.strategy_name_to_id.get(action_name, -1)
            self._execute_trades(best_legs_found, final_pnl_profile)
            return True
        else:
            return False

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

        # 1. Manually deduct the brokerage cost for the 2 NEW legs being added.
        self.realized_pnl -= len(priced_wings) * self.brokerage_per_leg

        # 2. Assemble final state and use the safe state-setter.
        final_condor_legs = original_legs + priced_wings
        pnl_profile = self._calculate_universal_risk_profile(final_condor_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_SHORT_IRON_CONDOR')
        self._set_new_portfolio_state(final_condor_legs, pnl_profile)

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

        # 1. Handle financial logic for the NEW legs only.
        new_legs_brokerage = len(priced_wings) * self.brokerage_per_leg
        self.realized_pnl -= new_legs_brokerage

        # 2. Assemble the final, complete list of legs (old + new).
        final_fly_legs = original_legs + priced_wings
        
        # 3. Calculate the unified profile for the new 4-leg strategy.
        pnl_profile = self._calculate_universal_risk_profile(final_fly_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_SHORT_IRON_FLY')
        
        # 4. Atomically set the new portfolio state using the safe helper.
        self._set_new_portfolio_state(final_fly_legs, pnl_profile)
        
    def convert_to_strangle(self, current_price: float, iv_bin_index: int, current_step: int):
        """Removes the long wings from an Iron Condor to become a strangle."""
        if len(self.portfolio) != 4: return
        original_creation_id = self.portfolio.iloc[0]['creation_id']
        legs_to_keep = self.portfolio[self.portfolio['direction'] == 'short'].to_dict(orient='records')
        legs_to_close = self.portfolio[self.portfolio['direction'] == 'long']
        self._process_leg_closures(legs_to_close, current_price, iv_bin_index, current_step)
        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_SHORT_STRANGLE_DELTA_20')
        # Atomically set the new 2-leg portfolio state.
        self._set_new_portfolio_state(legs_to_keep, pnl_profile)

    def convert_to_straddle(self, current_price: float, iv_bin_index: int, current_step: int):
        """Removes the long wings from an Iron Fly to become a straddle."""
        if len(self.portfolio) != 4: return
        original_creation_id = self.portfolio.iloc[0]['creation_id']
        legs_to_keep = self.portfolio[self.portfolio['direction'] == 'short'].to_dict(orient='records')
        legs_to_close = self.portfolio[self.portfolio['direction'] == 'long']
        self._process_leg_closures(legs_to_close, current_price, iv_bin_index, current_step)

        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_SHORT_STRADDLE')
        self._set_new_portfolio_state(legs_to_keep, pnl_profile)

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
        
        self._process_leg_closures(legs_to_close_df, current_price, iv_bin_index, current_step)

        legs_to_keep_list = legs_to_keep_df.to_dict(orient='records')
        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep_list, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_BULL_CALL_SPREAD')
        self._set_new_portfolio_state(legs_to_keep_list, pnl_profile)

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
        
        self._process_leg_closures(legs_to_close_df, current_price, iv_bin_index, current_step)

        legs_to_keep_list = legs_to_keep_df.to_dict(orient='records')
        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep_list, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_BEAR_CALL_SPREAD')
        self._set_new_portfolio_state(legs_to_keep_list, pnl_profile)

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
        
        self._process_leg_closures(legs_to_close_df, current_price, iv_bin_index, current_step)

        legs_to_keep_list = legs_to_keep_df.to_dict(orient='records')
        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep_list, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_BULL_PUT_SPREAD')
        self._set_new_portfolio_state(legs_to_keep_list, pnl_profile)

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
        
        self._process_leg_closures(legs_to_close_df, current_price, iv_bin_index, current_step)

        legs_to_keep_list = legs_to_keep_df.to_dict(orient='records')
        pnl_profile = self._calculate_universal_risk_profile(legs_to_keep_list, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_BEAR_PUT_SPREAD')
        self._set_new_portfolio_state(legs_to_keep_list, pnl_profile)

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

        # The portfolio is now empty. We can safely call _execute_trades, which will
        # correctly handle brokerage for the 2 new legs and set the state.
        pnl_profile = self._calculate_universal_risk_profile(priced_new_legs, self.realized_pnl, is_new_trade=True)
        pnl_profile['strategy_id'] = original_strategy_id
        self._execute_trades(priced_new_legs, pnl_profile) # This is now correct

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

    def manage_portfolio_based_on_intent(self, agent_intent: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float, volatility_bias: str) -> str:
        """
        The Tactical Engine. Receives agent intent and executes the best tactical move.
        Returns a string detailing the action taken for logging.
        """
        # <<< --- THE DEFINITIVE, CLEANER LOGIC --- >>>
        
        # --- Handle HOLD and CLOSE_ALL intents first, as they are simplest ---
        if agent_intent == 'HOLD':
            return "HOLD"
        
        if agent_intent == 'DECIDE_CLOSE_ALL':
            if not self.portfolio.empty:
                self.close_all_positions(current_price, iv_bin_index, current_step)
                return "CLOSE_ALL"
            else:
                return "HOLD (Portfolio Empty)"

        # --- If the intent is to open a position ---
        if self.portfolio.empty:
            open_intent = agent_intent.replace('DECIDE_', 'OPEN_') + '_POSITION'
            was_opened, resolved_action = self.resolve_and_open_strategy(
                open_intent, current_price, iv_bin_index, current_step, days_to_expiry, volatility_bias
            )
            return resolved_action if was_opened else "HOLD (Resolver Failed)"

        # --- If portfolio is active and intent is directional ---
        current_stance = self.current_portfolio_stance
        agent_view = agent_intent.replace('DECIDE_', '')

        if agent_view == current_stance:
            return self._manage_active_position(current_price, iv_bin_index, current_step)
        else:
            return self._handle_view_reversal(
                agent_intent, 
                current_price, 
                iv_bin_index, 
                current_step, 
                days_to_expiry, 
                volatility_bias
            )

    def _manage_active_position(self, current_price: float, iv_bin_index: int, current_step: int) -> str:
        """Rule-based logic for managing a position when the agent's view is consistent."""
        greeks = self.get_portfolio_greeks(current_price, iv_bin_index)
        delta_threshold = self.cfg.get('delta_neutral_threshold', 0.1)

        # If position is delta-imbalanced, the highest priority is to hedge.
        if abs(greeks['delta_norm']) > delta_threshold:
            # Find the best leg to roll to neutralize delta.
            best_leg_to_roll = self._find_best_leg_to_hedge_delta()
            if best_leg_to_roll is not None:
                self.hedge_portfolio_by_rolling_leg(best_leg_to_roll, current_price, iv_bin_index, current_step)
                return f"RISK_MGMT: HEDGE_ROLL_LEG_{best_leg_to_roll}"
        
        # If delta is fine, the default action is to hold and collect theta.
        return "HOLD (Maintain Position)"

    def _handle_view_reversal(self, new_intent: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float, volatility_bias: str) -> str:
        """
        Sophisticated, intent-aware rules for when the agent's view contradicts the current position.
        This version includes a configurable loss-acceptance threshold.
        """

        # --- PRIORITY 1: Check if the position is a safe winner ---
        if self._is_position_safely_itm(current_price):
            # If the position is deep in-the-money, override any reversal/morph
            # decision and simply hold. This prevents over-management of winners.
            return "HOLD (Let Winner Run)"

        pnl_verification = self.get_pnl_verification(current_price, iv_bin_index)
        unrealized_pnl = pnl_verification['unrealized_pnl']
        is_challenged = self._is_position_challenged(current_price)

        # 1. Calculate the loss acceptance threshold in dollars.
        #    This is based on the initial premium paid or credit received for the trade.
        initial_risk = abs(self.initial_net_premium)
        loss_acceptance_threshold = -initial_risk * self.cfg.get('morph_loss_acceptance_multiple', 0.5)

        # 2. The new condition: The engine will attempt to "morph" the position if it is either
        #    profitable (pnl > 0) OR if its loss is within our acceptable tolerance.
        if unrealized_pnl >= loss_acceptance_threshold and not is_challenged:
            
            # --- The "morph" logic (which we fixed previously) remains unchanged ---
            resolved_action_name = "HOLD (Morph Failed)"
            if new_intent == 'DECIDE_BULLISH':
                leg_to_shift_idx = self._find_best_leg_for_delta_shift(increase=True)
                if leg_to_shift_idx is not None:
                    self._execute_delta_shift(leg_to_shift_idx, increase=True, current_price=current_price, iv_bin_index=iv_bin_index, current_step=current_step)
                    resolved_action_name = f"MORPH: INCREASE_DELTA_LEG_{leg_to_shift_idx}"
            elif new_intent == 'DECIDE_BEARISH':
                leg_to_shift_idx = self._find_best_leg_for_delta_shift(increase=False)
                if leg_to_shift_idx is not None:
                    self._execute_delta_shift(leg_to_shift_idx, increase=False, current_price=current_price, iv_bin_index=iv_bin_index, current_step=current_step)
                    resolved_action_name = f"MORPH: DECREASE_DELTA_LEG_{leg_to_shift_idx}"
            elif new_intent == 'DECIDE_NEUTRAL':
                # If our goal is to become neutral, the correct morph from a directional
                # spread is to convert it into an Iron Condor.
                resolved_action_name = self._morph_spread_to_condor(current_price, iv_bin_index, current_step, days_to_expiry)

            return resolved_action_name

        # --- Default case: The position is losing BEYOND our tolerance OR it is challenged. ---
        # The best move is to cut losses and reverse. This logic is unchanged.
        self.close_all_positions(current_price, iv_bin_index, current_step)
        open_intent = new_intent.replace('DECIDE_', 'OPEN_') + '_POSITION'
        was_opened, resolved_action = self.resolve_and_open_strategy(
            open_intent, current_price, iv_bin_index, current_step, days_to_expiry, volatility_bias
        )
        if was_opened:
            return f"REVERSE: CLOSE_ALL & OPEN {resolved_action}"
        else:
            return "REVERSE: CLOSE_ALL (Open Failed)"

    def _is_position_safely_itm(self, current_price: float) -> bool:
        """
        A sophisticated check to see if a directional position is safely in-the-money,
        justifying a 'let the winner run' approach.
        """
        if self.portfolio.empty:
            return False

        stance = self.current_portfolio_stance
        if stance not in ["BULLISH", "BEARISH"]:
            return False # This rule only applies to directional trades

        # Get the breakeven points from the risk profile
        pnl_profile = self._calculate_universal_risk_profile(self.portfolio.to_dict('records'), self.realized_pnl)
        breakevens = pnl_profile.get('breakevens', [])
        if not breakevens:
            return False

        # --- Your "Let Winners Run" Logic ---
        if stance == "BULLISH":
            # For a bullish position, it's safely ITM if the current price
            # is above the highest breakeven point.
            highest_breakeven = max(breakevens)
            return current_price > highest_breakeven
            
        elif stance == "BEARISH":
            # For a bearish position, it's safely ITM if the current price
            # is below the lowest breakeven point.
            lowest_breakeven = min(breakevens)
            return current_price < lowest_breakeven
            
        return False

    def _morph_spread_to_condor(self, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float) -> str:
        """
        A smart helper to morph an existing 2-leg vertical spread into a 4-leg Iron Condor.
        Returns a descriptive string of the action taken.
        """
        # --- Guard Clauses ---
        if len(self.portfolio) != 2 or len(self.portfolio['type'].unique()) != 1:
            return "HOLD (Not a Spread)"
        if len(self.portfolio) > self.max_positions - 2:
            return "HOLD (Not Enough Slots)"

        original_legs = self.portfolio.to_dict('records')
        option_type = original_legs[0]['type']
        spread_width = abs(original_legs[0]['strike_price'] - original_legs[1]['strike_price'])

        # --- Define the New, Opposing Spread ---
        new_legs_def = []
        if option_type == 'put': # Original is a Put Spread, add a Call Spread
            highest_put_strike = max(leg['strike_price'] for leg in original_legs)
            new_short_strike = self._find_strike_for_delta(0.30, 'call', current_price, iv_bin_index, days_to_expiry)
            if new_short_strike is None or new_short_strike <= highest_put_strike: return "HOLD (Strikes Overlap)"
            new_long_strike = new_short_strike + spread_width
            new_legs_def = [
                {'type': 'call', 'direction': 'short', 'strike_price': new_short_strike},
                {'type': 'call', 'direction': 'long', 'strike_price': new_long_strike}
            ]
        else: # Original is a Call Spread, add a Put Spread
            lowest_call_strike = min(leg['strike_price'] for leg in original_legs)
            new_short_strike = self._find_strike_for_delta(-0.30, 'put', current_price, iv_bin_index, days_to_expiry)
            if new_short_strike is None or new_short_strike >= lowest_call_strike: return "HOLD (Strikes Overlap)"
            new_long_strike = new_short_strike - spread_width
            if new_long_strike <=0: return "HOLD (Invalid Strike)"
            new_legs_def = [
                {'type': 'put', 'direction': 'short', 'strike_price': new_short_strike},
                {'type': 'put', 'direction': 'long', 'strike_price': new_long_strike}
            ]
        
        # --- Execute the Transformation using the robust "rebuild" pattern ---
        for leg in new_legs_def:
            leg.update({'entry_step': current_step, 'days_to_expiry': days_to_expiry})
        
        priced_new_legs = self._price_legs(new_legs_def, current_price, iv_bin_index)
        if not priced_new_legs: return "HOLD (Pricing Failed)"

        self.realized_pnl -= len(priced_new_legs) * self.brokerage_per_leg
        final_condor_legs = original_legs + priced_new_legs
        pnl_profile = self._calculate_universal_risk_profile(final_condor_legs, self.realized_pnl)
        pnl_profile['strategy_id'] = self.strategy_name_to_id.get('OPEN_SHORT_IRON_CONDOR', -1)
        self._set_new_portfolio_state(final_condor_legs, pnl_profile)
        
        return "MORPH: ADD_SPREAD_TO_IRON_CONDOR"

    def _find_best_leg_for_delta_shift(self, increase: bool) -> int or None:
        """
        A simple helper to find the first valid candidate leg for a directional delta adjustment.
        Returns the index of the leg to be shifted.
        """
        if self.portfolio.empty: 
            return None
            
        # A more advanced version could score all legs based on their gamma, distance
        # from ATM, or other factors. For now, finding the first valid leg (index 0)
        # is a robust and sufficient strategy for the Tactical Engine.
        if len(self.portfolio) > 0:
            return 0
            
        return None

    def _execute_delta_shift(self, leg_index: int, increase: bool, current_price: float, iv_bin_index: int, current_step: int):
        """
        A private helper to resolve and execute a directional delta shift on a specific leg.
        """
        # This guard clause is important for safety.
        if not (0 <= leg_index < len(self.portfolio)):
            return

        leg = self.portfolio.iloc[leg_index]
        leg_type, leg_dir = leg['type'], leg['direction']
        shift_dir = ""

        # Resolve the "increase/decrease" intent into a concrete "UP/DOWN" shift direction.
        if increase:
            if (leg_type == 'call' and leg_dir == 'long') or (leg_type == 'put' and leg_dir == 'short'): shift_dir = 'DOWN'
            elif (leg_type == 'put' and leg_dir == 'long') or (leg_type == 'call' and leg_dir == 'short'): shift_dir = 'UP'
        else: # decrease
            if (leg_type == 'call' and leg_dir == 'long') or (leg_type == 'put' and leg_dir == 'short'): shift_dir = 'UP'
            elif (leg_type == 'put' and leg_dir == 'long') or (leg_type == 'call' and leg_dir == 'short'): shift_dir = 'DOWN'
        
        if shift_dir:
            # Perform the direct, direction-aware legality check before executing.
            delta = self.strike_distance if shift_dir == "UP" else -self.strike_distance
            new_strike = leg['strike_price'] + delta
            other_legs = self.portfolio.drop(leg_index)
            
            is_conflict = any(
                (p['strike_price'] == new_strike and p['type'] == leg_type and p['direction'] != leg_dir) 
                for _, p in other_legs.iterrows()
            )
            
            if not is_conflict:
                resolved_action_name = f"SHIFT_{shift_dir}_POS_{leg_index}"
                self.shift_position(resolved_action_name, current_price, iv_bin_index, current_step)

    def _is_position_challenged(self, current_price: float) -> bool:
        """ A helper to determine if any short leg is being threatened by the current price. """
        if self.portfolio.empty: return False
        for _, leg in self.portfolio.iterrows():
            if leg['direction'] == 'short':
                if leg['type'] == 'call' and current_price > leg['strike_price']: return True
                if leg['type'] == 'put' and current_price < leg['strike_price']: return True
        return False

    def _find_best_leg_to_hedge_delta(self) -> int or None:
        """A simple helper to find the most effective leg to roll for a delta hedge."""
        if self.portfolio.empty: return None
        # A more advanced version could score each leg. For now, we find the first eligible leg.
        for i in range(len(self.portfolio)):
            # Add logic here to find the best candidate, e.g., the one most OTM
            return i
        return None
