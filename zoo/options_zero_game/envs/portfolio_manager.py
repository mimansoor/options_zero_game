# zoo/options_zero_game/envs/portfolio_manager.py
import pandas as pd
import numpy as np
import math
from typing import Dict, List, Any, Tuple
from .black_scholes_manager import BlackScholesManager, _numba_cdf
from .market_rules_manager import MarketRulesManager

class PortfolioManager:
    """
    Manages the state of the agent's portfolio, including all trades,
    PnL calculations, and constructing the portfolio observation state.
    This class is stateful and is the heart of the trading logic.
    """
    def __init__(self, cfg: Dict, bs_manager: BlackScholesManager, market_rules_manager: MarketRulesManager):
        # Config
        self.initial_cash = cfg['initial_cash']
        self.lot_size = cfg['lot_size']
        self.bid_ask_spread_pct = cfg['bid_ask_spread_pct']
        self.max_positions = cfg['max_positions']
        self.start_price = cfg['start_price']
        self.strike_distance = cfg['strike_distance']
        self.undefined_risk_cap = self.initial_cash
        self.strategy_name_to_id = cfg.get('strategy_name_to_id', {})
        self.close_short_leg_on_profit_threshold = cfg.get('close_short_leg_on_profit_threshold', 0.0)
        self.is_eval_mode = cfg.get('is_eval_mode', False)
        self.brokerage_per_leg = cfg.get('brokerage_per_leg', 0.0)
        self.max_strike_offset = cfg['max_strike_offset']
        self.receipts_for_current_step: List[dict] = []
        self.steps_per_day = cfg.get('steps_per_day', 1)
        
        # Managers
        self.bs_manager = bs_manager
        self.market_rules_manager = market_rules_manager

        # State variables
        self.portfolio = pd.DataFrame()
        self.realized_pnl: float = 0.0
        self.high_water_mark: float = 0.0
        self.next_creation_id: int = 0
        self.initial_net_premium: float = 0.0
        self.portfolio_columns = ['type', 'direction', 'entry_step', 'strike_price', 'entry_premium', 'days_to_expiry', 'creation_id', 'strategy_id', 'strategy_max_profit', 'strategy_max_loss', 'is_hedged']
        self.portfolio_dtypes = {'type': 'object', 'direction': 'object', 'entry_step': 'int64', 'strike_price': 'float64', 'entry_premium': 'float64', 'days_to_expiry': 'float64', 'creation_id': 'int64', 'strategy_id': 'int64', 'strategy_max_profit': 'float64', 'strategy_max_loss': 'float64', 'is_hedged': 'bool'}
        self.highest_realized_profit = 0.0
        self.lowest_realized_loss = 0.0
        self.mtm_pnl_high = 0.0  # Tracks the highest MtM P&L seen
        self.mtm_pnl_low = 0.0   # Tracks the lowest MtM P&L seen (max drawdown)
    
    def reset(self):
        """Resets the portfolio to an empty state for a new episode."""
        self.portfolio = pd.DataFrame(columns=self.portfolio_columns).astype(self.portfolio_dtypes)
        self.realized_pnl = 0.0
        self.high_water_mark = self.initial_cash
        self.next_creation_id = 0
        self.initial_net_premium = 0.0
        self.receipts_for_current_step = []
        # Reset the MtM trackers for the new episode
        self.mtm_pnl_high = 0.0
        self.mtm_pnl_low = 0.0
        self.highest_realized_profit = 0.0
        self.lowest_realized_loss = 0.0

    # --- Public Methods (called by the main environment) ---

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
        
        # <<< YOUR SUGGESTION APPLIED: Redundant 'if' is removed >>>
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        for _, pos in self.portfolio.iterrows():
            direction_multiplier = 1 if pos['direction'] == 'long' else -1
            is_call = pos['type'] == 'call'
            offset = round((pos['strike_price'] - atm_price) / self.strike_distance)
            vol = self.market_rules_manager.get_implied_volatility(offset, pos['type'], iv_bin_index)
            leg_greeks = self.bs_manager.get_all_greeks_and_price(current_price, pos['strike_price'], pos['days_to_expiry'], vol, is_call)

            total_delta += leg_greeks['delta'] * self.lot_size * direction_multiplier
            total_gamma += leg_greeks['gamma'] * (self.lot_size**2 * current_price**2 / 100) * direction_multiplier
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

        # Route by most complex/specific keywords first to avoid misrouting
        if 'DELTA' in action_name and 'STRANGLE' in action_name:
            self._open_delta_strangle(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'FLY' in action_name and 'IRON' not in action_name:
            self._open_butterfly(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'IRON_CONDOR' in action_name:
            self._open_iron_condor(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'IRON_FLY' in action_name:
            self._open_iron_fly(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
        elif 'VERTICAL' in action_name:
            self._open_vertical_spread(action_name, current_price, iv_bin_index, current_step, days_to_expiry)
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
            vol = self.market_rules_manager.get_implied_volatility(offset, pos['type'], iv_bin_index)
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
            
            vec[current_pos_idx + pos_idx_map['DELTA']] = greeks['delta']
            vec[current_pos_idx + pos_idx_map['GAMMA']] = math.tanh(greeks['gamma'] * self.lot_size)
            vec[current_pos_idx + pos_idx_map['THETA']] = math.tanh(greeks['theta'] * self.lot_size)
            vec[current_pos_idx + pos_idx_map['VEGA']] = math.tanh(greeks['vega'] * self.lot_size)
            # Convert the boolean to a float for the neural network.
            vec[current_pos_idx + pos_idx_map['IS_HEDGED']] = 1.0 if pos['is_hedged'] else 0.0
           
    def get_total_pnl(self, current_price: float, iv_bin_index: int) -> float:
        if self.portfolio.empty:
            return self.realized_pnl
        
        def pnl_calculator(row):
            is_call = row['type'] == 'call'
            atm_price = self.market_rules_manager.get_atm_price(current_price)
            offset = round((row['strike_price'] - atm_price) / self.strike_distance)
            vol = self.market_rules_manager.get_implied_volatility(offset, row['type'], iv_bin_index)
            greeks = self.bs_manager.get_all_greeks_and_price(current_price, row['strike_price'], row['days_to_expiry'], vol, is_call)
            
            current_premium = self.bs_manager.get_price_with_spread(greeks['price'], row['direction'] == 'short', self.bid_ask_spread_pct)
            price_diff = current_premium - row['entry_premium']
            return price_diff * self.lot_size * (1 if row['direction'] == 'long' else -1)

        unrealized_pnl = self.portfolio.apply(pnl_calculator, axis=1).sum()
        return self.realized_pnl + unrealized_pnl

    def get_current_equity(self, current_price: float, iv_bin_index: int) -> float:
        return self.initial_cash + self.get_total_pnl(current_price, iv_bin_index)

    def update_positions_after_time_step(self, days_of_decay: int, current_price: float, iv_bin_index: int):
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
                self.close_position(idx, current_price, iv_bin_index)

        # Finally, check for any profitable short legs that can be closed for cheap.
        self._close_worthless_short_legs(current_price, iv_bin_index)

    def close_position(self, position_index: int, current_price: float, iv_bin_index: int):
        """
        Atomically closes a single leg and correctly re-evaluates the risk profile
        of the entire remaining strategy.
        """
        if position_index < 0:
            position_index = len(self.portfolio) + position_index

        assert 0 <= position_index < len(self.portfolio), f"Attempted to close invalid index: {position_index}."
        
        pos_to_close = self.portfolio.iloc[position_index].copy()
        original_creation_id = pos_to_close['creation_id']
        
        # 1. Calculate the realized PnL for the leg being closed.
        is_call = pos_to_close['type'] == 'call'
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        offset = round((pos_to_close['strike_price'] - atm_price) / self.strike_distance)
        vol = self.market_rules_manager.get_implied_volatility(offset, pos_to_close['type'], iv_bin_index)
        greeks = self.bs_manager.get_all_greeks_and_price(current_price, pos_to_close['strike_price'], pos_to_close['days_to_expiry'], vol, is_call)
        
        is_short = pos_to_close['direction'] == 'short'
        exit_premium = self.bs_manager.get_price_with_spread(greeks['price'], is_buy=is_short, bid_ask_spread_pct=self.bid_ask_spread_pct)
        
        pnl_multiplier = 1 if pos_to_close['direction'] == 'long' else -1
        pnl = (exit_premium - pos_to_close['entry_premium']) * pnl_multiplier

        pnl_for_leg = pnl * self.lot_size
        self.realized_pnl += pnl_for_leg

        # Deduct the brokerage fee for closing this leg.
        self.realized_pnl -= self.brokerage_per_leg

        if pnl_for_leg > self.highest_realized_profit:
            self.highest_realized_profit = pnl_for_leg
        if pnl_for_leg < self.lowest_realized_loss:
            self.lowest_realized_loss = pnl_for_leg

        receipt = {
            'position': f"{pos_to_close['direction'].upper()} {pos_to_close['type'].upper()}",
            'strike': pos_to_close['strike_price'],
            'entry_day': (pos_to_close['entry_step'] // self.steps_per_day) + 1,
            'entry_prem': pos_to_close['entry_premium'],
            'exit_prem': exit_premium,
            'realized_pnl': pnl_for_leg
        }
        self.receipts_for_current_step.append(receipt)

        # 2. Re-assemble the remaining legs of the strategy.
        remaining_legs = [
            row.to_dict() for idx, row in self.portfolio.iterrows()
            if row['creation_id'] == original_creation_id and idx != position_index
        ]
        
        # 3. Atomically update the portfolio: remove ALL old legs of the strategy.
        self.portfolio = self.portfolio[self.portfolio['creation_id'] != original_creation_id].reset_index(drop=True)

        # 4. If there are any legs left, re-profile and re-add them as a new strategy.
        if remaining_legs:
            # Determine the new strategy name and profile.
            num_legs = len(remaining_legs)
            new_strategy_name = f"CUSTOM_{num_legs}_LEGS"
            pnl_profile  = self._calculate_universal_risk_profile(remaining_legs)
            pnl_profile['strategy_id'] = self.strategy_name_to_id.get(new_strategy_name, -3)

            # Re-add the remaining legs with their new, correct, unified risk profile.
            for leg in remaining_legs:
                leg['creation_id'] = original_creation_id
            self._execute_trades(remaining_legs, pnl_profile)
        else:
            # If the portfolio is now empty, we MUST take a snapshot of it.
            # We call _execute_trades with an empty list to trigger the snapshot.
            self._execute_trades([], {})

        # 1. Always update the hedge status after a change.
        self._update_hedge_status()

    def close_all_positions(self, current_price: float, iv_bin_index: int):
        while not self.portfolio.empty:
            self.close_position(-1, current_price, iv_bin_index)

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
        Calculates a rich data set for the P&L payoff diagram, including:
        1. P&L at Expiry (the sharp, "V" shaped line).
        2. Mark-to-Market P&L for the current day (the curved, "T+0" line).
        3. Standard deviation price levels based on current volatility.
        """
        if self.portfolio.empty:
            return {'expiry_pnl': [], 'current_pnl': [], 'spot_price': current_price, 'sigma_levels': {}}

        price_range = np.linspace(current_price * 0.85, current_price * 1.15, 100)
        expiry_pnl_data = []
        current_pnl_data = []
        
        # We need the current DTE for the T+0 calculation. All legs share the same strategy.
        current_dte = self.portfolio.iloc[0]['days_to_expiry']

        for price in price_range:
            pnl_at_expiry = 0
            pnl_at_today = 0
            
            for _, pos in self.portfolio.iterrows():
                pnl_multiplier = 1 if pos['direction'] == 'long' else -1
                
                # --- P&L at Expiry Calculation ---
                if pos['type'] == 'call':
                    expiry_leg_pnl = max(0, price - pos['strike_price']) - pos['entry_premium']
                else: # Put
                    expiry_leg_pnl = max(0, pos['strike_price'] - price) - pos['entry_premium']
                pnl_at_expiry += expiry_leg_pnl * pnl_multiplier * self.lot_size
                
                # --- P&L at T+0 (Today) Calculation ---
                is_call = pos['type'] == 'call'
                atm_price = self.market_rules_manager.get_atm_price(price)
                offset = round((pos['strike_price'] - atm_price) / self.strike_distance)
                vol = self.market_rules_manager.get_implied_volatility(offset, pos['type'], iv_bin_index)
                
                # We use the current DTE for the T+0 line
                greeks = self.bs_manager.get_all_greeks_and_price(price, pos['strike_price'], current_dte, vol, is_call)
                today_leg_pnl = (greeks['price'] - pos['entry_premium']) * pnl_multiplier * self.lot_size
                pnl_at_today += today_leg_pnl
            
            expiry_pnl_data.append({'price': price, 'pnl': self.realized_pnl + pnl_at_expiry})
            current_pnl_data.append({'price': price, 'pnl': self.realized_pnl + pnl_at_today})
        
        # --- Calculate Standard Deviation Price Levels ---
        atm_iv = self.market_rules_manager.get_implied_volatility(0, 'call', iv_bin_index)
        expected_move_points = 0
        if current_dte > 0:
            expected_move_points = current_price * atm_iv * math.sqrt(current_dte / 365.25)
        
        sigma_levels = {
            'plus_one': current_price + expected_move_points,
            'minus_one': current_price - expected_move_points,
            'plus_two': current_price + (2 * expected_move_points),
            'minus_two': current_price - (2 * expected_move_points),
        }

        return {
            'expiry_pnl': expiry_pnl_data,
            'current_pnl': current_pnl_data,
            'spot_price': current_price,
            'sigma_levels': sigma_levels
        }

    # --- Private Methods ---
    def _close_worthless_short_legs(self, current_price: float, iv_bin_index: int):
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
            vol = self.market_rules_manager.get_implied_volatility(offset, pos['type'], iv_bin_index)
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
                self.close_position(index, current_price, iv_bin_index)

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

    def add_hedge(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int):
        """
        Adds a protective leg to a naked position using advanced, context-aware logic.
        - If the naked leg is In-The-Money (ITM), it places the hedge one strike
          away from the current ATM strike for a defensive adjustment.
        - If the naked leg is Out-of-the-Money (OTM), it places the hedge one strike
          away from the naked leg's strike to create a standard spread.
        This version is strategy-aware and correctly uses the MarketRulesManager.
        """
        try:
            position_index = int(action_name.split('_')[-1])
            if not (0 <= position_index < len(self.portfolio)): return

            naked_leg_to_hedge = self.portfolio.iloc[position_index].copy()
            
            if naked_leg_to_hedge['is_hedged'] or len(self.portfolio) >= self.max_positions:
                return
            
            original_creation_id = naked_leg_to_hedge['creation_id']

            # --- 1. Determine Hedge Direction and Type ---
            hedge_type = naked_leg_to_hedge['type']
            is_naked_leg_short = naked_leg_to_hedge['direction'] == 'short'
            hedge_direction = 'long' if is_naked_leg_short else 'short'
            
            # --- 2. DYNAMIC HEDGE PLACEMENT LOGIC ---
            naked_strike = naked_leg_to_hedge['strike_price']
            
            # Get the ATM strike from the single source of truth: the MarketRulesManager
            atm_strike = self.market_rules_manager.get_atm_price(current_price)
            
            # Determine if the naked leg is ITM
            is_itm = False
            if hedge_type == 'call' and current_price > naked_strike: is_itm = True
            if hedge_type == 'put' and current_price < naked_strike: is_itm = True

            if is_itm:
                # --- ITM Rule (Your brilliant correction) ---
                # The position is already a loser. Place the hedge defensively
                # one strike away from the CURRENT market price (ATM).
                if hedge_type == 'call':
                    hedge_strike = atm_strike + self.strike_distance
                else: # Put
                    hedge_strike = atm_strike - self.strike_distance
            else:
                # --- OTM Rule (Your New Dynamic Hedging Logic) ---
                # The position is not yet a loser. Create a risk-defined spread
                # by placing the hedge at the breakeven point.

                # <<< 1. Get the entry premium of the naked leg >>>
                entry_premium = naked_leg_to_hedge['entry_premium']
                
                # <<< 2. Calculate the theoretical breakeven price >>>
                if hedge_type == 'call':
                    # For a short call, BE = Strike + Credit Received
                    breakeven_price = naked_strike + entry_premium
                else: # Put
                    # For a short put, BE = Strike - Credit Received
                    breakeven_price = naked_strike - entry_premium
                    
                # <<< 3. Find the closest valid market strike to the breakeven price >>>
                # We can reuse the get_atm_price helper for this rounding logic.
                closest_valid_strike = self.market_rules_manager.get_atm_price(breakeven_price)
                
                # <<< 4. Set the hedge strike, with a safety fallback >>>
                # Edge Case: If breakeven is very close to the naked strike,
                # ensure the hedge is at least one strike away.
                if closest_valid_strike == naked_strike:
                    hedge_strike = (naked_strike + self.strike_distance if hedge_type == 'call' 
                                    else naked_strike - self.strike_distance)
                else:
                    hedge_strike = closest_valid_strike

            # --- 3. Create, Price, and Assemble the Full Strategy ---
            hedge_leg_definition = {
                'type': hedge_type, 'direction': hedge_direction, 'strike_price': hedge_strike,
                'entry_step': current_step, 'days_to_expiry': naked_leg_to_hedge['days_to_expiry']
            }
            
            # Price ONLY the new hedge leg to avoid double-charging spreads
            priced_hedge_leg = self._price_legs([hedge_leg_definition], current_price, iv_bin_index)[0]

            original_strategy_legs = [row.to_dict() for _, row in self.portfolio.iterrows() if row['creation_id'] == original_creation_id]
            transformed_strategy_legs = original_strategy_legs + [priced_hedge_leg]
            
            # --- 4. Determine New Strategy Name and PnL ---
            num_legs = len(transformed_strategy_legs)
            new_strategy_name = "CUSTOM_HEDGED"
            if num_legs == 2:
                net_premium = sum(leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1) for leg in transformed_strategy_legs)
                spread_direction = "LONG" if net_premium > 0 else "SHORT"
                new_strategy_name = f"{spread_direction}_VERTICAL_{hedge_type.upper()}_1"
            elif num_legs == 4:
                new_strategy_name = "SHORT_IRON_FLY"
            
            pnl  = self._calculate_universal_risk_profile(transformed_strategy_legs)
            pnl['strategy_id'] = self.strategy_name_to_id.get(new_strategy_name, -1)
            
            # --- 5. Update Portfolio Atomically ---
            self.portfolio = self.portfolio[self.portfolio['creation_id'] != original_creation_id].reset_index(drop=True)
            for leg in transformed_strategy_legs:
                leg['creation_id'] = original_creation_id
            self._execute_trades(transformed_strategy_legs, pnl)
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse HEDGE action '{action_name}'. Error: {e}")
            return

    def _get_unrealized_pnl_for_leg(self, position_row: pd.Series, current_price: float, iv_bin_index: int) -> float:
        """Calculates the current unrealized PnL for a single leg of the portfolio."""
        is_call = position_row['type'] == 'call'
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        offset = round((position_row['strike_price'] - atm_price) / self.strike_distance)
        vol = self.market_rules_manager.get_implied_volatility(offset, position_row['type'], iv_bin_index)
        greeks = self.bs_manager.get_all_greeks_and_price(current_price, position_row['strike_price'], position_row['days_to_expiry'], vol, is_call)
        
        current_premium = self.bs_manager.get_price_with_spread(greeks['price'], is_buy=(position_row['direction'] == 'short'), bid_ask_spread_pct=self.bid_ask_spread_pct)
        
        pnl_multiplier = 1 if position_row['direction'] == 'long' else -1
        pnl = (current_premium - position_row['entry_premium']) * pnl_multiplier * self.lot_size
        return pnl

    def shift_position(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int):
        """
        Shifts a position by one strike. This is now a two-step atomic operation
        that correctly calls close_position to ensure proper logging.
        """
        try:
            parts = action_name.split('_')
            direction, position_index = parts[1].upper(), int(parts[3])
            if not (0 <= position_index < len(self.portfolio)): return

            original_pos = self.portfolio.iloc[position_index].copy()
            
            # --- THE FIX ---
            # 1. First, officially close the original position. This will generate the receipt.
            self.close_position(position_index, current_price, iv_bin_index)
            
            # 2. Then, open the new, shifted leg.
            strike_modifier = self.strike_distance if direction == 'UP' else -self.strike_distance
            new_strike_price = original_pos['strike_price'] + strike_modifier
            
            new_leg = {
                'type': original_pos['type'], 'direction': original_pos['direction'],
                'strike_price': new_strike_price, 'entry_step': current_step,
                'days_to_expiry': original_pos['days_to_expiry'],
            }
            
            legs = self._price_legs([new_leg], current_price, iv_bin_index)
            pnl  = self._calculate_universal_risk_profile(legs)
            strategy_name = f"{legs[0]['direction'].upper()}_{legs[0]['type'].upper()}"
            pnl['strategy_id'] = self.strategy_name_to_id.get(strategy_name, -1)
            self._execute_trades(legs, pnl)

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse shift action '{action_name}'. Error: {e}")

    def shift_to_atm(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int):
        """
        Shifts a position to the ATM strike, correctly calling close_position.
        """
        try:
            position_index = int(action_name.split('_')[-1])
            if not (0 <= position_index < len(self.portfolio)): return

            original_pos = self.portfolio.iloc[position_index].copy()

            # --- THE FIX ---
            # 1. First, officially close the original position.
            self.close_position(position_index, current_price, iv_bin_index)
            
            # 2. Then, open the new ATM leg.
            new_atm_strike = self.market_rules_manager.get_atm_price(current_price)
            new_leg = {
                'type': original_pos['type'], 'direction': original_pos['direction'],
                'strike_price': new_atm_strike, 'entry_step': current_step,
                'days_to_expiry': original_pos['days_to_expiry'],
            }

            legs = self._price_legs([new_leg], current_price, iv_bin_index)
            pnl  = self._calculate_universal_risk_profile(legs)
            strategy_name = f"{legs[0]['direction'].upper()}_{legs[0]['type'].upper()}"
            pnl['strategy_id'] = self.strategy_name_to_id.get(strategy_name, -1)
            self._execute_trades(legs, pnl)
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse shift_to_atm action '{action_name}'. Error: {e}")

    def get_portfolio_summary(self, current_price: float, iv_bin_index: int) -> dict:
        """
        The definitive and universal summary calculator. It uses the universal risk engine
        to get breakeven points and then calculates POP for ANY strategy shape.
        """
        # --- 1. Handle Empty Portfolio ---
        if self.portfolio.empty:
            # When portfolio is empty, the only P&L is what's been realized.
            is_profitable = self.realized_pnl > 0
            return {
                'max_profit': self.realized_pnl, 'max_loss': self.realized_pnl,
                'rr_ratio': 0.0, 'prob_profit': 1.0 if is_profitable else 0.0
            }

        # <<< THE FIX: The local helper now requires realized_pnl >>>
        def _get_total_pnl_at_price(price, trade_legs, realized_pnl):
            unrealized_pnl = 0.0
            for _, leg in trade_legs.iterrows():
                pnl_mult = 1 if leg['direction'] == 'long' else -1
                leg_pnl = max(0, price - leg['strike_price']) if leg['type'] == 'call' else max(0, leg['strike_price'] - price)
                unrealized_pnl += (leg_pnl - leg['entry_premium']) * pnl_mult
            return (unrealized_pnl * self.lot_size) + realized_pnl

        # --- 2. Get Risk Profile from Universal Engine ---
        portfolio_legs_as_dict = self.portfolio.to_dict(orient='records')
        risk_profile = self._calculate_universal_risk_profile(portfolio_legs_as_dict, self.realized_pnl)
        
        max_profit = risk_profile['max_profit']
        max_loss = risk_profile['max_loss']
        breakevens = risk_profile['breakevens']
        rr_ratio = abs(max_profit / max_loss) if max_loss != 0 and max_profit is not None and max_loss is not None else float('inf')
        
        days_to_expiry = self.portfolio.iloc[0]['days_to_expiry']
        vol = self.market_rules_manager.get_implied_volatility(0, 'call', iv_bin_index)
        
        # --- 3. Calculate POP based on the found breakevens ---
        total_pop = 0.0

        if not breakevens:
            if max_profit > 0 and max_loss >= 0: total_pop = 1.0
            else: total_pop = 0.0
        
        elif len(breakevens) == 1:
            be = breakevens[0]
            test_price = be + (self.strike_distance * 0.1) 
            
            # <<< THE FIX: Pass self.realized_pnl to the helper >>>
            if _get_total_pnl_at_price(test_price, self.portfolio, self.realized_pnl) > 0:
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
                
                # <<< THE FIX: Pass self.realized_pnl to the helper >>>
                if _get_total_pnl_at_price(test_price, self.portfolio, self.realized_pnl) > 0:
                    greeks_upper = self.bs_manager.get_all_greeks_and_price(current_price, upper_b, days_to_expiry, vol, False) if upper_b != np.inf else None
                    prob_below_upper = _numba_cdf(-greeks_upper['d2']) if greeks_upper else 1.0

                    greeks_lower = self.bs_manager.get_all_greeks_and_price(current_price, lower_b, days_to_expiry, vol, False) if lower_b != -np.inf else None
                    prob_below_lower = _numba_cdf(-greeks_lower['d2']) if greeks_lower else 0.0
                    
                    total_pop += (prob_below_upper - prob_below_lower)

        return {
            'max_profit': max_profit, 'max_loss': max_loss,
            'rr_ratio': rr_ratio, 'prob_profit': np.clip(total_pop, 0.0, 1.0)
        }

    def _calculate_position_breakevens(self, portfolio_df: pd.DataFrame, net_premium: float) -> Tuple[float, float]:
        """
        The definitive, correct breakeven calculator. It is now fully strategy-aware and
        can correctly handle all standard structures, including Iron Flies and Butterflies.
        """
        lower_breakeven = 0.0
        upper_breakeven = float('inf')
        premium_cushion = abs(net_premium)
        
        is_single_leg = len(portfolio_df) == 1
        is_debit_trade = net_premium > 0

        # Group legs by strike for easier analysis
        strikes = portfolio_df['strike_price'].unique()
        
        # --- THE DEFINITIVE FIX: Identify strategy by structure ---

        if is_single_leg:
            # Case 1: Single Legs
            pos = portfolio_df.iloc[0]
            if pos['direction'] == 'long':
                if pos['type'] == 'call': lower_breakeven = pos['strike_price'] + premium_cushion
                else: upper_breakeven = pos['strike_price'] - premium_cushion
            else: # Single Short
                if pos['type'] == 'call': upper_breakeven = pos['strike_price'] + premium_cushion
                else: lower_breakeven = pos['strike_price'] - premium_cushion
        
        elif len(strikes) == 2:
            # Case 2: Two-Strike Strategies (Verticals, Strangles)
            if is_debit_trade: # Long Vertical or Long Strangle
                lower_breakeven = min(strikes) - premium_cushion
                upper_breakeven = max(strikes) + premium_cushion
            else: # Short Vertical or Short Strangle
                lower_breakeven = min(strikes) - premium_cushion
                upper_breakeven = max(strikes) + premium_cushion
        
        elif len(strikes) == 3:
            # Case 3: Three-Strike Strategies (Butterflies, Iron Condors, Iron Flies)
            center_strike = sorted(strikes)[1]
            if is_debit_trade: # Long Butterfly or Long Iron Fly
                # Breakevens are based on the center strike
                lower_breakeven = center_strike - premium_cushion
                upper_breakeven = center_strike + premium_cushion
            else: # Short Butterfly, Short Iron Fly, or Iron Condor
                short_strikes = sorted([p['strike_price'] for _, p in portfolio_df.iterrows() if p['direction'] == 'short'])
                lower_breakeven = short_strikes[0] - premium_cushion
                upper_breakeven = short_strikes[-1] + premium_cushion
        
        return lower_breakeven, upper_breakeven

    def _calculate_probability_outside_range(
        self, lower_bound: float, upper_bound: float,
        current_price: float, iv_bin_index: int, days_to_expiry: float
    ) -> float:
        """
        A specialized helper to calculate the probability of the price finishing
        outside a given lower and upper bound at expiration.
        """
        prob_of_loss = 0.0

        # Calculate probability of finishing below the lower bound (if it's a risk)
        if lower_bound > 0:
            # Use ATM IV as a reasonable proxy for the market's expectation
            vol_put = self.market_rules_manager.get_implied_volatility(0, 'put', iv_bin_index)
            greeks_put = self.bs_manager.get_all_greeks_and_price(current_price, lower_bound, days_to_expiry, vol_put, is_call=False)
            # The probability of a put expiring ITM is N(-d2)
            prob_of_loss += _numba_cdf(-greeks_put['d2'])

        # Calculate probability of finishing above the upper bound (if it's a risk)
        if upper_bound < float('inf'):
            vol_call = self.market_rules_manager.get_implied_volatility(0, 'call', iv_bin_index)
            greeks_call = self.bs_manager.get_all_greeks_and_price(current_price, upper_bound, days_to_expiry, vol_call, is_call=True)
            # The probability of a call expiring ITM is N(d2)
            prob_of_loss += _numba_cdf(greeks_call['d2'])

        return prob_of_loss

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
            vol = self.market_rules_manager.get_implied_volatility(offset, pos['type'], iv_bin_index)
            
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
        Sorts the portfolio. Uses a simple, chronological sort for training and a
        more complex, human-readable sort for evaluation.
        """
        if not self.portfolio.empty:
            # The key for human analysis and debugging
            sort_key = ['creation_id']
            self.portfolio = self.portfolio.sort_values(by=sort_key).reset_index(drop=True)

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
        source of truth for updating hedge status and taking the post-action snapshot.
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
            
            self.initial_net_premium = sum(leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1) for leg in trades_to_execute)
            
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

    def _price_legs(self, legs: List[Dict], current_price: float, iv_bin_index: int) -> List[Dict]:
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        for leg in legs:
            offset = round((leg['strike_price'] - atm_price) / self.strike_distance)
            vol = self.market_rules_manager.get_implied_volatility(offset, leg['type'], iv_bin_index)
            greeks = self.bs_manager.get_all_greeks_and_price(current_price, leg['strike_price'], leg['days_to_expiry'], vol, leg['type'] == 'call')
            leg['entry_premium'] = self.bs_manager.get_price_with_spread(greeks['price'], is_buy=(leg['direction'] == 'long'), bid_ask_spread_pct=self.bid_ask_spread_pct)
        return legs

    def _calculate_universal_risk_profile(self, legs: List[Dict], realized_pnl: float = 0.0) -> Dict:
        """
        The definitive, single-pass, universal risk engine.
        It simulates the P&L curve for any given set of legs and calculates
        max profit, max loss, profit factor, and all breakeven points in one go.
        It now correctly includes realized P&L for a total portfolio view.
        """
        # --- A. Handle Edge Cases ---
        if not legs:
            return {
                'max_profit': realized_pnl, 'max_loss': realized_pnl,
                'profit_factor': float('inf') if realized_pnl > 0 else 0.0, 'breakevens': []
            }

        if len(legs) == 1 and legs[0]['direction'] == 'short':
            entry_premium_total = legs[0]['entry_premium'] * self.lot_size
            # Even the naked short needs to consider realized pnl for its final value
            breakeven = legs[0]['strike_price'] + legs[0]['entry_premium']
            return {
                'max_profit': entry_premium_total + realized_pnl,
                'max_loss': -self.undefined_risk_cap + realized_pnl,
                'profit_factor': 0.01,
                'breakevens': [breakeven]
            }

        # --- B. The Main Simulation Loop ---
        def get_pnl_at_price(price, trade_legs):
            pnl = 0.0
            for leg in trade_legs:
                pnl_mult = 1 if leg['direction'] == 'long' else -1
                leg_pnl = max(0, price - leg['strike_price']) if leg['type'] == 'call' else max(0, leg['strike_price'] - price)
                pnl += (leg_pnl - leg['entry_premium']) * pnl_mult
            return pnl * self.lot_size

        sim_low = self.start_price * 0.25
        sim_high = self.start_price * 2.5
        price_range = np.linspace(sim_low, sim_high, 500)
        
        pnl_values = []
        breakevens = []

        # Calculate the initial P&L for the first point
        unrealized_pnl_prev = get_pnl_at_price(price_range[0], legs)
        pnl_prev = unrealized_pnl_prev + realized_pnl
        pnl_values.append(pnl_prev)

        for i in range(1, len(price_range)):
            price_curr = price_range[i]
            unrealized_pnl_curr = get_pnl_at_price(price_curr, legs)
            pnl_curr = unrealized_pnl_curr + realized_pnl
            pnl_values.append(pnl_curr)

            if np.sign(pnl_curr) != np.sign(pnl_prev):
                price_prev = price_range[i-1]
                be = price_prev - pnl_prev * (price_curr - price_prev) / (pnl_curr - pnl_prev)
                breakevens.append(be)
            pnl_prev = pnl_curr

        # --- C. Calculate Final Metrics from Simulation Results ---
        max_profit = max(pnl_values) if pnl_values else 0.0
        max_loss = min(pnl_values) if pnl_values else 0.0
        
        positive_pnls = [p for p in pnl_values if p > 0]
        negative_pnls = [p for p in pnl_values if p <= 0]
        sum_wins = sum(positive_pnls)
        sum_losses = abs(sum(negative_pnls))
        
        profit_factor = 999.0 if sum_losses < 1e-6 and sum_wins > 0 else (sum_wins / sum_losses if sum_losses > 1e-6 else 0.0)

        return {
            'max_profit': min(self.undefined_risk_cap, max_profit),
            'max_loss': max(-self.undefined_risk_cap, max_loss),
            'profit_factor': profit_factor,
            'breakevens': sorted(breakevens)
        }

    def _open_single_leg(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) < self.max_positions, "Illegal attempt to open a leg when portfolio is full. Action mask failed."

        if len(self.portfolio) >= self.max_positions: return
        _, direction, type, strike_str = action_name.split('_')
        offset = int(strike_str.replace('ATM', ''))
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        strike_price = atm_price + (offset * self.strike_distance)
        legs = [{'type': type.lower(), 'direction': direction.lower(), 'strike_price': strike_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}]
        legs = self._price_legs(legs, current_price, iv_bin_index)
        pnl  = self._calculate_universal_risk_profile(legs)
        pnl['strategy_id'] = self.strategy_name_to_id.get(f"{direction}_{type}", -1)
        self._execute_trades(legs, pnl)

    def _open_straddle(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) <= self.max_positions - 2, "Illegal attempt to open a straddle when there are less than two positions available."
        if len(self.portfolio) > self.max_positions - 2: return
        direction = 'long' if 'LONG' in action_name else 'short'
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        legs = [{'type': 'call', 'direction': direction, 'strike_price': atm_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': direction, 'strike_price': atm_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}]
        legs = self._price_legs(legs, current_price, iv_bin_index)
        pnl  = self._calculate_universal_risk_profile(legs)
        pnl['strategy_id'] = self.strategy_name_to_id.get(action_name.replace('OPEN_', '').replace('_ATM', ''), -1)
        self._execute_trades(legs, pnl)

    def _open_strangle(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) <= self.max_positions - 2, "Illegal attempt to open a strangle when there are less than two positions available."
        """Opens a two-leg strangle. (CORRECTED)"""
        if len(self.portfolio) > self.max_positions - 2: return

        parts = action_name.split('_')
        direction, width_str = parts[1], parts[4]
        strike_offset = int(width_str) * 2 * self.strike_distance

        atm_price = self.market_rules_manager.get_atm_price(current_price)
        
        legs = [
            {'type': 'call', 'direction': direction.lower(), 'strike_price': atm_price + strike_offset, 'entry_step': current_step, 'days_to_expiry': days_to_expiry},
            {'type': 'put', 'direction': direction.lower(), 'strike_price': atm_price - strike_offset, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}
        ]
        
        legs = self._price_legs(legs, current_price, iv_bin_index)
        
        # --- THE FIX ---
        canonical_strategy_name = action_name.replace('OPEN_', '').replace('_ATM', '')
        
        pnl  = self._calculate_universal_risk_profile(legs)
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)
        self._execute_trades(legs, pnl)

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

            vol = self.market_rules_manager.get_implied_volatility(offset, option_type, iv_bin_index)
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
                    # print(f"DEBUG: Successfully found strikes for tier {tier['short']}/{tier['long']}.")
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
        canonical_strategy_name = action_name.replace('OPEN_', '')
        pnl  = self._calculate_universal_risk_profile(legs)
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)
        self._execute_trades(legs, pnl)

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
        canonical_strategy_name = action_name.replace('OPEN_', '')
        pnl  = self._calculate_universal_risk_profile(legs)
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)
        self._execute_trades(legs, pnl)

    def _open_vertical_spread(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) <= self.max_positions - 2, "Illegal attempt to open a Vertical Spread."
        """Opens a two-leg vertical spread. (CORRECTED)"""
        if len(self.portfolio) > self.max_positions - 2: return
        
        parts = action_name.split('_')
        direction, option_type, width_str = parts[1], parts[3].lower(), parts[4]
        width_in_price = int(width_str) * 2 * self.strike_distance
        
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        strike1 = atm_price
        strike2 = atm_price + width_in_price if option_type == 'call' else atm_price - width_in_price

        legs = []
        if direction == 'LONG':
            legs.append({'type': option_type, 'direction': 'long', 'strike_price': strike1})
            legs.append({'type': option_type, 'direction': 'short', 'strike_price': strike2})
        else: # SHORT
            legs.append({'type': option_type, 'direction': 'short', 'strike_price': strike1})
            legs.append({'type': option_type, 'direction': 'long', 'strike_price': strike2})

        for leg in legs:
            leg['entry_step'] = current_step
            leg['days_to_expiry'] = days_to_expiry
        
        legs = self._price_legs(legs, current_price, iv_bin_index)
        
        # --- THE FIX ---
        canonical_strategy_name = action_name.replace('OPEN_', '')
        
        pnl  = self._calculate_universal_risk_profile(legs)
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)
        self._execute_trades(legs, pnl)

    def _open_butterfly(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) <= self.max_positions - 4, "Illegal attempt to open a ButterFly."
        """Opens a three-strike butterfly spread. (CORRECTED)"""
        if len(self.portfolio) > self.max_positions - 4:
            return

        parts = action_name.split('_')
        # e.g., from 'OPEN_LONG_CALL_FLY_1'
        direction, option_type, width_str = parts[1], parts[2].lower(), parts[4]
        width_in_price = int(width_str) * 2 * self.strike_distance

        atm_price = self.market_rules_manager.get_atm_price(current_price)
        strike_lower = atm_price - width_in_price
        strike_middle = atm_price
        strike_upper = atm_price + width_in_price

        wing_direction = 'long' if direction == 'LONG' else 'short'
        body_direction = 'short' if direction == 'LONG' else 'long'

        legs = [
            {'type': option_type, 'direction': wing_direction, 'strike_price': strike_lower},
            {'type': option_type, 'direction': body_direction, 'strike_price': strike_middle},
            {'type': option_type, 'direction': body_direction, 'strike_price': strike_middle},
            {'type': option_type, 'direction': wing_direction, 'strike_price': strike_upper},
        ]

        for leg in legs:
            leg['entry_step'] = current_step
            leg['days_to_expiry'] = days_to_expiry

        legs = self._price_legs(legs, current_price, iv_bin_index)

        # --- THE FIX ---
        # Derive the canonical name directly from the action name.
        # This turns 'OPEN_LONG_CALL_FLY_1' into 'LONG_CALL_FLY_1'
        canonical_strategy_name = action_name.replace('OPEN_', '')

        pnl  = self._calculate_universal_risk_profile(legs)
        # Look up the correct, derived key.
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)
        self._execute_trades(legs, pnl)

    def _open_delta_strangle(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        if len(self.portfolio) > self.max_positions - 2: return

        try:
            parts = action_name.split('_')
            direction = parts[1].lower()
            target_delta_int = int(parts[4])
        except (ValueError, IndexError): return

        legs = []
        
        # --- THE NEW TIERED LOGIC ---
        # 1. Create a prioritized list of deltas to search, starting with the agent's choice.
        all_deltas = [15, 20, 25, 30]
        start_index = all_deltas.index(target_delta_int)
        delta_search_list = [d / 100.0 for d in all_deltas[start_index:]] # e.g., [0.20, 0.25, 0.30]

        # 2. Use our new helper to find the best available strikes.
        strike_call = self._find_strike_from_delta_list(delta_search_list, 'call', current_price, iv_bin_index, days_to_expiry)
        strike_put = self._find_strike_from_delta_list([-d for d in delta_search_list], 'put', current_price, iv_bin_index, days_to_expiry)
        
        # 3. The guard rail remains the same, checking if the search was successful.
        if strike_call is not None and strike_put is not None and strike_put < strike_call:
            legs = [
                {'type': 'call', 'direction': direction, 'strike_price': strike_call, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': direction, 'strike_price': strike_put, 'days_to_expiry': days_to_expiry}
            ]
        else:
            print(f"Warning: Could not find any valid delta strikes for {action_name}. Falling back to fixed-width.")

        # --- Fallback Logic ---
        # If the `legs` list is still empty, it means the dynamic search failed.
        if not legs:
            # Fall back to a safe, fixed-width strangle (width = 2)
            atm_price = self.market_rules_manager.get_atm_price(current_price)
            strike_offset = 2 * self.strike_distance
            legs = [
                {'type': 'call', 'direction': direction, 'strike_price': atm_price + strike_offset, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': direction, 'strike_price': atm_price - strike_offset, 'days_to_expiry': days_to_expiry}
            ]

        # --- Finalize the Trade ---
        for leg in legs: leg['entry_step'] = current_step
        legs = self._price_legs(legs, current_price, iv_bin_index)
        
        # Use the original action name for the ID to give the agent proper credit
        canonical_strategy_name = action_name.replace('OPEN_', '')
        pnl  = self._calculate_universal_risk_profile(legs)
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)
        self._execute_trades(legs, pnl)

