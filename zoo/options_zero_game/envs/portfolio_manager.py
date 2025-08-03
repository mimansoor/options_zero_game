# zoo/options_zero_game/envs/portfolio_manager.py
import pandas as pd
import numpy as np
import math
from typing import Dict, List, Any
from .black_scholes_manager import BlackScholesManager, _numba_cdf

class PortfolioManager:
    """
    Manages the state of the agent's portfolio, including all trades,
    PnL calculations, and constructing the portfolio observation state.
    This class is stateful and is the heart of the trading logic.
    """
    def __init__(self, cfg: Dict, bs_manager: BlackScholesManager):
        # Config
        self.initial_cash = cfg['initial_cash']
        self.lot_size = cfg['lot_size']
        self.bid_ask_spread_pct = cfg['bid_ask_spread_pct']
        self.max_positions = cfg['max_positions']
        self.strike_distance = cfg['strike_distance']
        self.undefined_risk_cap = self.initial_cash * 10
        self.strategy_name_to_id = cfg.get('strategy_name_to_id', {})
        self.max_strike_offset = cfg['max_strike_offset']
        
        # Managers
        self.bs_manager = bs_manager

        # State variables
        self.portfolio = pd.DataFrame()
        self.realized_pnl: float = 0.0
        self.high_water_mark: float = 0.0
        self.next_creation_id: int = 0
        self.initial_net_premium: float = 0.0
        self.portfolio_columns = ['type', 'direction', 'entry_step', 'strike_price', 'entry_premium', 'days_to_expiry', 'creation_id', 'strategy_id', 'strategy_max_profit', 'strategy_max_loss']
        self.portfolio_dtypes = {'type': 'object', 'direction': 'object', 'entry_step': 'int64', 'strike_price': 'float64', 'entry_premium': 'float64', 'days_to_expiry': 'float64', 'creation_id': 'int64', 'strategy_id': 'int64', 'strategy_max_profit': 'float64', 'strategy_max_loss': 'float64'}
    
    def reset(self):
        """Resets the portfolio to an empty state for a new episode."""
        self.portfolio = pd.DataFrame(columns=self.portfolio_columns).astype(self.portfolio_dtypes)
        self.realized_pnl = 0.0
        self.high_water_mark = self.initial_cash
        self.next_creation_id = 0
        self.initial_net_premium = 0.0

    # --- Public Methods (called by the main environment) ---

    def open_strategy(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        """Routes any 'OPEN_' action to the correct specialized private method."""
        if len(self.portfolio) >= self.max_positions: return

        # Route by most complex/specific keywords first to avoid misrouting
        if 'FLY' in action_name and 'IRON' not in action_name:
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
        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
        
        for i, pos in self.portfolio.iterrows():
            if i >= self.max_positions: break

            is_call = pos['type'] == 'call'
            direction_multiplier = 1.0 if pos['direction'] == 'long' else -1.0
            
            offset = round((pos['strike_price'] - atm_price) / self.strike_distance)
            vol = self.bs_manager.get_implied_volatility(offset, pos['type'], iv_bin_index)
            greeks = self.bs_manager.get_all_greeks_and_price(current_price, pos['strike_price'], pos['days_to_expiry'], vol, is_call)
            
            current_pos_idx = start_idx + i * state_size
            vec[current_pos_idx + pos_idx_map['IS_OCCUPIED']] = 1.0
            vec[current_pos_idx + pos_idx_map['TYPE_NORM']] = 1.0 if is_call else -1.0
            vec[current_pos_idx + pos_idx_map['DIRECTION_NORM']] = direction_multiplier
            vec[current_pos_idx + pos_idx_map['STRIKE_DIST_NORM']] = (pos['strike_price'] - atm_price) / (self._cfg.max_strike_offset * self.strike_distance)
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
           
    def get_total_pnl(self, current_price: float, iv_bin_index: int) -> float:
        if self.portfolio.empty:
            return self.realized_pnl
        
        def pnl_calculator(row):
            is_call = row['type'] == 'call'
            atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
            offset = round((row['strike_price'] - atm_price) / self.strike_distance)
            vol = self.bs_manager.get_implied_volatility(offset, row['type'], iv_bin_index)
            greeks = self.bs_manager.get_all_greeks_and_price(current_price, row['strike_price'], row['days_to_expiry'], vol, is_call)
            
            current_premium = self.bs_manager.get_price_with_spread(greeks['price'], row['direction'] == 'short', self.bid_ask_spread_pct)
            price_diff = current_premium - row['entry_premium']
            return price_diff * self.lot_size * (1 if row['direction'] == 'long' else -1)

        unrealized_pnl = self.portfolio.apply(pnl_calculator, axis=1).sum()
        return self.realized_pnl + unrealized_pnl

    def get_current_equity(self, current_price: float, iv_bin_index: int) -> float:
        return self.initial_cash + self.get_total_pnl(current_price, iv_bin_index)

    def update_high_water_mark(self, equity: float):
        self.high_water_mark = max(self.high_water_mark, equity)

    def update_positions_after_time_step(self, time_decay_days: float, current_price: float, iv_bin_index: int):
        if not self.portfolio.empty:
            self.portfolio['days_to_expiry'] = (self.portfolio['days_to_expiry'] - time_decay_days).clip(lower=0)
            expired_indices = self.portfolio[self.portfolio['days_to_expiry'] <= 1e-6].index
            if not expired_indices.empty:
                for idx in sorted(expired_indices, reverse=True):
                    self.close_position(idx, current_price, iv_bin_index)

    def close_position(self, position_index: int, current_price: float, iv_bin_index: int):
        """
        Closes a position at a specific index, with corrected assertion logic.
        """
        # --- THE FIX IS HERE ---
        
        # 1. First, handle the negative indexing to convert it to a positive index.
        #    e.g., if portfolio size is 1, an index of -1 becomes 0.
        if position_index < 0:
            position_index = len(self.portfolio) + position_index

        # 2. Now, assert that the *normalized*, positive index is valid.
        #    This will now correctly check `0 <= 0 < 1`, which is True.
        assert 0 <= position_index < len(self.portfolio), f"Attempted to close invalid or out-of-bounds index: {position_index}. Portfolio size is {len(self.portfolio)}."
        
        # The rest of the method logic remains the same.
        pos_to_close = self.portfolio.iloc[position_index]
        is_call = pos_to_close['type'] == 'call'
        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
        offset = round((pos_to_close['strike_price'] - atm_price) / self.strike_distance)
        vol = self.bs_manager.get_implied_volatility(offset, pos_to_close['type'], iv_bin_index)
        greeks = self.bs_manager.get_all_greeks_and_price(current_price, pos_to_close['strike_price'], pos_to_close['days_to_expiry'], vol, is_call)
        
        is_short = pos_to_close['direction'] == 'short'
        exit_premium = self.bs_manager.get_price_with_spread(greeks['price'], is_buy=is_short, bid_ask_spread_pct=self.bid_ask_spread_pct)
        
        pnl = (exit_premium - pos_to_close['entry_premium']) if pos_to_close['direction'] == 'long' else (pos_to_close['entry_premium'] - exit_premium)
        self.realized_pnl += pnl * self.lot_size
        self.portfolio = self.portfolio.drop(self.portfolio.index[position_index]).reset_index(drop=True)

    def close_all_positions(self, current_price: float, iv_bin_index: int):
        while not self.portfolio.empty:
            self.close_position(-1, current_price, iv_bin_index)

    def shift_position(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int):
        """Closes an existing position and opens a new one one strike away."""
        try:
            parts = action_name.split('_')
            direction, position_index = parts[1].upper(), int(parts[3])

            if not (0 <= position_index < len(self.portfolio)):
                return

            original_pos = self.portfolio.iloc[position_index].copy()
            days_to_expiry = original_pos['days_to_expiry'] # Preserve DTE

            strike_modifier = self.strike_distance if direction == 'UP' else -self.strike_distance
            new_strike_price = original_pos['strike_price'] + strike_modifier

            # Close the old position first
            self.close_position(position_index, current_price, iv_bin_index)

            # Open the new single-leg position
            new_leg = {
                'type': original_pos['type'],
                'direction': original_pos['direction'],
                'strike_price': new_strike_price,
                'entry_step': current_step,
                'days_to_expiry': days_to_expiry,
            }

            legs = self._price_legs([new_leg], current_price, iv_bin_index)
            pnl = self._calculate_strategy_pnl(legs, "SHIFT") # Strategy name is not critical here
            strategy_name = f"{legs[0]['direction'].upper()}_{legs[0]['type'].upper()}"
            pnl['strategy_id'] = self.strategy_name_to_id.get(strategy_name, -1)
            self._execute_trades(legs, pnl)

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse shift action '{action_name}'. Error: {e}")
            return

    def get_portfolio_summary(self, current_price: float, iv_bin_index: int) -> Dict:
        """
        Calculates high-level summary statistics for the entire portfolio.
        """
        if self.portfolio.empty:
            return {
                'max_profit': 0.0, 'max_loss': 0.0, 'rr_ratio': 0.0, 'prob_profit': 0.5
            }

        max_profit = self.portfolio.iloc[0]['strategy_max_profit']
        max_loss = self.portfolio.iloc[0]['strategy_max_loss']

        abs_max_loss = abs(max_loss)
        rr_ratio = abs(max_profit) / abs_max_loss if abs_max_loss > 1e-6 else 0.0

        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
        
        all_greeks = []
        for _, pos in self.portfolio.iterrows():
            is_call = pos['type'] == 'call'
            
            # --- THE FIX IS HERE ---
            # 1. We must calculate the volatility (sigma) for the leg first.
            offset = round((pos['strike_price'] - atm_price) / self.strike_distance)
            vol = self.bs_manager.get_implied_volatility(offset, pos['type'], iv_bin_index)
            
            # 2. Now we can call the function with all 5 required arguments.
            greeks = self.bs_manager.get_all_greeks_and_price(
                current_price,
                pos['strike_price'],
                pos['days_to_expiry'],
                vol,      # <-- The missing `sigma` argument
                is_call   # <-- The final `is_call` argument
            )
            all_greeks.append({'greeks': greeks, 'pos': pos})
            
        main_leg = max(all_greeks, key=lambda x: abs(x['greeks']['delta']))
        main_greeks = main_leg['greeks']
        main_pos = main_leg['pos']
        
        if main_pos['type'] == 'call':
            prob_profit = _numba_cdf(main_greeks['d2']) if main_pos['direction'] == 'long' else 1 - _numba_cdf(main_greeks['d2'])
        else: # Put
            prob_profit = 1 - _numba_cdf(main_greeks['d2']) if main_pos['direction'] == 'long' else _numba_cdf(main_greeks['d2'])

        return {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'rr_ratio': rr_ratio,
            'prob_profit': prob_profit if math.isfinite(prob_profit) else 0.5
        }

    def get_portfolio_greeks(self, current_price: float, iv_bin_index: int) -> Dict:
        """Calculates and returns a dictionary of normalized portfolio-level Greeks."""
        total_delta, total_gamma, total_theta, total_vega = 0.0, 0.0, 0.0, 0.0
        if self.portfolio.empty:
            return {'delta_norm': 0.0, 'gamma_norm': 0.0, 'theta_norm': 0.0, 'vega_norm': 0.0}

        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance

        for _, pos in self.portfolio.iterrows():
            direction_multiplier = 1 if pos['direction'] == 'long' else -1
            is_call = pos['type'] == 'call'
            
            # --- THE FIX IS HERE ---
            # 1. We must first calculate sigma (implied volatility) for the leg.
            offset = round((pos['strike_price'] - atm_price) / self.strike_distance)
            vol = self.bs_manager.get_implied_volatility(offset, pos['type'], iv_bin_index)
            
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
        if not self.portfolio.empty:
            self.portfolio = self.portfolio.sort_values(by=['strike_price', 'type', 'creation_id']).reset_index(drop=True)

    def render(self, current_price: float, current_step: int, iv_bin_index: int, steps_per_day: int):
        total_pnl = self.get_total_pnl(current_price, iv_bin_index)
        day = current_step // steps_per_day + 1
        print(f"Step: {current_step:04d} | Day: {day:02d} | Price: ${current_price:9.2f} | Positions: {len(self.portfolio):1d} | Total PnL: ${total_pnl:9.2f}")
        if not self.portfolio.empty:
            print(self.portfolio.to_string(index=False))

    def _execute_trades(self, trades_to_execute: List[Dict], strategy_pnl: Dict):
        """
        Adds a list of trade legs to the portfolio after a final assertion check.
        """
        if not trades_to_execute: return

        strategy_id = strategy_pnl.get('strategy_id', -1)

        # --- Defensive Assertion ---
        # Fails fast if a strategy name was not found in the ID dictionary.
        # This prevents the agent from training on corrupted/meaningless data and
        # immediately flags inconsistencies between action names and the config.
        assert strategy_id != -1, (
            f"\n\nCRITICAL ERROR: Strategy ID not found.\n"
            f"  - This means a strategy's canonical name did not have a key in the strategy_name_to_id dictionary.\n"
            f"  - The PnL object that caused the failure was: {strategy_pnl}\n"
            f"  - Please check the logic in the `_open_*` method that was called and ensure the derived strategy name is correct.\n"
        )

        # --- NEW LOGIC: Calculate and store the net premium ---
        # This is the initial debit (positive) or credit (negative) of the trade.
        self.initial_net_premium = sum(
            leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1)
            for leg in trades_to_execute
        )

        for trade in trades_to_execute:
            trade['creation_id'] = self.next_creation_id
            self.next_creation_id += 1
            trade['strategy_id'] = strategy_id
            trade['strategy_max_profit'] = strategy_pnl.get('max_profit', 0.0)
            trade['strategy_max_loss'] = strategy_pnl.get('max_loss', 0.0)

        new_positions_df = pd.DataFrame(trades_to_execute).astype(self.portfolio_dtypes)
        self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)

    def _price_legs(self, legs: List[Dict], current_price: float, iv_bin_index: int) -> List[Dict]:
        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
        for leg in legs:
            offset = round((leg['strike_price'] - atm_price) / self.strike_distance)
            vol = self.bs_manager.get_implied_volatility(offset, leg['type'], iv_bin_index)
            greeks = self.bs_manager.get_all_greeks_and_price(current_price, leg['strike_price'], leg['days_to_expiry'], vol, leg['type'] == 'call')
            leg['entry_premium'] = self.bs_manager.get_price_with_spread(greeks['price'], is_buy=(leg['direction'] == 'long'), bid_ask_spread_pct=self.bid_ask_spread_pct)
        return legs

    def _calculate_strategy_pnl(self, legs: List[Dict], strategy_name: str) -> Dict:
        if not legs: return {'max_profit': 0.0, 'max_loss': 0.0}

        # Single leg
        if len(legs) == 1:
            leg = legs[0]
            entry_premium_total = leg['entry_premium'] * self.lot_size
            if leg['direction'] == 'long':
                max_loss = -entry_premium_total
                max_profit = self.undefined_risk_cap if leg['type'] == 'call' else (leg['strike_price'] * self.lot_size) - entry_premium_total
            else: # short
                max_profit = entry_premium_total
                max_loss = -self.undefined_risk_cap if leg['type'] == 'call' else -((leg['strike_price'] * self.lot_size) - entry_premium_total)
            return {'max_profit': max_profit, 'max_loss': max_loss}

        # Multi-leg
        net_premium = sum(leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1) for leg in legs)
        is_debit_spread = net_premium > 0

        # Undefined Risk (Straddle/Strangle)
        if 'STRADDLE' in strategy_name or 'STRANGLE' in strategy_name:
            if is_debit_spread: # Long
                max_loss = -net_premium * self.lot_size
                max_profit = self.undefined_risk_cap
            else: # Short
                max_profit = -net_premium * self.lot_size
                max_loss = -self.undefined_risk_cap
            return {'max_profit': max_profit, 'max_loss': max_loss}

        # Defined Risk (Verticals, Condors, Flies, Butterflies)
        else:
            call_strikes = sorted([leg['strike_price'] for leg in legs if leg['type'] == 'call'])
            put_strikes = sorted([leg['strike_price'] for leg in legs if leg['type'] == 'put'])
            call_width = (call_strikes[-1] - call_strikes[0]) if len(call_strikes) > 1 else 0
            put_width = (put_strikes[-1] - put_strikes[0]) if len(put_strikes) > 1 else 0
            max_width = max(call_width, put_width) * self.lot_size

            if is_debit_spread: # Long positions
                max_loss = -net_premium * self.lot_size
                max_profit = max_width + max_loss 
            else: # Short positions
                max_profit = -net_premium * self.lot_size
                max_loss = -max_width + max_profit
            return {'max_profit': max_profit, 'max_loss': max_loss}

    def _open_single_leg(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) < self.max_positions, "Illegal attempt to open a leg when portfolio is full. Action mask failed."

        if len(self.portfolio) >= self.max_positions: return
        _, direction, type, strike_str = action_name.split('_')
        offset = int(strike_str.replace('ATM', ''))
        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
        strike_price = atm_price + (offset * self.strike_distance)
        legs = [{'type': type.lower(), 'direction': direction.lower(), 'strike_price': strike_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}]
        legs = self._price_legs(legs, current_price, iv_bin_index)
        pnl = self._calculate_strategy_pnl(legs, action_name)
        pnl['strategy_id'] = self.strategy_name_to_id.get(f"{direction}_{type}", -1)
        self._execute_trades(legs, pnl)

    def _open_straddle(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) <= self.max_positions - 2, "Illegal attempt to open a straddle when there are less than two positions available."
        if len(self.portfolio) > self.max_positions - 2: return
        direction = 'long' if 'LONG' in action_name else 'short'
        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
        legs = [{'type': 'call', 'direction': direction, 'strike_price': atm_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': direction, 'strike_price': atm_price, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}]
        legs = self._price_legs(legs, current_price, iv_bin_index)
        pnl = self._calculate_strategy_pnl(legs, action_name)
        pnl['strategy_id'] = self.strategy_name_to_id.get(action_name.replace('OPEN_', '').replace('_ATM', ''), -1)
        self._execute_trades(legs, pnl)

    def _open_strangle(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) <= self.max_positions - 2, "Illegal attempt to open a strangle when there are less than two positions available."
        """Opens a two-leg strangle. (CORRECTED)"""
        if len(self.portfolio) > self.max_positions - 2: return

        parts = action_name.split('_')
        direction, width_str = parts[1], parts[4]
        strike_offset = int(width_str) * self.strike_distance

        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
        
        legs = [
            {'type': 'call', 'direction': direction.lower(), 'strike_price': atm_price + strike_offset, 'entry_step': current_step, 'days_to_expiry': days_to_expiry},
            {'type': 'put', 'direction': direction.lower(), 'strike_price': atm_price - strike_offset, 'entry_step': current_step, 'days_to_expiry': days_to_expiry}
        ]
        
        legs = self._price_legs(legs, current_price, iv_bin_index)
        
        # --- THE FIX ---
        canonical_strategy_name = action_name.replace('OPEN_', '').replace('_ATM', '')
        
        pnl = self._calculate_strategy_pnl(legs, action_name)
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)
        self._execute_trades(legs, pnl)

    def _find_strike_for_delta(
        self, target_delta: float, option_type: str,
        current_price: float, iv_bin_index: int, days_to_expiry: float
    ) -> float:
        """
        Iteratively finds the strike price for an option that is closest to a target delta.
        """
        is_call = option_type == 'call'
        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
        
        best_strike = atm_price
        smallest_delta_diff = float('inf')
        
        # Search a reasonable range of strikes around the at-the-money price
        for offset in range(-self.max_strike_offset, self.max_strike_offset + 1):
            strike_price = atm_price + (offset * self.strike_distance)
            if strike_price <= 0: continue

            vol = self.bs_manager.get_implied_volatility(offset, option_type, iv_bin_index)
            greeks = self.bs_manager.get_all_greeks_and_price(current_price, strike_price, days_to_expiry, vol, is_call)
            
            current_delta = greeks['delta']
            delta_diff = abs(current_delta - target_delta)
            
            if delta_diff < smallest_delta_diff:
                smallest_delta_diff = delta_diff
                best_strike = strike_price
                
        return best_strike

    def _open_iron_condor(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        if len(self.portfolio) > self.max_positions - 4: return
        
        direction = 'long' if 'LONG' in action_name else 'short'
        legs = []

        if direction == 'short':
            strike_short_put = self._find_strike_for_delta(-0.30, 'put', current_price, iv_bin_index, days_to_expiry)
            strike_short_call = self._find_strike_for_delta(0.30, 'call', current_price, iv_bin_index, days_to_expiry)
            strike_long_put = self._find_strike_for_delta(-0.15, 'put', current_price, iv_bin_index, days_to_expiry)
            strike_long_call = self._find_strike_for_delta(0.15, 'call', current_price, iv_bin_index, days_to_expiry)
            
            are_strikes_valid = (strike_long_put < strike_short_put < strike_short_call < strike_long_call)
            
            if are_strikes_valid:
                legs = [
                    {'type': 'put', 'direction': 'short', 'strike_price': strike_short_put, 'days_to_expiry': days_to_expiry},
                    {'type': 'call', 'direction': 'short', 'strike_price': strike_short_call, 'days_to_expiry': days_to_expiry},
                    {'type': 'put', 'direction': 'long', 'strike_price': strike_long_put, 'days_to_expiry': days_to_expiry},
                    {'type': 'call', 'direction': 'long', 'strike_price': strike_long_call, 'days_to_expiry': days_to_expiry}
                ]
            else:
                print(f"Warning: Dynamic delta strikes for Iron Condor were illogical. Falling back to fixed-width.")

        if not legs:
            atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
            body_direction = 'long' if direction == 'long' else 'short'
            wing_direction = 'short' if direction == 'long' else 'long'
            
            legs = [
                {'type': 'put', 'direction': body_direction, 'strike_price': atm_price - self.strike_distance, 'days_to_expiry': days_to_expiry},
                {'type': 'call', 'direction': body_direction, 'strike_price': atm_price + self.strike_distance, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': wing_direction, 'strike_price': atm_price - (2 * self.strike_distance), 'days_to_expiry': days_to_expiry},
                {'type': 'call', 'direction': wing_direction, 'strike_price': atm_price + (2 * self.strike_distance), 'days_to_expiry': days_to_expiry}
            ]

        for leg in legs:
            leg['entry_step'] = current_step
        
        legs = self._price_legs(legs, current_price, iv_bin_index)
        canonical_strategy_name = action_name.replace('OPEN_', '')
        pnl = self._calculate_strategy_pnl(legs, action_name)
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)
        self._execute_trades(legs, pnl)

    def _open_iron_fly(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        if len(self.portfolio) > self.max_positions - 4: return
        
        direction = 'long' if 'LONG' in action_name else 'short'
        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
        legs = []

        if direction == 'short':
            body_legs = [
                {'type': 'call', 'direction': 'short', 'strike_price': atm_price, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': 'short', 'strike_price': atm_price, 'days_to_expiry': days_to_expiry}
            ]
            body_legs = self._price_legs(body_legs, current_price, iv_bin_index)
            net_credit_received = body_legs[0]['entry_premium'] + body_legs[1]['entry_premium']
            wing_width = round(net_credit_received / self.strike_distance) * self.strike_distance
            
            MAX_SENSIBLE_WIDTH = 5 * self.strike_distance
            
            if self.strike_distance <= wing_width <= MAX_SENSIBLE_WIDTH:
                strike_long_put = atm_price - wing_width
                strike_long_call = atm_price + wing_width
                
                wing_legs = [
                    {'type': 'put', 'direction': 'long', 'strike_price': strike_long_put, 'days_to_expiry': days_to_expiry},
                    {'type': 'call', 'direction': 'long', 'strike_price': strike_long_call, 'days_to_expiry': days_to_expiry}
                ]
                legs = body_legs + wing_legs
                # Price the new wings
                legs = self._price_legs(legs, current_price, iv_bin_index)
            else:
                print(f"Warning: Dynamic wing width ({wing_width}) for Iron Fly was out of bounds. Falling back to fixed-width.")

        if not legs:
            body_direction = 'long' if direction == 'long' else 'short'
            wing_direction = 'short' if direction == 'long' else 'long'
            wing_width = self.strike_distance
            
            strike_long_put = atm_price - wing_width
            strike_long_call = atm_price + wing_width

            legs = [
                {'type': 'call', 'direction': body_direction, 'strike_price': atm_price, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': body_direction, 'strike_price': atm_price, 'days_to_expiry': days_to_expiry},
                {'type': 'call', 'direction': wing_direction, 'strike_price': strike_long_call, 'days_to_expiry': days_to_expiry},
                {'type': 'put', 'direction': wing_direction, 'strike_price': strike_long_put, 'days_to_expiry': days_to_expiry}
            ]
        
        for leg in legs:
            leg['entry_step'] = current_step
        
        legs = self._price_legs(legs, current_price, iv_bin_index)
        canonical_strategy_name = action_name.replace('OPEN_', '')
        pnl = self._calculate_strategy_pnl(legs, action_name)
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)
        self._execute_trades(legs, pnl)

    def _open_vertical_spread(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int, days_to_expiry: float):
        # --- Defensive Assertion ---
        assert len(self.portfolio) <= self.max_positions - 2, "Illegal attempt to open a Vertical Spread."
        """Opens a two-leg vertical spread. (CORRECTED)"""
        if len(self.portfolio) > self.max_positions - 2: return
        
        parts = action_name.split('_')
        direction, option_type, width_str = parts[1], parts[3].lower(), parts[4]
        width_in_price = int(width_str) * self.strike_distance
        
        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
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
        
        pnl = self._calculate_strategy_pnl(legs, action_name)
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
        width_in_price = int(width_str) * self.strike_distance

        atm_price = int(current_price / self.strike_distance + 0.5) * self.strike_distance
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

        pnl = self._calculate_strategy_pnl(legs, action_name)
        # Look up the correct, derived key.
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)
        self._execute_trades(legs, pnl)
