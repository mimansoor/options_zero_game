# zoo/options_zero_game/envs/portfolio_manager.py
import pandas as pd
import numpy as np
import math
from typing import Dict, List, Any
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
        self.strike_distance = cfg['strike_distance']
        self.undefined_risk_cap = self.initial_cash * 10
        self.strategy_name_to_id = cfg.get('strategy_name_to_id', {})
        self.max_strike_offset = cfg['max_strike_offset']
        
        # Managers
        self.bs_manager = bs_manager
        self.market_rules_manager = market_rules_manager

        # State variables
        self.portfolio = pd.DataFrame()
        self.pre_step_portfolio: pd.DataFrame = pd.DataFrame()
        self.realized_pnl: float = 0.0
        self.high_water_mark: float = 0.0
        self.next_creation_id: int = 0
        self.initial_net_premium: float = 0.0
        self.portfolio_columns = ['type', 'direction', 'entry_step', 'strike_price', 'entry_premium', 'days_to_expiry', 'creation_id', 'strategy_id', 'strategy_max_profit', 'strategy_max_loss', 'is_hedged']
        self.portfolio_dtypes = {'type': 'object', 'direction': 'object', 'entry_step': 'int64', 'strike_price': 'float64', 'entry_premium': 'float64', 'days_to_expiry': 'float64', 'creation_id': 'int64', 'strategy_id': 'int64', 'strategy_max_profit': 'float64', 'strategy_max_loss': 'float64', 'is_hedged': 'bool'}
    
    def reset(self):
        """Resets the portfolio to an empty state for a new episode."""
        self.portfolio = pd.DataFrame(columns=self.portfolio_columns).astype(self.portfolio_dtypes)
        self.pre_step_portfolio = self.portfolio.copy()
        self.realized_pnl = 0.0
        self.high_water_mark = self.initial_cash
        self.next_creation_id = 0
        self.initial_net_premium = 0.0

    # --- Public Methods (called by the main environment) ---

    def get_pre_step_portfolio(self) -> pd.DataFrame:
        """Public API to allow the logger to get the portfolio state before the time step."""
        return self.pre_step_portfolio

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

    def update_high_water_mark(self, equity: float):
        self.high_water_mark = max(self.high_water_mark, equity)

    def update_positions_after_time_step(self, days_of_decay: int, current_price: float, iv_bin_index: int):
        """
        Applies time decay and handles expirations. Crucially, it saves a snapshot
        of the portfolio state *before* the time step is applied.
        """
        # Save a snapshot of the portfolio right after the action was taken,
        # but BEFORE time has advanced to the next day.
        self.pre_step_portfolio = self.portfolio.copy()
        
        # Now, the rest of the method proceeds as before.
        if days_of_decay == 0 or self.portfolio.empty:
            return

        self.portfolio['days_to_expiry'] = (self.portfolio['days_to_expiry'] - days_of_decay).clip(lower=0)
        
        expired_indices = self.portfolio[self.portfolio['days_to_expiry'] == 0].index
        if not expired_indices.empty:
            for idx in sorted(expired_indices, reverse=True):
                self.close_position(idx, current_price, iv_bin_index)

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
        self.realized_pnl += pnl * self.lot_size
        
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
            pnl_profile = self._calculate_strategy_pnl(remaining_legs, new_strategy_name)
            pnl_profile['strategy_id'] = self.strategy_name_to_id.get(new_strategy_name, -3)

            # Re-add the remaining legs with their new, correct, unified risk profile.
            for leg in remaining_legs:
                leg['creation_id'] = original_creation_id
            self._execute_trades(remaining_legs, pnl_profile)

    def close_all_positions(self, current_price: float, iv_bin_index: int):
        while not self.portfolio.empty:
            self.close_position(-1, current_price, iv_bin_index)

    def _update_hedge_status(self):
        """
        Calculates and sets the 'is_hedged' status for every leg in the portfolio.
        A leg is truly HEDGED only if both its max profit AND max loss are defined.
        It is NAKED if either side has undefined risk/reward.
        """
        if self.portfolio.empty:
            return

        # --- THE FIX ---
        
        # Condition 1: Is the maximum loss defined?
        # This is TRUE if the max_loss is greater than our "infinite" placeholder.
        has_defined_max_loss = self.portfolio['strategy_max_loss'] > -self.undefined_risk_cap
        
        # Condition 2: Is the maximum profit defined?
        # This is TRUE if the max_profit is less than our "infinite" placeholder.
        has_defined_max_profit = self.portfolio['strategy_max_profit'] < self.undefined_risk_cap

        # A position is only considered hedged if BOTH conditions are true.
        is_hedged = has_defined_max_loss & has_defined_max_profit
        
        self.portfolio['is_hedged'] = is_hedged

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
                # --- OTM Rule ---
                # The position is not yet a loser. Create a standard 1-strike wide spread
                # relative to the NAKED leg's strike.
                hedge_width = self.strike_distance * 2
                if hedge_type == 'call':
                    hedge_strike = naked_strike + hedge_width
                else: # Put
                    hedge_strike = naked_strike - hedge_width

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
            
            pnl = self._calculate_strategy_pnl(transformed_strategy_legs, new_strategy_name)
            pnl['strategy_id'] = self.strategy_name_to_id.get(new_strategy_name, -1)
            
            # --- 5. Update Portfolio Atomically ---
            self.portfolio = self.portfolio[self.portfolio['creation_id'] != original_creation_id].reset_index(drop=True)
            for leg in transformed_strategy_legs:
                leg['creation_id'] = original_creation_id
            self._execute_trades(transformed_strategy_legs, pnl)
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse HEDGE action '{action_name}'. Error: {e}")
            return

    def shift_position(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int):
        """
        Atomically transforms an entire strategy by shifting one of its legs.
        The risk profile of the complete, new structure is re-evaluated.
        """
        try:
            parts = action_name.split('_')
            direction, position_index = parts[1].upper(), int(parts[3])
            if not (0 <= position_index < len(self.portfolio)): return

            original_pos = self.portfolio.iloc[position_index].copy()
            original_creation_id = original_pos['creation_id']
            
            # 1. Define the new leg that will replace the original one.
            strike_modifier = self.strike_distance if direction == 'UP' else -self.strike_distance
            new_strike_price = original_pos['strike_price'] + strike_modifier
            new_leg = {
                'type': original_pos['type'], 'direction': original_pos['direction'],
                'strike_price': new_strike_price, 'entry_step': current_step,
                'days_to_expiry': original_pos['days_to_expiry'],
            }

            # 2. Re-assemble the full, transformed strategy.
            other_legs = [row.to_dict() for idx, row in self.portfolio.iterrows() if row['creation_id'] == original_creation_id and idx != position_index]
            transformed_strategy_legs = other_legs + [new_leg]

            # 3. Price the new leg and determine the new strategy profile.
            transformed_strategy_legs = self._price_legs(transformed_strategy_legs, current_price, iv_bin_index)
            new_strategy_name = f"CUSTOM_SHIFTED_{len(transformed_strategy_legs)}" # A generic name for the new state
            pnl = self._calculate_strategy_pnl(transformed_strategy_legs, new_strategy_name)
            pnl['strategy_id'] = self.strategy_name_to_id.get(new_strategy_name, -2)

            # 4. Atomically update the portfolio: remove all old legs and insert all new legs.
            self.portfolio = self.portfolio[self.portfolio['creation_id'] != original_creation_id].reset_index(drop=True)
            for leg in transformed_strategy_legs:
                leg['creation_id'] = original_creation_id # Maintain the strategic link
            self._execute_trades(transformed_strategy_legs, pnl)

        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse shift action '{action_name}'. Error: {e}")

    def shift_to_atm(self, action_name: str, current_price: float, iv_bin_index: int, current_step: int):
        """
        Atomically transforms an entire strategy by shifting one leg to the ATM strike.
        The risk profile of the complete, new structure is re-evaluated.
        """
        try:
            position_index = int(action_name.split('_')[-1])
            if not (0 <= position_index < len(self.portfolio)): return

            original_pos = self.portfolio.iloc[position_index].copy()
            original_creation_id = original_pos['creation_id']
            
            # 1. Define the new leg.
            new_atm_strike = self.market_rules_manager.get_atm_price(current_price)
            new_leg = {
                'type': original_pos['type'], 'direction': original_pos['direction'],
                'strike_price': new_atm_strike, 'entry_step': current_step,
                'days_to_expiry': original_pos['days_to_expiry'],
            }

            # 2. Re-assemble the full, transformed strategy.
            other_legs = [row.to_dict() for idx, row in self.portfolio.iterrows() if row['creation_id'] == original_creation_id and idx != position_index]
            transformed_strategy_legs = other_legs + [new_leg]

            # 3. Price and profile the new strategy.
            transformed_strategy_legs = self._price_legs(transformed_strategy_legs, current_price, iv_bin_index)
            new_strategy_name = f"CUSTOM_SHIFTED_ATM_{len(transformed_strategy_legs)}"
            pnl = self._calculate_strategy_pnl(transformed_strategy_legs, new_strategy_name)
            pnl['strategy_id'] = self.strategy_name_to_id.get(new_strategy_name, -2)

            # 4. Atomically update the portfolio.
            self.portfolio = self.portfolio[self.portfolio['creation_id'] != original_creation_id].reset_index(drop=True)
            for leg in transformed_strategy_legs:
                leg['creation_id'] = original_creation_id
            self._execute_trades(transformed_strategy_legs, pnl)
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse shift_to_atm action '{action_name}'. Error: {e}")

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

        atm_price = self.market_rules_manager.get_atm_price(current_price)
        
        all_greeks = []
        for _, pos in self.portfolio.iterrows():
            is_call = pos['type'] == 'call'
            
            # --- THE FIX IS HERE ---
            # 1. We must calculate the volatility (sigma) for the leg first.
            offset = round((pos['strike_price'] - atm_price) / self.strike_distance)
            vol = self.market_rules_manager.get_implied_volatility(offset, pos['type'], iv_bin_index)
            
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
        The definitive method to add new legs to the portfolio. It correctly:
        1. Asserts the strategy_id is valid.
        2. Calculates and stores the initial net premium for the stop-loss rule.
        3. Adds all necessary keys before creating the DataFrame.
        4. Updates the hedge status correctly.
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
        
        # 1. First, calculate and store the net premium of the incoming trade.
        self.initial_net_premium = sum(
            leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1)
            for leg in trades_to_execute
        )
        
        # 2. Then, prepare all legs for DataFrame creation.
        for trade in trades_to_execute:
            trade['is_hedged'] = False # Default placeholder
            trade['creation_id'] = self.next_creation_id
            self.next_creation_id += 1
            trade['strategy_id'] = strategy_id
            trade['strategy_max_profit'] = strategy_pnl.get('max_profit', 0.0)
            trade['strategy_max_loss'] = strategy_pnl.get('max_loss', 0.0)
        
        new_positions_df = pd.DataFrame(trades_to_execute).astype(self.portfolio_dtypes)
        self.portfolio = pd.concat([self.portfolio, new_positions_df], ignore_index=True)
        
        # 3. Finally, update the hedge status for the new, complete portfolio.
        self._update_hedge_status()

    def _price_legs(self, legs: List[Dict], current_price: float, iv_bin_index: int) -> List[Dict]:
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        for leg in legs:
            offset = round((leg['strike_price'] - atm_price) / self.strike_distance)
            vol = self.market_rules_manager.get_implied_volatility(offset, leg['type'], iv_bin_index)
            greeks = self.bs_manager.get_all_greeks_and_price(current_price, leg['strike_price'], leg['days_to_expiry'], vol, leg['type'] == 'call')
            leg['entry_premium'] = self.bs_manager.get_price_with_spread(greeks['price'], is_buy=(leg['direction'] == 'long'), bid_ask_spread_pct=self.bid_ask_spread_pct)
        return legs

    def _calculate_strategy_pnl(self, legs: List[Dict], strategy_name: str) -> Dict:
        """
        A robust, strategy-agnostic risk engine. It calculates the max profit/loss
        by analyzing the structure of the legs, not by relying on a strategy name.
        """
        if not legs: return {'max_profit': 0.0, 'max_loss': 0.0}

        # --- 1. Handle Single Leg Strategies ---
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

        # --- 2. For Multi-Leg, Determine if Risk is Defined or Undefined ---
        is_defined_risk = True
        short_calls = [l for l in legs if l['type'] == 'call' and l['direction'] == 'short']
        long_calls = [l for l in legs if l['type'] == 'call' and l['direction'] == 'long']
        short_puts = [l for l in legs if l['type'] == 'put' and l['direction'] == 'short']
        long_puts = [l for l in legs if l['type'] == 'put' and l['direction'] == 'long']

        # Check for naked short calls: is every short call protected by a long call at a higher strike?
        for sc in short_calls:
            if not any(lc['strike_price'] > sc['strike_price'] for lc in long_calls):
                is_defined_risk = False
                break
        if not is_defined_risk: # Check for naked short puts
            for sp in short_puts:
                if not any(lp['strike_price'] < sp['strike_price'] for lp in long_puts):
                    is_defined_risk = False
                    break
        
        # --- 3. Calculate PnL Based on Risk Profile ---
        net_premium = sum(leg['entry_premium'] * (1 if leg['direction'] == 'long' else -1) for leg in legs)
        
        if not is_defined_risk:
            # Undefined Risk Profile (e.g., a Straddle, Strangle, or broken spread)
            if net_premium > 0: # Debit
                max_loss = -net_premium * self.lot_size
                max_profit = self.undefined_risk_cap
            else: # Credit
                max_profit = -net_premium * self.lot_size
                max_loss = -self.undefined_risk_cap
            return {'max_profit': max_profit, 'max_loss': max_loss}
        else:
            # Defined Risk Profile (e.g., Verticals, Condors, Butterflies)
            # The existing width-based calculation is general and correct for all defined-risk spreads.
            is_debit_spread = net_premium > 0
            call_strikes = sorted([leg['strike_price'] for leg in legs if leg['type'] == 'call'])
            put_strikes = sorted([leg['strike_price'] for leg in legs if leg['type'] == 'put'])
            
            call_width = (call_strikes[-1] - call_strikes[0]) if len(call_strikes) > 1 else 0
            put_width = (put_strikes[-1] - put_strikes[0]) if len(put_strikes) > 1 else 0
            max_width = max(call_width, put_width) * self.lot_size

            if is_debit_spread:
                max_loss = -net_premium * self.lot_size
                max_profit = max_width + max_loss 
            else: # Credit spread
                max_profit = -net_premium * self.lot_size
                max_loss = -max_width + max_profit
            return {'max_profit': max_profit, 'max_loss': max_loss}

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
        pnl = self._calculate_strategy_pnl(legs, action_name)
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
        strike_offset = int(width_str) * 2 * self.strike_distance

        atm_price = self.market_rules_manager.get_atm_price(current_price)
        
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
        atm_price = self.market_rules_manager.get_atm_price(current_price)
        
        best_strike = atm_price
        smallest_delta_diff = float('inf')
        
        # Search a reasonable range of strikes around the at-the-money price
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
            atm_price = self.market_rules_manager.get_atm_price(current_price)
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
        atm_price = self.market_rules_manager.get_atm_price(current_price)
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

        pnl = self._calculate_strategy_pnl(legs, action_name)
        # Look up the correct, derived key.
        pnl['strategy_id'] = self.strategy_name_to_id.get(canonical_strategy_name, -1)
        self._execute_trades(legs, pnl)
