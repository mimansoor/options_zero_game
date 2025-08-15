# zoo/options_zero_game/entry/regression_suite.py
# <<< The Definitive Environment Regression Test Suite (CORRECTED) >>>

import gymnasium as gym
import numpy as np
import copy
import traceback

from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config
import zoo.options_zero_game.envs.options_zero_game_env

# ==============================================================================
#                            TEST HELPER FUNCTIONS
# ==============================================================================

def create_test_env(forced_opening_strategy: str):
    """A helper function to create a clean environment for a specific test case."""
    env_cfg = copy.deepcopy(main_config.env)
    env_cfg.is_eval_mode = True
    env_cfg.disable_opening_curriculum = True
    env_cfg.forced_opening_strategy_name = forced_opening_strategy
    env_cfg.forced_historical_symbol = 'SPY' # Use a fixed, reliable symbol
    return gym.make('OptionsZeroGame-v0', cfg=env_cfg)

# ==============================================================================
#                            INDIVIDUAL TEST CASES
# ==============================================================================

def test_hedge_naked_put():
    """Tests if HEDGE_NAKED_POS correctly converts a naked put into a Bull Put Spread."""
    test_name = "test_hedge_naked_put"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_SHORT_PUT_ATM-5')
    try:
        # Step 0: Reset the environment
        obs_dict = env.reset(seed=42)
        
        # <<< THE FIX: Execute the forced opening action on Step 0 >>>
        # We pass a HOLD action, but the env will override it with the forced strategy.
        timestep = env.step(env.actions_to_indices['HOLD'])
        
        # --- Now, we can begin the actual test ---
        portfolio_before = env.portfolio_manager.get_portfolio()
        assert len(portfolio_before) == 1, "Setup failed: Did not open 1 leg."
        assert not portfolio_before.iloc[0]['is_hedged'], "Setup failed: Leg is not naked."
        
        # Step 1: Execute the hedge action
        action_to_take = env.actions_to_indices['HEDGE_NAKED_POS_0']
        timestep = env.step(action_to_take)
        
        # --- Assertions ---
        portfolio_after = env.portfolio_manager.get_portfolio()
        assert len(portfolio_after) == 2, "Hedge failed: Expected 2 legs."
        assert portfolio_after.iloc[0]['is_hedged'] and portfolio_after.iloc[1]['is_hedged'], "Hedge failed: Legs not marked as hedged."
        stats = env.portfolio_manager.get_raw_portfolio_stats(env.price_manager.current_price, env.iv_bin_index)
        assert stats['max_loss'] > -500000, "Hedge failed: Max loss is still undefined."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_convert_strangle_to_condor():
    """Tests if CONVERT_TO_IRON_CONDOR correctly adds wings to a strangle."""
    test_name = "test_convert_strangle_to_condor"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_SHORT_STRANGLE_DELTA_20')
    try:
        # <<< THE FIX: Execute the forced opening action on Step 0 >>>
        env.reset(seed=42)
        env.step(env.actions_to_indices['HOLD'])
        assert len(env.portfolio_manager.get_portfolio()) == 2, "Setup failed: Did not open a 2-leg strangle."
        
        # Step 1: Execute the convert action
        action_to_take = env.actions_to_indices['CONVERT_TO_IRON_CONDOR']
        env.step(action_to_take)
        
        # --- Assertions ---
        portfolio_after = env.portfolio_manager.get_portfolio()
        assert len(portfolio_after) == 4, "Conversion failed: Expected 4 legs."
        assert portfolio_after['direction'].value_counts()['long'] == 2, "Conversion failed: Did not add 2 long wings."
        stats = env.portfolio_manager.get_raw_portfolio_stats(env.price_manager.current_price, env.iv_bin_index)
        assert stats['max_loss'] > -500000, "Conversion failed: Max loss is still undefined."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_convert_condor_to_vertical():
    """Tests if CONVERT_TO_BULL_PUT_SPREAD correctly removes the call side of a condor."""
    test_name = "test_convert_condor_to_vertical"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_SHORT_IRON_CONDOR')
    try:
        # <<< THE FIX: Execute the forced opening action on Step 0 >>>
        env.reset(seed=42)
        env.step(env.actions_to_indices['HOLD'])
        assert len(env.portfolio_manager.get_portfolio()) == 4, "Setup failed: Did not open a 4-leg condor."
        
        # Step 1: Execute the convert action
        action_to_take = env.actions_to_indices['CONVERT_TO_BULL_PUT_SPREAD']
        env.step(action_to_take)
        
        # --- Assertions ---
        portfolio_after = env.portfolio_manager.get_portfolio()
        assert len(portfolio_after) == 2, "Conversion failed: Expected 2 legs."
        assert all(portfolio_after['type'] == 'put'), "Conversion failed: Remaining legs are not all puts."
        
        final_strat_id = portfolio_after.iloc[0]['strategy_id']
        expected_strat_id = env.strategy_name_to_id['BULL_PUT_SPREAD']
        assert final_strat_id == expected_strat_id, f"Incorrect strategy ID. Expected {expected_strat_id}, got {final_strat_id}."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_shift_preserves_strategy():
    """
    Tests if a SHIFT action modifies a leg but preserves the overall strategy.
    This version is robust to changes in the portfolio's sort order.
    """
    test_name = "test_shift_preserves_strategy"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_SHORT_STRANGLE_DELTA_20')
    try:
        # Step 0: Open the position and get its initial state
        env.reset(seed=42)
        env.step(env.actions_to_indices['HOLD'])

        portfolio_before = env.portfolio_manager.get_portfolio()
        assert len(portfolio_before) == 2, "Setup failed: Did not open a 2-leg strangle."

        # --- THE FIX: Identify the target leg by its properties, not its index ---
        call_leg_before = portfolio_before[portfolio_before['type'] == 'call'].iloc[0]
        original_strike = call_leg_before['strike_price']
        original_strat_id = call_leg_before['strategy_id']
        # The action is targeting the leg at index 0, which is the CALL in this setup.
        action_to_take = env.actions_to_indices['SHIFT_UP_POS_0']

        # Step 1: Execute the shift action
        env.step(action_to_take)

        # --- Assertions ---
        portfolio_after = env.portfolio_manager.get_portfolio()
        assert len(portfolio_after) == 2, "Shift failed: Number of legs changed."

        # --- THE FIX: Find the call leg again after the sort ---
        call_leg_after = portfolio_after[portfolio_after['type'] == 'call'].iloc[0]

        # Check that the strike was modified correctly
        expected_new_strike = original_strike + env.strike_distance
        assert call_leg_after['strike_price'] == expected_new_strike, f"Shift failed: Strike price did not update correctly. Expected {expected_new_strike}, got {call_leg_after['strike_price']}."

        # Check that the strategy's identity was preserved across all legs
        assert portfolio_after.iloc[0]['strategy_id'] == original_strat_id, "Shift failed: Strategy ID was not preserved."
        assert portfolio_after.iloc[1]['strategy_id'] == original_strat_id, "Shift failed: Strategy ID was not preserved."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

# ==============================================================================
#                                TEST SUITE RUNNER
# ==============================================================================
if __name__ == "__main__":
    print("="*60)
    print("      STARTING OPTIONS-ZERO-GAME REGRESSION TEST SUITE")
    print("="*60)

    # A list of all test functions to be executed
    tests_to_run = [
        test_hedge_naked_put,
        test_convert_strangle_to_condor,
        test_convert_condor_to_vertical,
        test_shift_preserves_strategy,
    ]

    failures = []
    
    for test_func in tests_to_run:
        if not test_func():
            failures.append(test_func.__name__)
            
    print("\n" + "="*60)
    print("                 REGRESSION TEST SUMMARY")
    print("="*60)

    if not failures:
        print("\n✅ ✅ ✅  ALL TESTS PASSED SUCCESSFULLY! ✅ ✅ ✅\n")
    else:
        print(f"\n❌ ❌ ❌  {len(failures)} TEST(S) FAILED: ❌ ❌ ❌")
        for f in failures:
            print(f"    - {f}")
        print("\n")
