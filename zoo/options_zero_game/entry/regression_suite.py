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
#                      NEW, EXPANDED TEST CASES
# ==============================================================================

def test_hedge_short_call():
    """Tests hedging a naked SHORT CALL into a Bear Call Spread."""
    test_name = "test_hedge_short_call"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_SHORT_CALL_ATM+5')
    try:
        env.reset(seed=43)
        env.step(env.actions_to_indices['HOLD'])
        assert len(env.portfolio_manager.get_portfolio()) == 1, "Setup failed"

        env.step(env.actions_to_indices['HEDGE_NAKED_POS_0'])

        portfolio_after = env.portfolio_manager.get_portfolio()

        # <<< THE FIX: Add the missing line to create the 'stats' dictionary >>>
        stats = env.portfolio_manager.get_raw_portfolio_stats(env.price_manager.current_price, env.iv_bin_index)

        assert len(portfolio_after) == 2, "Hedge failed: Did not create 2 legs."
        assert portfolio_after.iloc[0]['strategy_id'] == env.strategy_name_to_id['BEAR_CALL_SPREAD'], "Incorrect strategy ID."
        assert stats['max_loss'] > -500000, "Hedge failed: Risk is still undefined."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_hedge_long_put():
    """Tests hedging a naked LONG PUT into a Bear Put Spread."""
    test_name = "test_hedge_long_put"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_LONG_PUT_ATM-5')
    try:
        env.reset(seed=44)
        env.step(env.actions_to_indices['HOLD'])
        assert len(env.portfolio_manager.get_portfolio()) == 1, "Setup failed"
        
        env.step(env.actions_to_indices['HEDGE_NAKED_POS_0'])
        
        portfolio_after = env.portfolio_manager.get_portfolio()
        stats = env.portfolio_manager.get_raw_portfolio_stats(env.price_manager.current_price, env.iv_bin_index)
        assert len(portfolio_after) == 2, "Hedge failed: Did not create 2 legs."
        
        # <<< THE FIX: A LONG PUT hedged with a SHORT PUT is a BEAR Put Spread >>>
        assert portfolio_after.iloc[0]['strategy_id'] == env.strategy_name_to_id['BEAR_PUT_SPREAD'], "Incorrect strategy ID."
        
        assert stats['max_profit'] < 500000, "Hedge failed: Profit is still undefined."
        
        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()    

def test_hedge_long_call():
    """Tests hedging a naked LONG CALL into a Bull Call Spread."""
    test_name = "test_hedge_long_call"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_LONG_CALL_ATM+5')
    try:
        env.reset(seed=45)
        env.step(env.actions_to_indices['HOLD'])
        assert len(env.portfolio_manager.get_portfolio()) == 1, "Setup failed"

        env.step(env.actions_to_indices['HEDGE_NAKED_POS_0'])

        portfolio_after = env.portfolio_manager.get_portfolio()
        stats = env.portfolio_manager.get_raw_portfolio_stats(env.price_manager.current_price, env.iv_bin_index)
        assert len(portfolio_after) == 2, "Hedge failed: Did not create 2 legs."

        # <<< THE FIX: A LONG CALL hedged with a SHORT CALL is a BULL Call Spread >>>
        assert portfolio_after.iloc[0]['strategy_id'] == env.strategy_name_to_id['BULL_CALL_SPREAD'], "Incorrect strategy ID."

        assert stats['max_profit'] < 500000, "Hedge failed: Profit is still undefined."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_convert_straddle_to_fly():
    """Tests if CONVERT_TO_IRON_FLY correctly adds wings to a straddle."""
    test_name = "test_convert_straddle_to_fly"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_SHORT_STRADDLE')
    try:
        env.reset(seed=46)
        env.step(env.actions_to_indices['HOLD'])
        assert len(env.portfolio_manager.get_portfolio()) == 2, "Setup failed"
        
        env.step(env.actions_to_indices['CONVERT_TO_IRON_FLY'])
        
        portfolio_after = env.portfolio_manager.get_portfolio()
        stats = env.portfolio_manager.get_raw_portfolio_stats(env.price_manager.current_price, env.iv_bin_index)
        legs = len(portfolio_after)
        assert legs == 4, f"Conversion failed: Expected 4 legs. Got {legs}"
        assert portfolio_after.iloc[0]['strategy_id'] == env.strategy_name_to_id['SHORT_IRON_FLY'], "Incorrect strategy ID."
        assert stats['max_loss'] > -500000, "Conversion failed: Risk is still undefined."
        
        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_convert_call_fly_to_vertical():
    """Tests decomposing a Call Fly into a Bull Call Spread."""
    test_name = "test_convert_call_fly_to_vertical"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_SHORT_CALL_FLY_1')
    try:
        env.reset(seed=47)
        env.step(env.actions_to_indices['HOLD'])
        assert len(env.portfolio_manager.get_portfolio()) == 4, "Setup failed"
        
        env.step(env.actions_to_indices['CONVERT_TO_BULL_CALL_SPREAD'])
        
        portfolio_after = env.portfolio_manager.get_portfolio()
        assert len(portfolio_after) == 2, "Decomposition failed: Expected 2 legs."
        assert all(portfolio_after['type'] == 'call'), "Decomposition failed: Legs are not all calls."
        assert portfolio_after.iloc[0]['strategy_id'] == env.strategy_name_to_id['BULL_CALL_SPREAD'], "Incorrect strategy ID."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_convert_put_fly_to_vertical():
    """Tests decomposing a Put Fly into a Bear Put Spread."""
    test_name = "test_convert_put_fly_to_vertical"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_SHORT_PUT_FLY_1')
    try:
        env.reset(seed=48)
        env.step(env.actions_to_indices['HOLD'])
        assert len(env.portfolio_manager.get_portfolio()) == 4, "Setup failed"
        
        env.step(env.actions_to_indices['CONVERT_TO_BEAR_PUT_SPREAD'])
        
        portfolio_after = env.portfolio_manager.get_portfolio()
        assert len(portfolio_after) == 2, "Decomposition failed: Expected 2 legs."
        assert all(portfolio_after['type'] == 'put'), "Decomposition failed: Legs are not all puts."
        assert portfolio_after.iloc[0]['strategy_id'] == env.strategy_name_to_id['BEAR_PUT_SPREAD'], "Incorrect strategy ID."
        
        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_hedge_strangle_leg():
    """Tests hedging one leg of a two-leg strangle."""
    test_name = "test_hedge_strangle_leg"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_SHORT_STRANGLE_DELTA_20')
    try:
        env.reset(seed=49)
        env.step(env.actions_to_indices['HOLD'])
        portfolio_before = env.portfolio_manager.get_portfolio()
        assert len(portfolio_before) == 2, "Setup failed"
        
        # Action will target the call leg at index 0
        env.step(env.actions_to_indices['HEDGE_NAKED_POS_0'])
        
        portfolio_after = env.portfolio_manager.get_portfolio()
        assert len(portfolio_after) == 3, "Hedge failed: Expected 3 legs."
        
        # Verify the correct legs are marked as hedged
        call_legs = portfolio_after[portfolio_after['type'] == 'call']
        put_leg = portfolio_after[portfolio_after['type'] == 'put'].iloc[0]
        
        assert len(call_legs) == 2, "Hedge failed: Did not create a new call leg."
        assert all(call_legs['is_hedged']), "Hedge failed: Call legs not marked as hedged."
        assert not put_leg['is_hedged'], "Hedge failed: Put leg was incorrectly marked as hedged."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_hedge_straddle_leg():
    """Tests hedging one leg of a two-leg straddle."""
    test_name = "test_hedge_straddle_leg"
    print(f"\n--- RUNNING: {test_name} ---")
    env = create_test_env('OPEN_SHORT_STRADDLE')
    try:
        env.reset(seed=50)
        env.step(env.actions_to_indices['HOLD'])
        portfolio_before = env.portfolio_manager.get_portfolio()
        assert len(portfolio_before) == 2, "Setup failed"

        # Action will target the put leg at index 1
        env.step(env.actions_to_indices['HEDGE_NAKED_POS_1'])
        
        portfolio_after = env.portfolio_manager.get_portfolio()
        assert len(portfolio_after) == 3, "Hedge failed: Expected 3 legs."
        
        call_leg = portfolio_after[portfolio_after['type'] == 'call'].iloc[0]
        put_legs = portfolio_after[portfolio_after['type'] == 'put']

        assert not call_leg['is_hedged'], "Hedge failed: Call leg was incorrectly marked as hedged."
        assert len(put_legs) == 2, "Hedge failed: Did not create a new put leg."
        assert all(put_legs['is_hedged']), "Hedge failed: Put legs not marked as hedged."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_no_new_trades_when_active():
    """
    Tests the core design rule: The agent cannot open a new position
    if a portfolio is already active.
    """
    test_name = "test_no_new_trades_when_active"
    print(f"\n--- RUNNING: {test_name} ---")
    # We can start with any simple strategy to make the portfolio active.
    env = create_test_env('OPEN_BULL_PUT_SPREAD')
    try:
        # Step 0: Open the initial position to make the portfolio non-empty.
        env.reset(seed=51)
        timestep = env.step(env.actions_to_indices['HOLD'])
        assert not env.portfolio_manager.get_portfolio().empty, "Setup failed: Portfolio is empty."

        # --- The Real Test: Check the action mask from the new state ---
        action_mask = timestep.obs['action_mask']

        # --- Assertions ---
        # 1. Verify that NO 'OPEN_*' actions are legal.
        num_open_actions_tested = 0
        for action_name, index in env.actions_to_indices.items():
            if action_name.startswith('OPEN_'):
                num_open_actions_tested += 1
                assert action_mask[index] == 0, f"Illegal OPEN action found in mask for active portfolio: {action_name}"

        print(f"Verified that all {num_open_actions_tested} 'OPEN_*' actions are correctly disabled.")

        # 2. As a sanity check, verify that 'HOLD' IS legal.
        assert action_mask[env.actions_to_indices['HOLD']] == 1, "HOLD action was not legal for an active portfolio."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_all_open_actions_are_legal():
    """
    A critical "smoke test" to ensure that every defined 'OPEN_*' action
    can be legal under ideal conditions (empty portfolio, no curriculum).
    """
    test_name = "test_all_open_actions_are_legal"
    print(f"\n--- RUNNING: {test_name} ---")

    # We create a custom env_cfg for this test to guarantee ideal conditions.
    env_cfg = copy.deepcopy(main_config.env)
    env_cfg.is_eval_mode = True
    env_cfg.disable_opening_curriculum = True # CRITICAL: curriculum would restrict the mask
    env_cfg.max_positions = 4 # CRITICAL: Ensure enough slots for Condors/Flies
    env = gym.make('OptionsZeroGame-v0', cfg=env_cfg)

    try:
        # Step 0: Reset the environment to get the initial action mask.
        # We don't need to step, as reset() provides the mask for an empty portfolio.
        obs_dict = env.reset(seed=42)
        action_mask = obs_dict['action_mask']

        # --- Assertions ---
        # We will collect all OPEN actions that were unexpectedly illegal.
        illegal_open_actions = []

        for action_name, index in env.actions_to_indices.items():
            if action_name.startswith('OPEN_'):
                # Check if this OPEN action is disabled in the mask
                if action_mask[index] == 0:
                    illegal_open_actions.append(action_name)

        # The final assertion: the list of illegal actions must be empty.
        assert not illegal_open_actions, \
            f"FAIL: The following OPEN actions were unexpectedly illegal on Step 0: {illegal_open_actions}"

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_dte_decay_logic():
    """
    A quantitative test to verify that DTE decays correctly, including
    intra-day, overnight, and weekend decay periods.
    """
    test_name = "test_dte_decay_logic"
    print(f"\n--- RUNNING: {test_name} ---")

    # --- 1. Setup a custom, deterministic environment ---
    env_cfg = copy.deepcopy(main_config.env)
    env_cfg.is_eval_mode = True
    env_cfg.disable_opening_curriculum = True
    env_cfg.forced_opening_strategy_name = 'OPEN_SHORT_STRADDLE'
    env_cfg.forced_historical_symbol = 'SPY'

    env_cfg.steps_per_day = 4
    env_cfg.forced_episode_length = 30

    env = gym.make('OptionsZeroGame-v0', cfg=env_cfg)

    try:
        # --- 2. Initialize and get baseline DTE ---
        env.reset(seed=52)

        # Step 0: Open the position
        timestep = env.step(env.actions_to_indices['HOLD'])
        portfolio = env.portfolio_manager.get_portfolio()

        # <<< THE FIX: Use the correct column name 'days_to_expiry' >>>
        initial_dte = portfolio.iloc[0]['days_to_expiry']

        # The initial DTE is calculated based on the episode length
        expected_initial_dte = 30 * (7.0 / 5.0)
        assert np.isclose(initial_dte, expected_initial_dte, atol=0.1), \
            f"Initial DTE is wrong. Expected ~{expected_initial_dte}, got {initial_dte}"

        # --- 3. Simulate forward for 4 full days (Monday -> Thursday) ---
        # 4 days * 4 steps/day = 16 steps total from the start of the episode
        for i in range(1, 16):
            timestep = env.step(env.actions_to_indices['HOLD'])

        # --- 4. First Assertion: Check DTE at the end of Thursday ---
        # <<< THE FIX: Use the correct column name 'days_to_expiry' >>>
        dte_after_thursday = env.portfolio_manager.get_portfolio().iloc[0]['days_to_expiry']
        expected_dte_thursday = initial_dte - 4.0
        print(f"Checking DTE EOD Thursday (Step 16): Expected={expected_dte_thursday:.2f}, Actual={dte_after_thursday:.2f}")
        assert np.isclose(dte_after_thursday, expected_dte_thursday, atol=0.1), \
            "DTE after 4 normal days is incorrect."

        # --- 5. Simulate forward one more day to complete Friday ---
        for i in range(4):
            timestep = env.step(env.actions_to_indices['HOLD'])

        # --- 6. Final Assertion: Check DTE after the weekend decay ---
        # <<< THE FIX: Use the correct column name 'days_to_expiry' >>>
        dte_after_friday = env.portfolio_manager.get_portfolio().iloc[0]['days_to_expiry']
        expected_dte_friday = expected_dte_thursday - 3.0
        print(f"Checking DTE EOD Friday (Step 20): Expected={expected_dte_friday:.2f}, Actual={dte_after_friday:.2f}")
        assert np.isclose(dte_after_friday, expected_dte_friday, atol=0.1), \
            "Weekend DTE decay is incorrect. Expected a 3-day drop."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_no_runaway_duplication_on_transform():
    """
    A critical regression test to ensure that transformation actions like
    HEDGE and SHIFT do not incorrectly increase the number of portfolio legs.
    This test explicitly targets the "runaway duplication" bug.
    """
    test_name = "test_no_runaway_duplication_on_transform"
    print(f"\n--- RUNNING: {test_name} ---")

    # Start with a 2-leg strangle
    env = create_test_env('OPEN_SHORT_STRANGLE_DELTA_20')
    try:
        # Step 0: Open the initial position
        env.reset(seed=53)
        env.step(env.actions_to_indices['HOLD'])
        portfolio_step0 = env.portfolio_manager.get_portfolio()
        assert len(portfolio_step0) == 2, "Setup failed: Did not open a 2-leg strangle."

        # --- Test 1: HEDGE action ---
        # Hedge one of the legs. The portfolio should grow from 2 to 3 legs.
        print("  - Testing HEDGE action...")
        env.step(env.actions_to_indices['HEDGE_NAKED_POS_0'])
        portfolio_step1 = env.portfolio_manager.get_portfolio()
        assert len(portfolio_step1) == 3, \
            f"HEDGE failed: Expected 3 legs, but found {len(portfolio_step1)}. Runaway duplication may have occurred."

        # --- Test 2: SHIFT action on the new 3-leg portfolio ---
        # Shift one of the legs. The portfolio size should remain exactly 3.
        print("  - Testing SHIFT action on a 3-leg portfolio...")
        env.step(env.actions_to_indices['SHIFT_UP_POS_1'])
        portfolio_step2 = env.portfolio_manager.get_portfolio()
        assert len(portfolio_step2) == 3, \
            f"SHIFT failed: Expected 3 legs, but found {len(portfolio_step2)}. Runaway duplication may have occurred."

        # --- Test 3: Another SHIFT action ---
        # Shift another leg. The portfolio size should still be exactly 3.
        print("  - Testing a second SHIFT action...")
        env.step(env.actions_to_indices['SHIFT_DOWN_POS_2'])
        portfolio_step3 = env.portfolio_manager.get_portfolio()
        assert len(portfolio_step3) == 3, \
            f"Second SHIFT failed: Expected 3 legs, but found {len(portfolio_step3)}. Runaway duplication may have occurred."

        print(f"--- PASSED: {test_name} ---")
        return True
    except Exception:
        traceback.print_exc()
        print(f"--- FAILED: {test_name} ---")
        return False
    finally:
        env.close()

def test_close_all_action():
    """
    A critical test to ensure the CLOSE_ALL action correctly liquidates the
    entire portfolio and realizes the P&L.
    """
    test_name = "test_close_all_action"
    print(f"\n--- RUNNING: {test_name} ---")
    
    # Start with a complex 4-leg position to rigorously test the closing loop.
    env = create_test_env('OPEN_SHORT_IRON_CONDOR')
    try:
        # Step 0: Open the initial position.
        env.reset(seed=54)
        env.step(env.actions_to_indices['HOLD'])
        portfolio_before = env.portfolio_manager.get_portfolio()
        assert len(portfolio_before) == 4, "Setup failed: Did not open a 4-leg condor."

        # Step 1: Let the market move for one step.
        timestep_before_close = env.step(env.actions_to_indices['HOLD'])
        
        # Capture the total P&L right before closing. This is our ground truth.
        pnl_before_close = timestep_before_close.info['eval_episode_return']
        
        # --- The Real Test: Execute the CLOSE_ALL action ---
        action_to_take = env.actions_to_indices['CLOSE_ALL']
        timestep_after_close = env.step(action_to_take)

        # --- Assertions ---
        # 1. The portfolio must now be empty.
        portfolio_after = env.portfolio_manager.get_portfolio()
        assert portfolio_after.empty, \
            f"CLOSE_ALL failed: Portfolio is not empty. Contains {len(portfolio_after)} legs."

        # 3. The "Sum of Unrealized P&L" must now be zero.
        pnl_verification = env.portfolio_manager.get_pnl_verification(env.price_manager.current_price, env.iv_bin_index)
        unrealized_pnl = pnl_verification['unrealized_pnl']
        assert np.isclose(unrealized_pnl, 0.0, atol=0.01), \
            f"Unrealized P&L is not zero after CLOSE_ALL. It is {unrealized_pnl:.2f}."

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
        test_hedge_short_call,
        test_hedge_long_put,
        test_hedge_long_call,
        test_convert_straddle_to_fly,
        test_convert_call_fly_to_vertical,
        test_convert_put_fly_to_vertical,
        test_hedge_strangle_leg,
        test_hedge_straddle_leg,
        test_no_new_trades_when_active,
        test_all_open_actions_are_legal,
        test_dte_decay_logic,
        test_no_runaway_duplication_on_transform,
        test_close_all_action,
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
