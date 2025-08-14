# zoo/options_zero_game/entry/debug_env.py
# <<< The Definitive Manual Runner and Debugging Script >>>

import gymnasium as gym
import numpy as np
import random
import copy

from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config
# We need to explicitly import the env to register it with gymnasium
import zoo.options_zero_game.envs.options_zero_game_env

def run_manual_test():
    """
    This function tests the OptionsZeroGameEnv in complete isolation.
    It runs one full episode using a random agent that only takes legal actions.
    If any error occurs inside the env, this script will crash and show the true traceback.
    """
    print("--- Starting Standalone Manual Environment Test ---")
    
    try:
        # 1. Use a deep copy of the main config
        env_cfg = copy.deepcopy(main_config.env)
        
        # 2. Set it to evaluation mode to test the correct logic
        env_cfg.is_eval_mode = True
        # Let's also test the agent's choice curriculum
        env_cfg.disable_opening_curriculum = True
        
        # 3. Create the environment directly using gym.make
        # This bypasses all the complex framework wrappers.
        env = gym.make('OptionsZeroGame-v0', cfg=env_cfg)
        
        print(f"Environment created successfully. Version: {env.VERSION}")
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")

        print("\n--- Testing env.reset() ---")
        obs_dict = env.reset(seed=32)
        # The observation from the raw env is a dict
        obs = obs_dict['observation']
        action_mask = obs_dict['action_mask']
        
        assert env.observation_space.contains(obs), "Initial observation is not in the defined space!"
        print("env.reset() successful. Initial observation is valid.")

        print("\n--- Running full episode with random (but legal) actions ---")
        done = False
        step_count = 0
        total_reward = 0.0
        
        while not done:
            # The action mask is now part of the observation from the raw env
            legal_actions = np.where(action_mask == 1)[0]

            # Failsafe if for some reason no actions are legal
            if len(legal_actions) == 0:
                print("Warning: No legal actions available. Forcing HOLD.")
                action = env.actions_to_indices['HOLD']
            else:
                action = random.choice(legal_actions)

            action_name = env.indices_to_actions.get(action, "INVALID_ACTION")
            print(f"\n--- Step {step_count+1}: Taking Action: {action_name} (Index: {action}) ---")

            # The raw env.step() returns a BaseEnvTimestep object
            timestep = env.step(action)
            
            # Unpack the timestep for the next loop
            obs = timestep.obs['observation']
            action_mask = timestep.obs['action_mask']
            reward = timestep.reward
            done = timestep.done
            info = timestep.info
            
            total_reward += reward

            print(f"  - Reward: {reward:.4f}")
            print(f"  - Total PnL: ${info.get('eval_episode_return', 'N/A'):.2f}")
            
            assert np.isfinite(reward), f"Reward at step {step_count} is not finite! Value: {reward}"
            env.render() # This will print the bias meter summary
            
            step_count += 1

            if done:
                print(f"\nEpisode finished after {step_count} steps.")
                print(f"Final PnL: ${info.get('eval_episode_return', 'N/A'):.2f}")
                print(f"Total Shaped Reward: {total_reward:.4f}")

        env.close()
        print("\n--- Standalone Manual Test Passed Successfully! ---")

    except Exception as e:
        print("\n--- !!! STANDALONE MANUAL TEST FAILED !!! ---")
        print("The environment crashed. The traceback below is the REAL error:")
        # Re-raise the exception to get the full traceback
        raise e

if __name__ == "__main__":
    run_manual_test()
