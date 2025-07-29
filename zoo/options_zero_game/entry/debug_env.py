import gymnasium as gym
import numpy as np
import random

# Import necessary components
from ding.envs import BaseEnvTimestep
import zoo.options_zero_game.envs.options_zero_game_env
import zoo.options_zero_game.envs.log_replay_env

def run_random_agent_test():
    """
    This function tests the OptionsZeroGameEnv in isolation from the LightZero framework.
    It runs one full episode using a random agent that only takes legal actions.
    """
    print("--- Starting Standalone Environment Test ---")
    
    try:
        from zoo.options_zero_game.config.options_zero_game_muzero_config import main_config
        env_cfg = main_config.env
        
        # Clean up the config for gym.make
        if 'manager' in env_cfg: del env_cfg.manager
        if 'collector_env_num' in env_cfg: del env_cfg.collector_env_num
        if 'evaluator_env_num' in env_cfg: del env_cfg.evaluator_env_num
        if 'n_evaluator_episode' in env_cfg: del env_cfg.n_evaluator_episode
        if 'env_id' in env_cfg: del env_cfg.env_id

        env = gym.make('OptionsZeroGame-v0', cfg=env_cfg)
        
        print(f"Environment created successfully. Version: {env.VERSION}")
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")

        print("\n--- Testing env.reset() ---")
        timestep = env.reset(seed=42)
        obs = timestep.obs 
        assert env.observation_space.contains(obs['observation']), "Initial observation is not in the defined space!"
        print("env.reset() successful. Initial observation is valid.")

        print("\n--- Running full episode with random (but legal) actions ---")
        done = False
        step_count = 0
        while not done:
            action_mask = obs['action_mask']
            legal_actions = np.where(action_mask == 1)[0]
            
            if len(legal_actions) == 0:
                action = env.actions_to_indices['HOLD']
            else:
                action = random.choice(legal_actions)

            # <<< NEW: Set a conditional breakpoint to pause only at the problematic step
            if step_count == 104:
                print(f"\n--- Pausing at Step {step_count} right before the error ---")
                import pdb; pdb.set_trace()
            
            timestep = env.step(action)
            obs, reward, done, info = timestep.obs, timestep.reward, timestep.done, timestep.info
            
            # Print the observation before the assertion
            print(f"\n--- Debugging Step {step_count} ---")
            print(f"Observation['observation'] shape: {obs['observation'].shape}")
            
            assert env.observation_space.contains(obs['observation']), f"Observation at step {step_count} is not in the defined space!"
            assert np.isfinite(reward), f"Reward at step {step_count} is not finite! Value: {reward}"
            
            env.render()
            step_count += 1
            
            if done:
                print(f"\nEpisode finished after {step_count} steps.")
                print(f"Final PnL: {info.get('eval_episode_return', 'N/A')}")

        env.close()
        print("\n--- Standalone Environment Test Passed Successfully! ---")

    except Exception as e:
        print("\n--- !!! STANDALONE ENVIRONMENT TEST FAILED !!! ---")
        print("An error occurred while testing the environment in isolation.")
        raise e

if __name__ == "__main__":
    run_random_agent_test()
