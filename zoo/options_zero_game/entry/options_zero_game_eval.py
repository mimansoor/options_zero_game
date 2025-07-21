import numpy as np

# Use the dedicated evaluation entry point
from lzero.entry import eval_muzero
# Import OUR game's configuration
from zoo.options_zero_game.config.options_zero_game_stochastic_muzero_config import main_config, create_config

if __name__ == "__main__":
    """
    Entry point for evaluating a trained agent on the Options-Zero-Game.
    """

    # Set model_path to your trained model's .pth.tar file, or None to use a random agent.
    # e.g., './exp/options_zero_game.../ckpt/ckpt_best.pth.tar'
    model_path = None

    returns_mean_seeds = []
    returns_seeds = []
    seeds = [0] # You can add more seeds for more robust evaluation, e.g., [0, 1, 2]
    num_episodes_each_seed = 3 # Evaluate for 3 episodes

    # To visualize the agent's actions, we can override the render_mode.
    # For our text-based environment, 'state_realtime_mode' is a good choice.
    # NOTE: You would need to implement the render() method in the environment first.
    # main_config.env.render_mode = 'state_realtime_mode' 
    
    # Set up evaluation-specific settings
    total_test_episodes = num_episodes_each_seed * len(seeds)
    create_config.env_manager.type = 'base'  # Visualization requires 'base' manager
    main_config.env.evaluator_env_num = 1   # And only 1 evaluator environment
    main_config.env.n_evaluator_episode = total_test_episodes
    
    for seed in seeds:
        returns_mean, returns = eval_muzero(
            [main_config, create_config],
            seed=seed,
            num_episodes_each_seed=num_episodes_each_seed,
            print_seed_details=True,
            model_path=model_path
        )
        returns_mean_seeds.append(returns_mean)
        returns_seeds.append(returns)

    returns_mean_seeds = np.array(returns_mean_seeds)
    returns_seeds = np.array(returns_seeds)

    print("=" * 20)
    print(f'We eval a total of {len(seeds)} seeds. In each seed, we eval {num_episodes_each_seed} episodes.')
    print(f'For seeds {seeds}, the mean returns were: {returns_mean_seeds}')
    print(f'The returns for each individual episode were: {returns_seeds}')
    print(f'Across all seeds, the grand mean of returns is: {returns_mean_seeds.mean():.2f}')
    print("=" * 20)
