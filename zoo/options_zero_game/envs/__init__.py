from gym.envs.registration import register
from .options_zero_game_env import OptionsZeroGameEnv

# Register the environment so it can be created by gym.make()
register(
    id='OptionsZeroGame-v0',
    entry_point='zoo.options_zero_game.envs.options_zero_game_env:OptionsZeroGameEnv',
)
