from gymnasium.envs.registration import register

# This is the standard way to register a custom environment with Gymnasium.
# It tells gymnasium.make() where to find our class.
register(
    id='OptionsZeroGame-v0',
    entry_point='zoo.options_zero_game.envs.options_zero_game_env:OptionsZeroGameEnv',
)
