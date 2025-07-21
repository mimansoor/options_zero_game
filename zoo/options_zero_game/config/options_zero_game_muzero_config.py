import warnings
from easydict import EasyDict

# ==============================================================
# NEW: Suppress annoying warnings from dependencies
# ==============================================================
# This will hide the DeprecationWarnings and FutureWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================
#                 Options-Zero-Game Config
# ==============================================================
collector_env_num = 4
evaluator_env_num = 2
batch_size = 128
num_simulations = 50 
update_per_collect = 1000
max_env_step = 100000
reanalyze_ratio = 0.

# ==============================================================
#                    Main Config
# ==============================================================
options_zero_game_muzero_config = dict(
    exp_name=f'options_zero_game_stochastic_muzero_b{batch_size}_ns{num_simulations}',
    env=dict(
        env_id='OptionsZeroGame-v0',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        model=dict(
            observation_shape=4,
            action_space_size=8,
            model_type='mlp',
            lstm_hidden_size=512,
            latent_state_dim=512,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
            stochastic_dynamics=True,
            num_of_possible_chance_moves=5,
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=30,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='AdamW',
        lr_piecewise_constant_decay=False,
        learning_rate=0.0005,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=8,
        eval_freq=int(1e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        gumbel_algo=True,
    ),
)
options_zero_game_muzero_config = EasyDict(options_zero_game_muzero_config)
main_config = options_zero_game_muzero_config

# ==============================================================
#                  Create-Config
# ==============================================================
create_config = dict(
    env=dict(
        type='options_zero_game',
        import_names=['zoo.options_zero_game.envs'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='stochastic_muzero'
    ),
)
create_config = EasyDict(create_config)

if __name__ == '__main__':
    # ==============================================================
    # THE FIX: Use the generic 'serial_pipeline' entry point
    # ==============================================================
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0, max_env_step=max_env_step)
