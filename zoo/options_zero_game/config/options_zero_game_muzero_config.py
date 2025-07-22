from easydict import EasyDict

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
action_space_size = 36 # Unchanged from last step

# ==============================================================
#                    Main Config (The Parameters)
# ==============================================================
options_zero_game_muzero_config = dict(
    exp_name=f'options_zero_game_muzero_rich_state_ns{num_simulations}_upc{update_per_collect}_bs{batch_size}',
    env=dict(
        env_id='OptionsZeroGame-v0',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        ignore_legal_actions=True,
    ),
    policy=dict(
        model=dict(
            # <<< MODIFIED: Update the observation shape to 35
            observation_shape=35,
            action_space_size=action_space_size,
            model_type='mlp',
            lstm_hidden_size=512,
            latent_state_dim=512,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        cuda=True,
        game_segment_length=30,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        optim_type='Adam',
        learning_rate=0.0005,
        weight_decay=1e-4,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=8,
        eval_freq=int(1e3),
        replay_buffer_size=int(1e5),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
    ),
)
main_config = EasyDict(options_zero_game_muzero_config)

# ==============================================================
#                  Create-Config (The Blueprint)
# ==============================================================
create_config = dict(
    env=dict(
        type='options_zero_game',
        import_names=['zoo.options_zero_game.envs'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='muzero',
        import_names=['lzero.policy.muzero'],
    ),
)
create_config = EasyDict(create_config)

if __name__ == "__main__":
    from lzero.entry import train_muzero
    train_muzero([main_config, create_config], seed=0, max_env_step=max_env_step)
