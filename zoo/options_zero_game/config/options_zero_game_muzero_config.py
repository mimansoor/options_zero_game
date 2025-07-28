from easydict import EasyDict
import copy
import random

# ==============================================================
#                 Options-Zero-Game Config
# ==============================================================
collector_env_num = 4
evaluator_env_num = 2
batch_size = 128
num_simulations = 25
update_per_collect = 200
max_env_step = int(5e6)
reanalyze_ratio = 0.
action_space_size = 42
n_episode = 8

# <<< MODIFIED: The final, correct observation shape
# 5 (global) + 4 slots * 9 features/slot + 42 (action mask) = 5 + 36 + 42 = 83
observation_shape = 83

market_regimes = [
    {'name': 'Developed_Markets', 'mu': 0.00006, 'omega': 0.000005, 'alpha': 0.08, 'beta': 0.90},
    {'name': 'Emerging_Markets', 'mu': 0.0001, 'omega': 0.00005, 'alpha': 0.12, 'beta': 0.86},
    {'name': 'Individual_Stocks', 'mu': 0.00008, 'omega': 0.00007, 'alpha': 0.10, 'beta': 0.88},
    {'name': 'Commodities_Oil', 'mu': 0.0000, 'omega': 0.0002, 'alpha': 0.28, 'beta': 0.70},
    {'name': 'Foreign_Exchange_FX', 'mu': 0.0000, 'omega': 0.000003, 'alpha': 0.08, 'beta': 0.91},
    {'name': 'Cryptocurrencies', 'mu': 0.001, 'omega': 0.001, 'alpha': 0.20, 'beta': 0.75},
    {'name': 'Bond_Markets', 'mu': 0.00001, 'omega': 0.000002, 'alpha': 0.05, 'beta': 0.92},
    {'name': 'Volatility_VIX', 'mu': 0.0, 'omega': 0.0005, 'alpha': 0.25, 'beta': 0.65},
    {'name': 'Frontier_Markets', 'mu': 0.0002, 'omega': 0.0001, 'alpha': 0.20, 'beta': 0.70},
    {'name': 'Tech_Sector', 'mu': 0.0001, 'omega': 0.00004, 'alpha': 0.11, 'beta': 0.87},
    {'name': 'Utilities_Sector', 'mu': 0.00004, 'omega': 0.00001, 'alpha': 0.08, 'beta': 0.90},
    {'name': 'Crisis_HighVol', 'mu': -0.0005, 'omega': 0.0001, 'alpha': 0.15, 'beta': 0.82},
    {'name': 'Bull_LowVol', 'mu': 0.0005, 'omega': 0.000002, 'alpha': 0.05, 'beta': 0.94},
]

# ==============================================================
#                    Main Config (The Parameters)
# ==============================================================
options_zero_game_muzero_config = dict(
    exp_name=f'options_zero_game_muzero_final_agent_ns{num_simulations}_upc{update_per_collect}_bs{batch_size}',
    env=dict(
        env_id='OptionsZeroGame-v0',
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        drawdown_penalty_weight=0.1,
        market_regimes=market_regimes,
        illegal_action_penalty=-1.0,
        rolling_vol_window=5,
        steps_per_day=75,
    ),
    policy=dict(
        model=dict(
            observation_shape=observation_shape,
            action_space_size=action_space_size,
            model_type='mlp',
            lstm_hidden_size=512,
            latent_state_dim=512,
            self_supervised_learning_loss=True,
            discrete_action_encoding_type='one_hot',
            norm_type='BN',
        ),
        model_path = None, # Set to None for a fresh training run
        cuda=True,
        game_segment_length=30,
        update_per_collect=update_per_collect,
        batch_size=batch_size,
        td_steps=10,
        discount_factor=0.999,
        manual_temperature_decay=True,
        threshold_training_steps_for_final_temperature=int(1e5),
        piecewise_decay_lr_scheduler=False,
        optim_type='Adam',
        learning_rate=3e-3,
        weight_decay=1e-4,
        num_simulations=num_simulations,
        reanalyze_ratio=reanalyze_ratio,
        n_episode=n_episode,
        eval_freq=int(2e3),
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
