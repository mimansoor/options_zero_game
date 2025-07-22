import copy

import gym
import numpy as np
from arch import arch_model
from easydict import EasyDict
from gym import spaces
from gymnasium.utils import seeding # Use the modern gymnasium seeding utility

from ding.envs import BaseEnvTimestep
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('options_zero_game')
class OptionsZeroGameEnv(gym.Env):
    """
    Options-Zero-Game Environment for LightZero
    This class implements the core logic of the options trading simulator,
    structured to be compatible with the lzero framework.
    """
    metadata = {'render.modes': ['human']}

    config = dict(
        start_price=100.0,
        trend=0.0001,
        volatility=0.2,
        time_to_expiry=30,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None):
        self._cfg = self.default_config()
        if cfg is not None:
            self._cfg.update(cfg)

        self.start_price = self._cfg.start_price
        self.trend = self._cfg.trend
        self.volatility = self._cfg.volatility
        self.total_steps = self._cfg.time_to_expiry
        self.action_space_size = 8

        self._action_space = spaces.Discrete(self.action_space_size)
        
        self._observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space_size,), dtype=np.int8),
            'to_play': spaces.Box(low=-1, high=-1, shape=(1,), dtype=np.int8)
        })

        self._reward_range = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        
        # This will be initialized in the seed method
        self.np_random = None

        self._initialize_price_simulator()

    # ==============================================================
    # THE FINAL FIX: Implement the seed method with the correct signature
    # to match what the framework is calling.
    # ==============================================================
    def seed(self, seed: int, dynamic_seed: int = None):
        """
        Set the random seed for the environment.
        This method is called by the DI-engine framework.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _initialize_price_simulator(self):
        # We use a fixed seed here for initialization to ensure the model
        # structure is consistent, but the actual simulation will use self.np_random
        init_returns = np.random.RandomState(0).normal(loc=self.trend, scale=self.volatility / np.sqrt(252), size=1000)
        self.garch_model = arch_model(init_returns * 100, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
        self.garch_fit = self.garch_model.fit(disp='off', show_warning=False)

    def _simulate_price_step(self):
        """Simulates one step of price evolution using the GARCH model."""
        forecast = self.garch_fit.forecast(horizon=1, reindex=False)
        cond_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        # Use the environment's seeded random number generator
        shock = self.np_random.normal(loc=self.trend, scale=cond_vol)
        self.current_price *= (1 + shock)

    def _get_observation(self):
        action_mask = np.ones(self.action_space_size, dtype=np.int8)
        portfolio_delta, portfolio_gamma, portfolio_vega = 0.0, 0.0, 0.0
        obs = np.array([self.current_price, portfolio_delta, portfolio_gamma, portfolio_vega], dtype=np.float32)
        return {'observation': obs, 'action_mask': action_mask, 'to_play': np.array([-1], dtype=np.int8)}

    def _get_portfolio_value(self):
        return 0.0

    def reset(self, seed: int = None, **kwargs):
        """Resets the environment. Note: `reset` now accepts a seed argument."""
        if seed is not None:
            self.seed(seed)
            
        self.current_step = 0
        self.current_price = self.start_price
        self.portfolio = []
        self.realized_pnl = 0.0
        self._final_eval_reward = 0.0
        return self._get_observation()

    def step(self, action: int):
        assert self._action_space.contains(action), f"Invalid action: {action}"
        self._simulate_price_step()
        self.current_step += 1
        previous_portfolio_value = self._get_portfolio_value()
        current_portfolio_value = self._get_portfolio_value()
        reward = float(current_portfolio_value - previous_portfolio_value)
        done = self.current_step >= self.total_steps
        obs = self._get_observation()
        self._final_eval_reward += reward
        info = {'price': self.current_price, 'eval_episode_return': self._final_eval_reward}
        return BaseEnvTimestep(obs, reward, done, info)

    def render(self, mode='human'):
        print(f"Step: {self.current_step:02d} | Price: ${self.current_price:8.2f} | Positions: {len(self.portfolio):1d}")

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_range

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> list:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> list:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        return [cfg for _ in range(evaluator_env_num)]

    def __repr__(self):
        return "LightZero Options-Zero-Game Env"
