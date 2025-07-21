import copy

import gym
import numpy as np
from arch import arch_model
from easydict import EasyDict
from gym import spaces

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

    # Class-level config dictionary holds the default parameters for the environment.
    config = dict(
        start_price=100.0,
        trend=0.0001,
        volatility=0.2,
        time_to_expiry=30,
    )

    @classmethod
    def default_config(cls) -> EasyDict:
        """
        Returns the default configuration for the environment.
        This is a required method for the lzero framework.
        """
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: dict = None):
        """
        Initializes the environment.
        """
        # Use the provided config, or fall back to the default.
        self._cfg = self.default_config()
        if cfg is not None:
            self._cfg.update(cfg)

        # Access parameters from self._cfg
        self.start_price = self._cfg.start_price
        self.trend = self._cfg.trend
        self.volatility = self._cfg.volatility
        self.total_steps = self._cfg.time_to_expiry
        self.action_space_size = 8

        # Define action space
        self._action_space = spaces.Discrete(self.action_space_size)

        # Define observation space as a dictionary (required by lzero policies)
        self._observation_space = spaces.Dict({
            'observation': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=(self.action_space_size,), dtype=np.int8),
            'to_play': spaces.Box(low=-1, high=-1, shape=(1,), dtype=np.int8)
        })

        # Define the reward space (required by DI-engine's BaseEnvManager)
        self._reward_range = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        # Initialize the GARCH price simulator
        self._initialize_price_simulator()

    def _initialize_price_simulator(self):
        """Initializes or re-initializes the GARCH(1,1) model."""
        returns = np.random.normal(loc=self.trend, scale=self.volatility / np.sqrt(252), size=1000)
        self.garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
        self.garch_fit = self.garch_model.fit(disp='off', show_warning=False)

    def _simulate_price_step(self):
        """Simulates one step of price evolution using the GARCH model."""
        forecast = self.garch_fit.forecast(horizon=1, reindex=False)
        cond_vol = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        shock = np.random.normal(loc=self.trend, scale=cond_vol)
        self.current_price *= (1 + shock)

    def _get_observation(self):
        """Constructs the observation dictionary in the required format."""
        # TODO: Implement real action mask logic based on PRD rules
        action_mask = np.ones(self.action_space_size, dtype=np.int8)

        # TODO: Implement real portfolio greeks
        portfolio_delta, portfolio_gamma, portfolio_vega = 0.0, 0.0, 0.0

        obs = np.array([
            self.current_price,
            portfolio_delta,
            portfolio_gamma,
            portfolio_vega,
        ], dtype=np.float32)

        return {'observation': obs, 'action_mask': action_mask, 'to_play': np.array([-1], dtype=np.int8)}

    def _get_portfolio_value(self):
        """Placeholder for portfolio valuation logic."""
        # TODO: Implement this logic when we handle actions.
        return 0.0

    def reset(self):
        """Resets the environment to an initial state."""
        self.current_step = 0
        self.current_price = self.start_price
        self.portfolio = []
        self.realized_pnl = 0.0
        self._final_eval_reward = 0.0  # Used for final evaluation metric
        return self._get_observation()

    def step(self, action: int):
        """Executes one time step within the environment."""
        assert self._action_space.contains(action), f"Invalid action: {action}"

        self._simulate_price_step()
        self.current_step += 1

        # TODO: Implement action logic here to modify the portfolio
        previous_portfolio_value = self._get_portfolio_value()
        current_portfolio_value = self._get_portfolio_value()

        reward = float(current_portfolio_value - previous_portfolio_value)
        done = self.current_step >= self.total_steps

        obs = self._get_observation()
        self._final_eval_reward += reward
        info = {'price': self.current_price, 'eval_episode_return': self._final_eval_reward}

        # Return the DI-engine standard timestep object
        return BaseEnvTimestep(obs, reward, done, info)

    def render(self, mode='human'):
        """Renders the environment to the console."""
        print(
            f"Step: {self.current_step:02d} | "
            f"Price: ${self.current_price:8.2f} | "
            f"Positions: {len(self.portfolio):1d}"
        )

    # ==============================================================
    # Required properties for lzero framework compatibility
    # ==============================================================
    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_range

    # ==============================================================
    # Required static methods for creating env instances
    # ==============================================================
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
