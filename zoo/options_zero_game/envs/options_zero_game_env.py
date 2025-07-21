import gym
import numpy as np
from gym import spaces
from arch import arch_model

# ==============================================================
# THE FIX: Import the DI-engine registry and register our environment
# ==============================================================
from ding.utils import ENV_REGISTRY

# To use py_vollib, you might need to install it: pip install py_vollib
# from py_vollib.black_scholes import black_scholes
# from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega

@ENV_REGISTRY.register('options_zero_game') # This key must match the 'type' in your config
class OptionsZeroGameEnv(gym.Env):
    """
    Options-Zero-Game Environment for LightZero
    This class implements the core logic of the options trading simulator.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, config=None):
        super(OptionsZeroGameEnv, self).__init__()
        self.config = config if config is not None else {}

        # ===== Simulation Parameters from PRD =====
        self.start_price = self.config.get('start_price', 100.0)
        self.trend = self.config.get('trend', 0.0001)
        self.volatility = self.config.get('volatility', 0.2)
        self.total_steps = self.config.get('time_to_expiry', 30)

        # ===== Action Space Definition =====
        self.action_space = spaces.Discrete(8)
        
        # ===== Observation Space Definition =====
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        self._initialize_price_simulator()
        
        self.reset()

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
        """Constructs the observation state."""
        portfolio_delta, portfolio_gamma, portfolio_vega = self._calculate_portfolio_greeks()
        
        return np.array([
            self.current_price,
            portfolio_delta,
            portfolio_gamma,
            portfolio_vega,
        ], dtype=np.float32)

    def _calculate_portfolio_greeks(self):
        return 0.0, 0.0, 0.0

    def _get_portfolio_value(self):
        return 0.0

    def step(self, action: int):
        """Execute one time step within the environment."""
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        self._simulate_price_step()
        self.current_step += 1

        previous_portfolio_value = self._get_portfolio_value()
        current_portfolio_value = self._get_portfolio_value()
        reward = (current_portfolio_value - previous_portfolio_value)

        done = self.current_step >= self.total_steps
        obs = self._get_observation()
        info = {'price': self.current_price}

        return obs, reward, done, info

    def reset(self):
        """Reset the state of the environment to an initial state."""
        self.current_step = 0
        self.current_price = self.start_price
        self.portfolio = []
        self.realized_pnl = 0.0
        return self._get_observation()

    def render(self, mode='human'):
        """Render the environment to the console."""
        unrealized_pnl = self._get_portfolio_value()
        print(
            f"Step: {self.current_step:02d} | "
            f"Price: ${self.current_price:8.2f} | "
            f"Positions: {len(self.portfolio):1d} | "
            f"Unrealized PnL: {unrealized_pnl:8.2f} | "
            f"Realized PnL: {self.realized_pnl:8.2f}"
        )
