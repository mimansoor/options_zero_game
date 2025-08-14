# Options-Zero-Game: A Reinforcement Learning Environment for Autonomous Options Trading

**Options-Zero-Game** is a high-fidelity, modular, and feature-rich reinforcement learning environment designed for training an autonomous options trading agent. Built within the **LightZero** framework, it leverages the powerful **MuZero** algorithm to navigate a complex and realistic options market.

The environment is designed from the ground up to be a sophisticated training ground, featuring a dynamic action space, advanced risk management rules, and a rich, high-dimensional observation space augmented by pre-trained classical machine learning models.

## 🌟 Key Features

*   **Modular Architecture:** The environment logic is cleanly separated into specialized managers for price action, portfolio management, market rules, and Black-Scholes calculations, making it highly maintainable and extensible.
*   **Hybrid AI System:** The MuZero agent's observation space is augmented with predictions from a suite of pre-trained **LightGBM "expert" models**, providing it with forecasts for market trend (EMA Ratio), momentum (RSI State), and volatility (Historical Volatility).
*   **Advanced Trading Rules:** The environment's "physics" include a suite of professional-grade trading rules to guide the agent, including:
    *   A "must-open" curriculum on Day 1 to ensure exploration.
    *   Dynamic take-profit rules (e.g., 25% of max profit for credit spreads, 2x debit for debit spreads).
    *   A portfolio-level "home run" profit target (e.g., 6%).
    *   A configurable stop-loss based on a multiple of the initial trade cost.
    *   A "liquidation period" that forces the agent to exit positions in the final two days before expiry.
    *   Strict, configurable delta and strike offset thresholds for opening naked positions.
*   **Dynamic & Sophisticated Action Space:** The agent has a large, discrete action space (over 200 actions) including dynamic, market-aware strategies like:
    *   Opening strangles based on a range of target deltas (15, 20, 25, 30).
    *   Opening Iron Condors and Iron Flies with intelligent, tiered fallbacks based on live market data.
*   **Rich, Data-Driven Visualization:** A powerful, web-based replayer built in **React** that provides a deep analysis of each episode, including:
    *   A live P&L diagram with T+0 (current) and expiry curves.
    *   Standard deviation bands on the payoff chart to show expected market range.
    *   A detailed portfolio risk profile with all Greeks and summary statistics (Max Profit, R:R Ratio, POP).
    *   A log of all closed trades with realized P&L.
    *   A data-driven "Bias Meter" that synthesizes the agent's complex observation vector into a human-readable market bias.

## 🚀 The Visualizer in Action

The evaluation script generates a rich replay log that can be viewed in the included React-based visualizer, providing deep insight into the agent's decision-making process.

*(Recommendation: Add a GIF or a final screenshot of your visualizer here)*

## 🛠️ Core Technologies

*   **RL Framework:** LightZero / DI-engine
*   **RL Algorithm:** MuZero
*   **Neural Networks:** PyTorch
*   **Environment:** Gymnasium (formerly OpenAI Gym)
*   **Data & Numerics:** Pandas, NumPy, Numba (for JIT-compiled Black-Scholes)
*   **Expert Models:** LightGBM, Scikit-learn, `pandas-ta`
*   **Frontend Visualizer:** React, Chart.js

## 🏛️ Architectural Overview

The environment is designed with a clean, modular architecture to separate concerns:

*   **`OptionsZeroGameEnv`:** The main orchestrator. It manages the episode flow, handles the agent's actions, and assembles the final observation vector.
*   **`PriceActionManager`:** Manages the market. It generates price paths (GARCH or Historical) and provides predictions from the pre-trained "Holy Trinity" expert models.
*   **`PortfolioManager`:** The heart of the trading logic. It manages the portfolio DataFrame, executes all trades, calculates P&L, and enforces advanced trading rules like stop-losses and take-profits.
*   **`MarketRulesManager`:** A stateless expert on market structure. It manages the IV skew table and strike price calculations.
*   **`BlackScholesManager`:** A stateless, JIT-compiled library for all financial math (option pricing and all Greeks).

## ⚙️ Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd options-zero-game
    ```

2.  **Setup Python Environment:** It is highly recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install lightgbm scikit-learn joblib pandas-ta
    ```

3.  **Setup Frontend UI:**
    ```bash
    cd zoo/options_zero_game/visualizer-ui/
    npm install
    cd ../../../.. # Return to project root
    ```

## 📈 Standard User Workflow

Follow these steps in order from the project's root directory to train the agent and view the results.

**Step 1: Build the Historical Data Cache**
This script downloads ~10 years of daily data for a predefined list of symbols. This only needs to be run once.
```bash
python3 zoo/options_zero_game/data/cache_builder.py
```

**Step 2: Train the "Holy Trinity" Expert Models**
This script uses the cached historical data to train the LightGBM models that predict trend, momentum, and volatility. This only needs to be run once (or whenever you update your historical data).
```bash
python3 zoo/options_zero_game/experts/holy_trinity_trainer.py
```
**Step 3: Train the MuZero Agent**
This is the main training script. It will run for max_env_step (default: 50 million steps) and save the best model checkpoint in its experiment directory.
```bash
# For a reproducible run with a fixed seed
python3 -u zoo/options_zero_game/config/options_zero_game_muzero_config.py --seed 42

# For a run with a random seed
python3 -u zoo/options_zero_game/config/options_zero_game_muzero_config.py --seed -1
```
**Step 4: Evaluate and Generate Replay Log**
After training is complete, copy the best checkpoint (ckpt_best.pth.tar) to your project's root best_ckpt/ directory. Then, run the evaluation script to generate the replay_log.json file for the visualizer.
```bash
# Example: Run an evaluation on the ^NSEI symbol for 21 days
python3 -u zoo/options_zero_game/entry/options_zero_game_eval.py --symbol ^NSEI --days 21
```
**Step 5: Build and Serve the Visualizer**
First, build the static React app. This compiles the frontend code.
```bash
cd zoo/options_zero_game/visualizer-ui/
npm run build
cd ../../../.. # Return to project root
```
Then, run the simple Python server to view the results in your browser.
```bsh
python3 zoo/options_zero_game/serve.py
```
Now, open your web browser and navigate to ```http://<your-ip-address>:5001``` to view the replay.

## 🔬 Advanced Analysis

The project includes a powerful script for performing detailed backtesting and comparative analysis of the agent's learned strategies.

**`strategy_analyzer.py`**
This script runs a specified number of episodes for one or all opening strategies and computes detailed performance statistics, allowing you to build a complete profile of your agent's strengths and weaknesses.

**Usage Examples:**

*   **Analyze a single strategy for 100 episodes:**
    ```bash
    python3 zoo/options_zero_game/entry/strategy_analyzer.py --strategy OPEN_SHORT_IRON_CONDOR -n 100
    ```
*   **Run a full comparative analysis of ALL opening strategies (10 episodes each):**
    ```bash
    python3 zoo/options_zero_game/entry/strategy_analyzer.py --strategy ALL -n 10
    ```
*   **Analyze a strategy against a specific historical market:**
    ```bash
    python3 zoo/options_zero_game/entry/strategy_analyzer.py --strategy OPEN_LONG_CALL_ATM --symbol SPY -n 50
    ```

## 📚 Future Work

This project provides a strong foundation for future research and enhancements, including:

*   **Stochastic Environment:** Modifying the environment to generate price movements on-the-fly within the `step` method to create a true Stochastic MDP, which is a more faithful representation of market dynamics.
*   **Stochastic MuZero:** Upgrading the agent to use the Stochastic MuZero algorithm to better handle the probabilistic nature of the market and learn more robust risk management policies.
*   **Hierarchical Reinforcement Learning (HRL):** Implementing a two-level agent where a "Master" agent chooses the overall strategy (e.g., "sell volatility") and specialized "Slave" agents manage the execution and fine-grained adjustments of that strategy.
*   **Expanded Action Space:** Adding more complex options strategies like Calendars, Diagonals, or Ratio Spreads to the agent's toolkit.
*   **Continuous Control:** Adapting the environment for continuous control agents that can choose not just the strategy, but the precise quantity or strike price.

## 📄 License

This project is licensed under the MIT License.
