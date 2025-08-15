# Options-Zero-Game: A High-Fidelity Reinforcement Learning Environment for Autonomous Options Trading

**Options-Zero-Game** is a modular, and feature-rich reinforcement learning environment designed for training an autonomous options trading agent. Built within the LightZero framework, it leverages the powerful **MuZero** algorithm to learn, plan, and navigate a complex and realistic options market.

The environment is architected to be a state-of-the-art research platform, featuring a dynamic and stochastic market, a sophisticated action space with intelligent adjustments, and a rich, high-dimensional observation space augmented by a suite of pre-trained SOTA Transformer "expert" models.



---

## 🌟 Key Features

*   **Modular Architecture:** The environment's logic is cleanly separated into specialized managers for price action, portfolio management, market rules, and Black-Scholes calculations, ensuring high maintainability and extensibility.
*   **SOTA Hybrid AI System:** The MuZero agent is augmented with a suite of expert models:
    *   **Holy Duality (LightGBM):** Fast, reliable experts for predicting market trend (EMA Ratio) and momentum (RSI State).
    *   **Transformer Experts (PyTorch):**
        *   A **Volatility Analyst** that generates a rich 128-dimensional embedding of the market's volatility context.
        *   A SOTA **Hybrid GRU-Transformer** as a "Tail Risk" expert, predicting the probability of significant downward moves.
*   **Dynamic & Stochastic Market Physics:**
    *   **IV Regimes with Markov Chains:** The environment doesn't use a single IV skew. At the start of each episode, it selects a volatility regime (e.g., "High Volatility (Fear)", "Complacent") based on a stationary distribution derived from real market data, and the regime evolves from day to day according to a learned transition matrix.
    *   **Expert-Driven GARCH Simulation:** The "mixed mode" price generator is a self-consistent simulation where the Volatility Transformer's forward-looking predictions dynamically calibrate the GARCH model's volatility at each step.
*   **Sophisticated & Intelligent Action Space:** The agent possesses a powerful set of over 200 actions, including:
    *   **Dynamic Openings:** Intelligently opens vertical spreads based on a tiered search for optimal risk/reward ratios.
    *   **Risk-Based Adjustments:** A single `ADJUST_TO_DELTA_NEUTRAL` action that allows the agent to re-center its entire strategy to a market-neutral position.
    *   **Strategy Morphing:** A full suite of `CONVERT_TO_*` actions that allow the agent to dynamically transform its position's structure (e.g., Strangle → Iron Condor, Butterfly → Vertical Spread) to adapt to changing market conditions.
*   **Rich, Data-Driven Visualization:** A powerful, web-based replayer built in React that provides a deep analysis of each episode, including:
    *   A live P&L diagram with T+0 (current) and expiry curves.
    *   Standard deviation bands on the payoff chart to show the expected market range.
    *   A detailed portfolio risk profile with all Greeks, breakevens, and advanced metrics.
    *   A "Bias Meter" that synthesizes the agent's complex observation vector into a human-readable market bias.
*   **Robust Testing & Validation Suite:** Includes a full regression test suite (`regression_suite.py`) to validate all complex actions and a dedicated script (`expert_evaluator.py`) to scientifically measure the performance of all expert models on holdout data.

---

## 🏛️ Architectural Overview

The environment is designed with a clean separation of concerns, managed by a central orchestrator.

*   `OptionsZeroGameEnv`: The main orchestrator. It manages the episode flow, handles the agent's actions, and assembles the final observation vector.
*   `PriceActionManager`: The market. It generates price paths (Expert-Driven GARCH or Historical) and provides predictions from all expert models (Holy Duality & Transformers).
*   `PortfolioManager`: The heart of the trading logic. It manages the portfolio DataFrame, executes all trades (including complex transformations like `SHIFT` and `CONVERT_TO_*`), calculates P&L, and enforces advanced trading rules.
*   `MarketRulesManager`: A stateless expert on market structure. It manages the dynamic IV skew table and strike price calculations for each episode's randomly selected IV Regime.
*   `BlackScholesManager`: A stateless, Numba JIT-compiled library for all high-performance financial math (option pricing and all Greeks).

---

## 🛠️ Core Technologies

*   **RL Framework:** LightZero / DI-engine
*   **RL Algorithm:** MuZero
*   **Neural Networks:** PyTorch
*   **Environment:** Gymnasium (formerly OpenAI Gym)
*   **Data & Numerics:** Pandas, NumPy, Numba
*   **Expert Models:** LightGBM, Scikit-learn, PyTorch (Transformers)
*   **Frontend Visualizer:** React, Chart.js

---

## 🚀 Setup and Standard Workflow

Follow these steps from the project's root directory to train the agent and view the results.

**1. Setup Python Environment**
It is highly recommended to use a virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Setup Frontend UI**
```bash
cd zoo/options_zero_game/visualizer-ui/
npm install
cd ../../../.. # Return to project root
```

**3. Build the Historical Data Cache**
This script downloads ~10 years of daily data. This only needs to be run once.
```bash
python3 zoo/options_zero_game/data/cache_builder.py
```

**4. Build the IV Regime Model**
This script analyzes historical VIX data to create the Markov Chain for IV regime transitions. Run once.
```bash
python3 zoo/options_zero_game/experts/iv_regime_analyzer.py
```

**5. Train the Expert Models**
You must train the experts in the correct order.
```bash
# First, train the "Holy Duality" (LGBM models)
python3 zoo/options_zero_game/experts/holy_trinity_trainer.py

# Second, train the Volatility Transformer (required for the next step)
python3 zoo/options_zero_game/experts/transformer_expert_trainer.py --model_type volatility

# Third, train the SOTA Directional Strategist
python3 zoo/options_zero_game/experts/transformer_expert_trainer.py --model_type directional```

**6. Train the MuZero Agent**
This is the main training script. It includes automatic resumption logic.
```bash
# For a reproducible run with a fixed seed
python3 -u zoo/options_zero_game/config/options_zero_game_muzero_config.py --seed 42

# For a random seed
python3 -u zoo/options_zero_game/config/options_zero_game_muzero_config.py --seed -1
```
*Tip: For debugging `CUDA error: device-side assert triggered`, run with `CUDA_LAUNCH_BLOCKING=1 python3 ...`*

**7. Evaluate and Generate Replay Log**
After training, copy the best checkpoint (e.g., `ckpt_best.pth.tar`) to the `./best_ckpt/` directory. Then, run the evaluation script.
```bash
# Example: Run an evaluation, letting the agent choose its own move
python3 -u zoo/options_zero_game/entry/options_zero_game_eval.py --agents_choice --symbol ^NSEI
```

**8. Build and Serve the Visualizer**
First, build the static React app.
```bash
cd zoo/options_zero_game/visualizer-ui/
npm run build
cd ../../../.. # Return to project root
```
Then, run the Python server to view the results.
```bash
python3 zoo/options_zero_game/serve.py
```
Now, open your web browser to `http://<your-ip-address>:5001` to view the replay.

---

## 🔬 Advanced Usage & Analysis

The project includes a powerful suite of scripts for analysis and testing.

*   **Strategy Analyzer:** Perform a deep statistical backtest on the agent's performance with specific opening strategies.
    ```bash
    # Run a full comparative analysis of ALL opening strategies (10 episodes each)
    python3 zoo/options_zero_game/entry/strategy_analyzer.py --strategy ALL -n 10
    ```
*   **Expert Evaluator:** Scientifically measure the predictive power of your trained expert models on unseen holdout data.
    ```bash
    # Evaluate the Transformer experts
    python3 zoo/options_zero_game/experts/expert_evaluator.py --expert_type transformer
    ```
*   **Regression Test Suite:** Run a full suite of tests to validate the environment's core mechanics and prevent regressions.
    ```bash
    python3 zoo/options_zero_game/entry/regression_suite.py
    ```

---

## 📚 Future Work

This project provides a strong foundation for future research, including:
*   **Stochastic MuZero:** Upgrading the agent to use the Stochastic MuZero algorithm to better handle the probabilistic nature of the market.
*   **Hierarchical Reinforcement Learning (HRL):** Implementing a two-level agent where a "Master" agent chooses the overall strategy and a "Slave" agent manages the execution.
*   **Advanced Architectures:** Experimenting with cutting-edge, long-sequence models like Informer or Mamba for the expert system.
*   **Continuous Control:** Adapting the environment for continuous control agents that can choose the precise quantity or strike price.
