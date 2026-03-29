# Project README: Reinforcement Learning for Asset Allocation

This project implements and evaluates several reinforcement learning (RL) agents for solving dynamic asset allocation problems. The codebase is structured to separate the core RL infrastructure, the agent models, and the experimental test harness.

## Quick Start

Users can run `PolicyGradient.ipynb` cell by cell to see the results of each model independently. And it is also possible to run `asset_allocation_unittest.py` directly to see the detailed test results.

## Project Structure Overview

The project is organized into two main components:

1. **`asset_allocation/`**: A Python package that contains all the reusable, core components of the RL system. This is the project's "engine."
2. **`asset_allocation_unittest.py`**: An experiment runner script that uses the `asset_allocation` package to conduct various "tests" or experiments.

```text
HKUST_5370/
|-- asset_allocation/          # Core RL Package (Infrastructure & Models)
|   |-- __init__.py
|   |-- common.py              # Infrastructure: Base environment, utilities, plotting
|   |-- experiment_envs.py     # Infrastructure: Specialized environments for tests
|   |-- method1_policy_gradient.py # Model 1: Policy Gradient (REINFORCE)
|   |-- method2_actor_critic.py    # Model 2: Actor-Critic (Linear Policy)
|   |-- method3_neural_actor_critic.py # Model 3: Actor-Critic (Neural Policy)
|   |-- method4_neural_ac_adjustable_lr.py # Model 4: Actor-Critic (Neural Policy with Adjustable LR)
|-- asset_allocation_unittest.py # Experiment Runner (Tests)
|-- PolicyGradient.ipynb         # Notebook for experimentation (to be refactored)
|-- optional/                    # Legacy or unused files
```

---

## 1. Core RL Package: `asset_allocation/`

This package contains the foundational building blocks (infrastructure) and the RL agents (models).

### Infrastructure (基建)

These modules provide the underlying framework needed to define and run asset allocation experiments.

- **`asset_allocation/common.py`**: This is the primary infrastructure file.

  - **Base Environment**: Defines the `AssetAllocationEnv`, which simulates a static market. This is the foundational world where agents learn.
  - **Portfolio Mechanics**: Includes critical functions for portfolio management, such as `project_to_simplex` (ensuring weights sum to 1) and `apply_turnover_toward_target` (enforcing trading constraints).
  - **Utility & Plotting**: Contains the `cara_utility` function to measure performance and shared plotting functions (`plot_training_history`, `plot_eval_wealth`) to visualize results.
- **`asset_allocation/experiment_envs.py`**: This file extends the infrastructure with more complex, non-static environments used for specific tests.

  - `SingleAdvantageEnv`: An environment where one asset is consistently better, testing if the agent can identify and exploit it.
  - `TwoStateRotationEnv`: An environment with a hidden market regime that switches, testing the agent's ability to adapt to changing conditions.

### Models (模型)

These modules contain the implementations of the different RL agents (or "models") that learn how to allocate assets.

- **`asset_allocation/method1_policy_gradient.py`**:

  - **Algorithm**: Implements a **Policy Gradient (REINFORCE)** agent.
  - **Policy**: Uses a simple `LinearGaussianPolicy`.
  - **Features**: A key feature is its support for different baseline functions (`none`, `linear`, `neural`) to reduce variance during training. This allows for comparing how different variance reduction techniques affect a classic REINFORCE agent.
- **`asset_allocation/method2_actor_critic.py`**:

  - **Algorithm**: Implements a classic **Actor-Critic** agent.
  - **Policy (Actor)**: Uses a `LinearGaussianPolicy`.
  - **Value Function (Critic)**: Can be configured with `none`, `linear`, or `neural` critics.
  - **Purpose**: Serves as a bridge between simple policy gradient and more advanced neural network-based actor-critics.
- **`asset_allocation/method3_neural_actor_critic.py`**:

  - **Algorithm**: Implements a more advanced **Actor-Critic** agent.
  - **Policy (Actor)**: Uses a `NeuralGaussianPolicyB`, a neural network that outputs portfolio weights.
  - **Value Function (Critic)**: Uses a `NeuralValueCritic`, a separate neural network to estimate state value.
  - **Features**: This is the most sophisticated model, designed to handle complex dynamics and serve as the primary agent for the experiments in the `unittest` script.
- **`asset_allocation/method4_neural_ac_adjustable_lr.py`**:

  - **Algorithm**: An extension of `method3`.
  - **Features**: This version adds a **learning rate scheduler**. The learning rate starts at an `initial_lr` and gradually decays to a `final_lr` over a set number of `lr_decay_steps`. This can help the model converge more effectively by taking large steps early in training and smaller, more precise steps later on.

**Model Improvement Path:**

![1774793875734](image/README/1774793875734.png)

---

## 2. Experiment Runner: `asset_allocation_unittest.py`

This script is not a "unit test" in the traditional software engineering sense, but rather a **test harness for running scientific experiments**. It imports the infrastructure and models from the `asset_allocation` package to evaluate agent performance under different scenarios.

The script is controlled by a `mode` variable, which determines which experiment to run.

### Tests / Experiments Conducted

- **`mode = 'static'`**:

  - **Test**: Evaluates the neural actor-critic (`method3`) in the basic `AssetAllocationEnv`.
  - **Purpose**: To establish a baseline performance in a simple, non-changing market.
- **`mode = 'single_advantage'`**:

  - **Test**: Evaluates the agent in the `SingleAdvantageEnv`.
  - **Purpose**: To test if the agent can learn to consistently overweight the asset with a higher expected return.
- **`mode = 'two_state'`**:

  - **Test**: Evaluates the agent in the `TwoStateRotationEnv`.
  - **Purpose**: To test the agent's adaptability. Can it change its strategy when the underlying market regime shifts?
- **`mode = 'turnover_scan'`**:

  - **Test**: Runs the `two_state` experiment multiple times, each with a different `turnover_limit` (trading friction).
  - **Purpose**: To analyze the trade-off between performance and trading costs. It generates a plot showing how the agent's final wealth is affected by transaction constraints.

By separating the core logic into the `asset_allocation` package and the experiment orchestration into this script, the project allows for clean, readable, and reproducible RL research.
