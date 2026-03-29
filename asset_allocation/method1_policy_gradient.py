import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .common import AssetAllocationEnv, build_static_env_config, valid_portfolio_update


class LinearGaussianPolicy:
    def __init__(self, state_dim, action_dim, init_std=0.05, seed=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.W = 0.01 * self.rng.normal(size=(action_dim, state_dim))
        self.b = np.zeros(action_dim)
        self.log_std = np.log(init_std) * np.ones(action_dim)

    def forward(self, state):
        mu = self.W @ state + self.b
        std = np.exp(self.log_std)
        return mu, std

    def sample_action(self, state):
        mu, std = self.forward(state)
        raw_action = mu + std * self.rng.normal(size=self.action_dim)
        z_value = (raw_action - mu) / std
        log_prob = -0.5 * np.sum(z_value ** 2 + 2 * np.log(std) + np.log(2 * np.pi))
        return raw_action, log_prob, mu, std

    def grad_log_prob(self, state, raw_action):
        mu, std = self.forward(state)
        diff = (raw_action - mu) / (std ** 2)
        grad_W = np.outer(diff, state)
        grad_b = diff
        return grad_W, grad_b

    def param_norm(self):
        return np.sqrt(np.sum(self.W ** 2) + np.sum(self.b ** 2))


class NoBaseline:
    @property
    def name(self):
        return "none"

    def predict(self, state):
        return 0.0

    def fit_batch(self, states, returns, **kwargs):
        return 0.0

    def param_norm(self):
        return 0.0


class LinearValueBaseline:
    def __init__(self, state_dim, seed=None):
        self.rng = np.random.default_rng(seed)
        self.w = np.zeros(state_dim)

    @property
    def name(self):
        return "linear"

    def predict(self, state):
        return float(np.dot(self.w, state))

    def fit_batch(self, states, returns, lr=1e-2, epochs=20, **kwargs):
        states = np.array(states)
        returns = np.array(returns)
        last_loss = 0.0
        for _ in range(epochs):
            preds = states @ self.w
            err = preds - returns
            grad = states.T @ err / len(states)
            self.w -= lr * grad
            last_loss = np.mean(err ** 2)
        return float(last_loss)

    def param_norm(self):
        return float(np.linalg.norm(self.w))


class NeuralValueBaseline(nn.Module):
    def __init__(self, state_dim, hidden_dim=32, lr=1e-3, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    @property
    def name(self):
        return "neural"

    def forward(self, states):
        return self.net(states).squeeze(-1)

    def predict(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value = self.forward(state_t).item()
        return float(value)

    def fit_batch(self, states, returns, epochs=20, **kwargs):
        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        returns_t = torch.tensor(np.array(returns), dtype=torch.float32)

        last_loss = 0.0
        for _ in range(epochs):
            preds = self.forward(states_t)
            loss = self.loss_fn(preds, returns_t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            last_loss = loss.item()
        return float(last_loss)

    def param_norm(self):
        total = 0.0
        with torch.no_grad():
            for parameter in self.parameters():
                total += torch.sum(parameter ** 2).item()
        return float(np.sqrt(total))


def make_baseline(baseline_type, state_dim, baseline_lr=1e-2, hidden_dim=32, seed=None):
    baseline_type = baseline_type.lower()
    if baseline_type == "none":
        return NoBaseline()
    if baseline_type == "linear":
        return LinearValueBaseline(state_dim=state_dim, seed=seed)
    if baseline_type in ["neural", "nn", "mlp"]:
        return NeuralValueBaseline(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            lr=baseline_lr,
            seed=seed,
        )
    raise ValueError(f"Unknown baseline_type: {baseline_type}")


def run_episode(env, policy, max_turnover=0.10):
    states = []
    raw_actions = []
    rewards = []
    infos = []

    state = env.reset()
    done = False
    while not done:
        raw_action, log_prob, mu, std = policy.sample_action(state)
        old_weights = env.weights.copy()
        new_weights = valid_portfolio_update(old_weights, raw_action, max_turnover=max_turnover)
        next_state, reward, done, info = env.step(new_weights)

        states.append(state)
        raw_actions.append(raw_action)
        rewards.append(reward)
        infos.append({
            "old_w": old_weights,
            "new_w": new_weights,
            "mu": mu,
            "std": std,
            "log_prob": log_prob,
            **info,
        })
        state = next_state

    episode_return = sum(rewards)
    terminal_wealth = infos[-1]["wealth"]
    return states, raw_actions, rewards, infos, episode_return, terminal_wealth


def evaluate_policy(env_config, policy, n_eval=200, max_turnover=0.10, seed=999):
    utilities = []
    wealths = []

    for path_index in range(n_eval):
        env = AssetAllocationEnv(
            means=env_config["means"],
            variances=env_config["variances"],
            cash_rate=env_config["cash_rate"],
            horizon=env_config["horizon"],
            init_weights=env_config["init_weights"],
            init_wealth=env_config["init_wealth"],
            risk_aversion=env_config["risk_aversion"],
            seed=seed + path_index,
        )

        state = env.reset()
        done = False
        while not done:
            mu, _ = policy.forward(state)
            action_weights = valid_portfolio_update(env.weights.copy(), mu, max_turnover=max_turnover)
            state, reward, done, info = env.step(action_weights)

        utilities.append(reward)
        wealths.append(info["wealth"])

    return np.mean(utilities), np.std(utilities), np.mean(wealths), np.std(wealths)


def train_policy_gradient(
    env,
    episodes=5000,
    policy_lr=5e-3,
    baseline_type="linear",
    baseline_lr=1e-2,
    baseline_epochs=5,
    baseline_hidden_dim=32,
    max_turnover=0.10,
    eval_every=100,
    eval_paths=200,
    seed=None,
):
    state_dim = 2 + env.n_assets_total
    action_dim = env.n_assets_total

    policy = LinearGaussianPolicy(state_dim, action_dim, init_std=0.05, seed=seed)
    baseline = make_baseline(
        baseline_type=baseline_type,
        state_dim=state_dim,
        baseline_lr=baseline_lr,
        hidden_dim=baseline_hidden_dim,
        seed=seed,
    )
    env_config = build_static_env_config(env)

    history = {
        "value_type": baseline_type,
        "train_utility": [],
        "train_terminal_wealth": [],
        "policy_norm": [],
        "value_norm": [],
        "value_loss": [],
        "eval_episode": [],
        "eval_mean_utility": [],
        "eval_std_utility": [],
        "eval_mean_wealth": [],
        "eval_std_wealth": [],
    }

    for episode in range(episodes):
        states, raw_actions, rewards, infos, episode_return, terminal_wealth = run_episode(
            env, policy, max_turnover=max_turnover
        )

        returns = [episode_return for _ in states]
        baseline_loss = baseline.fit_batch(
            states,
            returns,
            lr=baseline_lr,
            epochs=baseline_epochs,
        )

        grad_W_sum = np.zeros_like(policy.W)
        grad_b_sum = np.zeros_like(policy.b)
        for state, action in zip(states, raw_actions):
            advantage = episode_return - baseline.predict(state)
            grad_W, grad_b = policy.grad_log_prob(state, action)
            grad_W_sum += advantage * grad_W
            grad_b_sum += advantage * grad_b

        grad_W_sum /= len(states)
        grad_b_sum /= len(states)
        policy.W += policy_lr * grad_W_sum
        policy.b += policy_lr * grad_b_sum

        history["train_utility"].append(episode_return)
        history["train_terminal_wealth"].append(terminal_wealth)
        history["policy_norm"].append(policy.param_norm())
        history["value_norm"].append(baseline.param_norm())
        history["value_loss"].append(baseline_loss)

        if (episode + 1) % eval_every == 0:
            mean_u, std_u, mean_w, std_w = evaluate_policy(
                env_config,
                policy,
                n_eval=eval_paths,
                max_turnover=max_turnover,
                seed=999,
            )
            history["eval_episode"].append(episode + 1)
            history["eval_mean_utility"].append(mean_u)
            history["eval_std_utility"].append(std_u)
            history["eval_mean_wealth"].append(mean_w)
            history["eval_std_wealth"].append(std_w)

            print(
                f"[{baseline_type}] Episode {episode + 1:4d} | "
                f"Train utility (last100 mean) = {np.mean(history['train_utility'][-100:]):.6f} | "
                f"Baseline loss = {np.mean(history['value_loss'][-100:]):.6f} | "
                f"Eval mean utility = {mean_u:.6f} | "
                f"Eval mean wealth = {mean_w:.4f}"
            )

    return policy, baseline, history
