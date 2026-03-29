import matplotlib.pyplot as plt
import numpy as np


def cara_utility(wealth, risk_aversion):
    return -np.exp(-risk_aversion * wealth)


def project_to_simplex(vector):
    vector = np.asarray(vector, dtype=float)
    size = len(vector)
    sorted_vector = np.sort(vector)[::-1]
    cumulative_sum = np.cumsum(sorted_vector)
    rho = np.nonzero(sorted_vector * np.arange(1, size + 1) > (cumulative_sum - 1))[0][-1]
    theta = (cumulative_sum[rho] - 1) / (rho + 1.0)
    return np.maximum(vector - theta, 0.0)


def enforce_turnover_constraint(old_weights, new_weights, max_turnover=0.10):
    diff = new_weights - old_weights
    l1_norm = np.sum(np.abs(diff))
    if l1_norm <= max_turnover:
        return new_weights
    alpha = max_turnover / (l1_norm + 1e-12)
    return old_weights + alpha * diff


def valid_portfolio_update(old_weights, raw_delta, max_turnover=0.10):
    delta = raw_delta - np.mean(raw_delta)
    l1_norm = np.sum(np.abs(delta))
    if l1_norm > max_turnover:
        delta = delta * (max_turnover / (l1_norm + 1e-12))

    candidate = project_to_simplex(old_weights + delta)
    candidate = enforce_turnover_constraint(old_weights, candidate, max_turnover=max_turnover)
    return project_to_simplex(candidate)


def apply_turnover_toward_target(old_weights, target_weights, max_turnover=0.10):
    old_weights = np.asarray(old_weights, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)

    target_weights = np.clip(target_weights, 0.0, 1.0)
    target_weights = target_weights / (np.sum(target_weights) + 1e-12)

    diff = target_weights - old_weights
    l1_norm = np.sum(np.abs(diff))

    if l1_norm <= max_turnover + 1e-12:
        new_weights = target_weights.copy()
    else:
        alpha = max_turnover / (l1_norm + 1e-12)
        new_weights = old_weights + alpha * diff

    new_weights = np.clip(new_weights, 0.0, 1.0)
    return new_weights / (np.sum(new_weights) + 1e-12)


class AssetAllocationEnv:
    def __init__(
        self,
        means,
        variances,
        cash_rate,
        horizon,
        init_weights,
        init_wealth=1.0,
        risk_aversion=1.0,
        seed=None,
    ):
        self.means = np.array(means, dtype=float)
        self.variances = np.array(variances, dtype=float)
        self.stds = np.sqrt(self.variances)
        self.n_risky = len(means)
        self.n_assets_total = self.n_risky + 1
        self.r = cash_rate
        self.T = horizon
        self.init_weights = np.array(init_weights, dtype=float)
        self.init_wealth = init_wealth
        self.risk_aversion = risk_aversion
        self.rng = np.random.default_rng(seed)

        assert len(self.init_weights) == self.n_assets_total
        assert np.isclose(np.sum(self.init_weights), 1.0)
        assert np.all(self.init_weights >= 0)
        assert self.T < 10
        assert self.n_risky < 5

        self.reset()

    def reset(self):
        self.t = 0
        self.wealth = self.init_wealth
        self.weights = self.init_weights.copy()
        return self._state()

    def _state(self):
        return np.concatenate(([self.t / self.T, np.log(max(self.wealth, 1e-8))], self.weights))

    def step(self, new_weights):
        new_weights = np.asarray(new_weights, dtype=float)

        risky_returns = self.rng.normal(self.means, self.stds)
        gross_returns = np.concatenate(([1.0 + self.r], 1.0 + risky_returns))

        portfolio_gross = np.dot(new_weights, gross_returns)
        next_wealth = self.wealth * portfolio_gross

        next_values = self.wealth * new_weights * gross_returns
        next_weights = next_values / np.sum(next_values)

        self.wealth = next_wealth
        self.weights = next_weights
        self.t += 1

        done = self.t == self.T
        reward = cara_utility(self.wealth, self.risk_aversion) if done else 0.0

        return self._state(), reward, done, {
            "wealth": self.wealth,
            "weights": self.weights.copy(),
            "risky_returns": risky_returns,
            "gross_returns": gross_returns,
        }


def build_static_env_config(env):
    return {
        "means": env.means.copy(),
        "variances": env.variances.copy(),
        "cash_rate": env.r,
        "horizon": env.T,
        "init_weights": env.init_weights.copy(),
        "init_wealth": env.init_wealth,
        "risk_aversion": env.risk_aversion,
    }


def build_static_env_from_config(env_config, seed):
    return AssetAllocationEnv(
        means=env_config["means"],
        variances=env_config["variances"],
        cash_rate=env_config["cash_rate"],
        horizon=env_config["horizon"],
        init_weights=env_config["init_weights"],
        init_wealth=env_config["init_wealth"],
        risk_aversion=env_config["risk_aversion"],
        seed=seed,
    )


def moving_average(values, window=100):
    values = np.asarray(values)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_training_history(history, ma_window=100, value_label="Value", title_prefix=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    prefix = title_prefix or f"{value_label}: {history.get('value_type', 'unknown')}"

    axes[0, 0].plot(history["train_utility"], alpha=0.25, label="Episode utility")
    ma_utility = moving_average(history["train_utility"], ma_window)
    if len(ma_utility) > 0:
        x_ma = np.arange(len(ma_utility)) + (ma_window - 1 if len(history["train_utility"]) >= ma_window else 0)
        axes[0, 0].plot(x_ma, ma_utility, linewidth=2, label=f"Moving avg ({ma_window})")
    axes[0, 0].set_title(f"{prefix} - Training terminal utility")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Utility")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["train_terminal_wealth"], alpha=0.25, label="Episode terminal wealth")
    ma_wealth = moving_average(history["train_terminal_wealth"], ma_window)
    if len(ma_wealth) > 0:
        x_ma = np.arange(len(ma_wealth)) + (ma_window - 1 if len(history["train_terminal_wealth"]) >= ma_window else 0)
        axes[0, 1].plot(x_ma, ma_wealth, linewidth=2, label=f"Moving avg ({ma_window})")
    axes[0, 1].set_title(f"{prefix} - Training terminal wealth")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Terminal wealth")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    eval_episode = np.array(history["eval_episode"])
    eval_mean_utility = np.array(history["eval_mean_utility"])
    eval_std_utility = np.array(history["eval_std_utility"])
    if len(eval_episode) > 0:
        axes[1, 0].plot(eval_episode, eval_mean_utility, marker="o", label="Eval mean utility")
        axes[1, 0].fill_between(
            eval_episode,
            eval_mean_utility - eval_std_utility,
            eval_mean_utility + eval_std_utility,
            alpha=0.2,
        )
    axes[1, 0].set_title(f"{prefix} - Fixed-seed evaluation utility")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Mean utility")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history["policy_norm"], label="Policy parameter norm")
    axes[1, 1].plot(history["value_norm"], label=f"{value_label} parameter norm")
    axes[1, 1].set_title(f"{prefix} - Parameter norms")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Norm")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_eval_wealth(history, value_label="Value", title_prefix=None):
    plt.figure(figsize=(7, 5))
    prefix = title_prefix or f"{value_label}: {history.get('value_type', 'unknown')}"
    eval_episode = np.array(history["eval_episode"])
    eval_mean_wealth = np.array(history["eval_mean_wealth"])
    eval_std_wealth = np.array(history["eval_std_wealth"])
    if len(eval_episode) > 0:
        plt.plot(eval_episode, eval_mean_wealth, marker="o", label="Eval mean terminal wealth")
        plt.fill_between(
            eval_episode,
            eval_mean_wealth - eval_std_wealth,
            eval_mean_wealth + eval_std_wealth,
            alpha=0.2,
        )
    plt.title(f"{prefix} - Fixed-seed evaluation terminal wealth")
    plt.xlabel("Episode")
    plt.ylabel("Mean terminal wealth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
